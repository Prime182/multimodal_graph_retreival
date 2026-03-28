"""External biomedical corpus integration for entity enrichment.

PRAGMATIC APPROACH:
- Uses authoritative known biomedical terms as primary source (never fails)
- Queries real APIs (NCBI) for genes where possible with robust XML parsing
- Provides reliable, tested aliases and hierarchies
- Gracefully handles network failures
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
import requests
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Optional

from .circuit_breaker import CircuitBreakerOpenError, get_circuit_breaker
from .domain_config import get_domain_knowledge


# API endpoints
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OLS_BASE = "https://www.ebi.ac.uk/ols4/api"
CORPUS_API_TIMEOUT = float(os.getenv("CORPUS_API_TIMEOUT_SEC", "3"))

# Rate limits (requests per second)
NCBI_DELAY = 0.4  # Stay within 3 req/sec free-tier limit (conservative)
OLS_DELAY = 0.2

# Last request timestamp for rate limiting
_last_request_time: dict[str, float] = {"ncbi": 0.0, "ols": 0.0}
_DOMAIN_KNOWLEDGE = get_domain_knowledge()
_CORPUS_CONFIG = _DOMAIN_KNOWLEDGE.get("corpus", {})


def _rate_limit(service: str, delay: float) -> None:
    """Enforce rate limiting for API calls."""
    now = time.time()
    last = _last_request_time.get(service, 0.0)
    sleep_time = delay - (now - last)
    if sleep_time > 0:
        time.sleep(sleep_time)
    _last_request_time[service] = time.time()


@dataclass
class CorpusMatch:
    """Result from corpus lookup."""
    found: bool
    label: str
    aliases: list[str] = field(default_factory=list)
    ontology: Optional[str] = None
    external_id: Optional[str] = None
    definition: Optional[str] = None
    category: Optional[str] = None
    parent_terms: list[str] = field(default_factory=list)


def _ols_known_terms() -> dict[str, dict[str, dict[str, object]]]:
    raw_terms = _CORPUS_CONFIG.get("ols_known_terms", {})
    return {
        str(ontology): {
            str(term): {
                "label": str(values.get("label", term)),
                "aliases": [str(alias) for alias in values.get("aliases", [])],
                "external_id": str(values.get("external_id", "")),
            }
            for term, values in term_map.items()
        }
        for ontology, term_map in raw_terms.items()
    }


def _known_ols_match(query: str, ontology: str) -> Optional[CorpusMatch]:
    query_norm = query.strip().lower()
    ont_lower = ontology.lower()
    known_terms = _ols_known_terms()
    term_dict = known_terms.get(ont_lower, {})
    if query_norm not in term_dict:
        return None
    term_info = term_dict[query_norm]
    return CorpusMatch(
        found=True,
        label=str(term_info["label"]),
        aliases=[str(alias) for alias in term_info.get("aliases", [])],
        ontology=ontology.upper(),
        external_id=str(term_info["external_id"]),
    )


def _coerce_alias_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _lookup_ols_api(query: str, ontology: str) -> Optional[CorpusMatch]:
    breaker = get_circuit_breaker()
    try:
        breaker.guard("ols")
    except CircuitBreakerOpenError:
        return None

    try:
        _rate_limit("ols", OLS_DELAY)
        response = requests.get(
            f"{OLS_BASE}/search",
            params={
                "q": query,
                "ontology": ontology.lower(),
                "exact": "true",
                "rows": 1,
            },
            timeout=CORPUS_API_TIMEOUT,
        )
        if response.status_code != 200:
            breaker.record_failure("ols")
            return None

        payload = response.json()
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            breaker.record_success("ols")
            return None

        doc = docs[0]
        label = str(doc.get("label") or doc.get("short_form") or query).strip()
        if not label:
            breaker.record_success("ols")
            return None

        description_raw = doc.get("description")
        description = None
        if isinstance(description_raw, list):
            description = next((str(item).strip() for item in description_raw if str(item).strip()), None)
        elif isinstance(description_raw, str) and description_raw.strip():
            description = description_raw.strip()

        external_id = doc.get("obo_id") or doc.get("short_form") or doc.get("iri")
        aliases = _coerce_alias_list(doc.get("synonym"))
        category = str(doc.get("type") or "").strip() or None
        breaker.record_success("ols")
        return CorpusMatch(
            found=True,
            label=label,
            aliases=aliases,
            ontology=str(doc.get("ontology_name") or ontology).upper(),
            external_id=str(external_id) if external_id else None,
            definition=description,
            category=category,
        )
    except Exception:
        breaker.record_failure("ols")
        return None


def _cache_key(label: str, entity_type: str, preferred_corpus: str | None) -> str:
    return "::".join(
        [
            entity_type.strip().lower(),
            label.strip().lower(),
            (preferred_corpus or "").strip().lower(),
        ]
    )


class CorpusClient:
    """SQLite-cached corpus lookup client."""

    def __init__(self, cache_path: str | os.PathLike[str] | None = None) -> None:
        self.cache_path = Path(
            cache_path
            or os.getenv("CORPUS_CACHE_PATH")
            or ".cache/graphrag_corpus.sqlite3"
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_misses (
                label TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                preferred_corpus TEXT NOT NULL,
                miss_count INTEGER NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (label, entity_type, preferred_corpus)
            )
            """
        )
        self._conn.commit()

    def _read_cache(self, cache_key: str) -> CorpusMatch | None:
        row = self._conn.execute(
            "SELECT payload_json FROM corpus_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return CorpusMatch(**payload)

    def _write_cache(self, cache_key: str, match: CorpusMatch) -> None:
        self._conn.execute(
            """
            INSERT INTO corpus_cache (cache_key, payload_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (cache_key, json.dumps(asdict(match)), time.time()),
        )
        self._conn.commit()

    def _record_miss(self, label: str, entity_type: str, preferred_corpus: Optional[str]) -> None:
        self._conn.execute(
            """
            INSERT INTO corpus_misses (label, entity_type, preferred_corpus, miss_count, last_seen)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(label, entity_type, preferred_corpus) DO UPDATE SET
                miss_count = corpus_misses.miss_count + 1,
                last_seen = excluded.last_seen
            """,
            (
                label.strip(),
                entity_type.strip().lower(),
                (preferred_corpus or "").strip().lower(),
                time.time(),
            ),
        )
        self._conn.commit()

    def list_misses(self, limit: int = 100) -> list[dict[str, object]]:
        cursor = self._conn.execute(
            """
            SELECT label, entity_type, preferred_corpus, miss_count, last_seen
            FROM corpus_misses
            ORDER BY miss_count DESC, last_seen DESC
            LIMIT ?
            """,
            (limit,),
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def enrich_entity(
        self,
        label: str,
        entity_type: str,
        preferred_corpus: Optional[str] = None,
    ) -> CorpusMatch:
        cache_key = _cache_key(label, entity_type, preferred_corpus)
        cached = self._read_cache(cache_key)
        if cached is not None:
            if not cached.found:
                self._record_miss(label, entity_type, preferred_corpus)
            return cached

        result = _enrich_entity_uncached(
            label=label,
            entity_type=entity_type,
            preferred_corpus=preferred_corpus,
        )
        if not result.found:
            self._record_miss(label, entity_type, preferred_corpus)
        self._write_cache(cache_key, result)
        return result


def lookup_mesh(query: str, search_type: str = "descriptor") -> Optional[CorpusMatch]:
    """Query MeSH terms - kept for compatibility, uses known terms.
    
    Args:
        query: Search term
        search_type: 'descriptor' or 'qualifier' (for compatibility)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    # For now, use OLS which has reliable known mappings
    return lookup_ols(query, "go")


def lookup_ols(query: str, ontology: str = "chebi") -> Optional[CorpusMatch]:
    """Look up biomedical terms in ontologies with API-first fallback behavior.
    
    Args:
        query: Search term
        ontology: Ontology ('go', 'chebi', 'doid', 'hp', etc.)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        api_match = _lookup_ols_api(query, ontology)
        if api_match is not None:
            return api_match
    except Exception:
        pass

    return _known_ols_match(query, ontology)


def lookup_cellosaurus(query: str) -> Optional[CorpusMatch]:
    """Look up cell line information using known cell lines.
    
    Args:
        query: Cell line name
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        known_cell_lines = _CORPUS_CONFIG.get("cell_lines", {})
        query_norm = query.strip().lower()
        if query_norm in known_cell_lines:
            info = known_cell_lines[query_norm]
            return CorpusMatch(
                found=True,
                label=str(info["label"]),
                aliases=[str(alias) for alias in info.get("aliases", [])],
                ontology="Cellosaurus",
                category=str(info["category"]),
            )
        return None
    except Exception:
        return None


def lookup_ncbi_gene(query: str, organism: str = "Homo sapiens") -> Optional[CorpusMatch]:
    """Query NCBI Gene database for gene/protein information.
    
    Uses real API with XML parsing for reliable gene lookups.
    
    Args:
        query: Gene name or symbol
        organism: Organism name (default: human)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    breaker = get_circuit_breaker()
    try:
        breaker.guard("ncbi")
    except CircuitBreakerOpenError:
        return None

    try:
        _rate_limit("ncbi", NCBI_DELAY)
        
        # Search for the gene using XML format
        search_url = f"{NCBI_BASE}/esearch.fcgi"
        params = {
            "db": "gene",
            "term": f'"{query}"[GENE] AND {organism}[ORGN]',
            "retmax": 1,
        }
        response = requests.get(search_url, params=params, timeout=CORPUS_API_TIMEOUT)
        
        if response.status_code != 200:
            breaker.record_failure("ncbi")
            return None
        
        # Parse XML response
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            breaker.record_failure("ncbi")
            return None
        
        count_elem = root.find("Count")
        count = int(count_elem.text) if count_elem is not None else 0
        
        if count == 0:
            breaker.record_success("ncbi")
            return None
        
        id_list = root.find("IdList")
        if id_list is None:
            breaker.record_failure("ncbi")
            return None
        
        ids = [elem.text for elem in id_list.findall("Id")]
        if not ids:
            breaker.record_failure("ncbi")
            return None
        
        gene_id = ids[0]
        
        # Fetch gene summary
        summary_url = f"{NCBI_BASE}/esummary.fcgi"
        summary_params = {"db": "gene", "id": gene_id}
        summary_response = requests.get(summary_url, params=summary_params, timeout=CORPUS_API_TIMEOUT)
        
        if summary_response.status_code != 200:
            breaker.record_success("ncbi")
            return CorpusMatch(found=True, label=query, aliases=[], ontology="NCBI-Gene", external_id=gene_id)
        
        # Parse summary XML
        try:
            root_summary = ET.fromstring(summary_response.text)
        except ET.ParseError:
            breaker.record_success("ncbi")
            return CorpusMatch(found=True, label=query, aliases=[], ontology="NCBI-Gene", external_id=gene_id)
        
        doc_sum = root_summary.find(".//DocSum")
        if doc_sum is None:
            breaker.record_success("ncbi")
            return CorpusMatch(found=True, label=query, aliases=[], ontology="NCBI-Gene", external_id=gene_id)
        
        # Extract metadata
        label = query
        description = ""
        for item in doc_sum.findall("Item"):
            name = item.get("Name", "")
            if name == "Name":
                label = item.text or query
            elif name == "Description":
                description = item.text or ""
        
        breaker.record_success("ncbi")
        return CorpusMatch(
            found=True,
            label=label,
            aliases=[],
            ontology="NCBI-Gene",
            external_id=gene_id,
            definition=description,
            category="gene",
        )
    except Exception:
        breaker.record_failure("ncbi")
        return None


def _enrich_entity_uncached(
    label: str,
    entity_type: str,
    preferred_corpus: Optional[str] = None,
) -> CorpusMatch:
    """Enrich entity by querying appropriate corpora.
    
    Never fails - returns CorpusMatch(found=False) if not found.
    
    Args:
        label: Entity label to enrich
        entity_type: 'method', 'concept', 'dataset', 'gene', etc.
        preferred_corpus: Preferred corpus to try first
    
    Returns:
        CorpusMatch with enrichment data (found=False if not found)
    """
    
    if entity_type == "dataset":
        # Try cell line database
        result = lookup_cellosaurus(label)
        if result:
            return result
    
    elif entity_type == "gene":
        # Try NCBI Gene (real API)
        result = lookup_ncbi_gene(label)
        if result:
            return result
    
    elif entity_type == "method":
        for ontology in ["obi", "efo", "go"]:
            result = lookup_ols(label, ontology)
            if result:
                return result

    elif entity_type == "concept":
        for ontology in ["go", "chebi", "doid", "efo"]:
            result = lookup_ols(label, ontology)
            if result:
                return result
    
    # Return not-found result
    return CorpusMatch(found=False, label=label, aliases=[])


_default_client: CorpusClient | None = None


def get_corpus_client() -> CorpusClient:
    global _default_client
    if _default_client is None:
        _default_client = CorpusClient()
    return _default_client


def enrich_entity(
    label: str,
    entity_type: str,
    preferred_corpus: Optional[str] = None,
) -> CorpusMatch:
    return get_corpus_client().enrich_entity(
        label=label,
        entity_type=entity_type,
        preferred_corpus=preferred_corpus,
    )


def get_hierarchy(term: str, ontology: str = "go") -> list[str]:
    """Get parent terms (ontological hierarchy).
    
    Uses known, reliable hierarchies rather than external APIs.
    
    Args:
        term: Search term
        ontology: Ontology to search
    
    Returns:
        List of parent term labels
    """
    try:
        ont_lower = ontology.lower()
        term_lower = term.lower().strip()
        known_hierarchies = _CORPUS_CONFIG.get("hierarchies", {})
        if ont_lower in known_hierarchies:
            hierarchy_dict = known_hierarchies[ont_lower]
            if term_lower in hierarchy_dict:
                return [str(item) for item in hierarchy_dict[term_lower]]
        
        return []
    except Exception:
        return []
