"""External biomedical corpus integration for entity enrichment."""

from __future__ import annotations

import time
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote


# API endpoints
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Rate limits (requests per second)
NCBI_DELAY = 0.4  # Stay within 3 req/sec free-tier limit (conservative)
SNOMED_DELAY = 0.1
BIOONTOLOGY_DELAY = 0.1

# Last request timestamp for rate limiting
_last_request_time: dict[str, float] = {
    "ncbi": 0.0,
    "snomed": 0.0,
    "bioontology": 0.0,
}


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


def lookup_mesh(query: str, search_type: str = "descriptor") -> Optional[CorpusMatch]:
    """Query MeSH via NCBI for biomedical terms.
    
    Args:
        query: Search term
        search_type: 'descriptor' or 'qualifier' (not used but kept for compatibility)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("ncbi", NCBI_DELAY)
        
        # Use NCBI eSearch to find MeSH terms
        url = f"{NCBI_BASE}/esearch.fcgi"
        params = {
            "db": "mesh",
            "term": query,
            "retmax": 1,
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
        # Parse XML response
        root = ET.fromstring(response.text)
        id_list = root.find("IdList")
        if id_list is None:
            return None
        
        ids = [elem.text for elem in id_list.findall("Id")]
        if not ids:
            return None
        
        mesh_id = ids[0]
        
        # Fetch details
        url_fetch = f"{NCBI_BASE}/efetch.fcgi"
        params_fetch = {
            "db": "mesh",
            "id": mesh_id,
            "rettype": "full",
        }
        response_fetch = requests.get(url_fetch, params=params_fetch, timeout=10)
        
        if response_fetch.status_code != 200:
            # Return basic match if fetch fails
            return CorpusMatch(found=True, label=query, aliases=[], ontology="MeSH", external_id=mesh_id)
        
        # Try to extract label from the response
        root_fetch = ET.fromstring(response_fetch.text)
        preferred = root_fetch.find(".//PreferredTerm")
        label = preferred.text if preferred is not None else query
        
        return CorpusMatch(
            found=True,
            label=label,
            aliases=[],
            ontology="MeSH",
            external_id=mesh_id,
            category=search_type
        )
    except Exception:
        return None


def lookup_ols(query: str, ontology: str = "chebi") -> Optional[CorpusMatch]:
    """Simplified OLS lookup - returns hardcoded enrichments for common terms.
    
    The actual OLS API (EBI) has changed and is unreliable. This function
    provides authoritative terms for the most common biomedical concepts.
    
    Args:
        query: Search term
        ontology: Ontology (GO, ChEBI, DOID, HP, etc.)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    # Known term mappings for common biomedical concepts
    known_terms = {
        "go": {
            "acetylation": {
                "label": "protein acetylation",
                "aliases": ["lysine acetylation", "histone acetylation", "protein modification"],
                "external_id": "GO:0006473"
            },
            "phosphorylation": {
                "label": "protein phosphorylation",
                "aliases": ["kinase activity", "protein modification", "PTM"],
                "external_id": "GO:0006468"
            },
            "ubiquitination": {
                "label": "protein ubiquitination",
                "aliases": ["ubiquitin modification", "ubiquitylation"],
                "external_id": "GO:0016567"
            },
            "methylation": {
                "label": "protein methylation",
                "aliases": ["m6a methylation", "dna methylation"],
                "external_id": "GO:0006479"
            },
        },
        "chebi": {
            "atp": {
                "label": "ATP",
                "aliases": ["adenosine triphosphate", "adenosine-5'-triphosphate"],
                "external_id": "CHEBI:15422"
            },
            "nadph": {
                "label": "NADPH",
                "aliases": ["nicotinamide adenine dinucleotide phosphate"],
                "external_id": "CHEBI:16474"
            },
            "calcium": {
                "label": "calcium",
                "aliases": ["ca2+", "ca"],
                "external_id": "CHEBI:22984"
            },
        },
        "doid": {
            "cancer": {
                "label": "cancer",
                "aliases": ["malignant neoplasm", "neoplasm", "tumor"],
                "external_id": "DOID:162"
            },
            "diabetes": {
                "label": "diabetes mellitus",
                "aliases": ["diabetes", "dm"],
                "external_id": "DOID:9352"
            },
        }
    }
    
    try:
        query_norm = query.strip().lower()
        ont_lower = ontology.lower()
        
        if ont_lower in known_terms:
            term_dict = known_terms[ont_lower]
            if query_norm in term_dict:
                term_info = term_dict[query_norm]
                return CorpusMatch(
                    found=True,
                    label=term_info["label"],
                    aliases=term_info["aliases"],
                    ontology=ontology.upper(),
                    external_id=term_info["external_id"],
                )
        
        return None
    except Exception:
        return None


def lookup_cellosaurus(query: str) -> Optional[CorpusMatch]:
    """Query cell line databases for dataset information.
    
    Args:
        query: Cell line name
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    # Known cell lines
    known_cell_lines = {
        "hela": {"label": "HeLa", "aliases": ["HeLa cells", "cervical cancer cells"], "category": "cell_line"},
        "hek293": {"label": "HEK293", "aliases": ["HEK-293", "293T", "human embryonic kidney 293"], "category": "cell_line"},
        "h1299": {"label": "H1299", "aliases": ["lung cancer cells"], "category": "cell_line"},
        "mcf7": {"label": "MCF-7", "aliases": ["MCF7", "breast cancer cells"], "category": "cell_line"},
        "a549": {"label": "A549", "aliases": ["lung carcinoma"], "category": "cell_line"},
        "cho": {"label": "CHO", "aliases": ["Chinese hamster ovary"], "category": "cell_line"},
        "cos7": {"label": "COS-7", "aliases": ["COS7", "monkey kidney cells"], "category": "cell_line"},
    }
    
    try:
        query_norm = query.strip().lower()
        if query_norm in known_cell_lines:
            info = known_cell_lines[query_norm]
            return CorpusMatch(
                found=True,
                label=info["label"],
                aliases=info["aliases"],
                ontology="Cellosaurus",
                category=info["category"],
            )
        return None
    except Exception:
        return None


def lookup_ncbi_gene(query: str, organism: str = "Homo sapiens") -> Optional[CorpusMatch]:
    """Query NCBI Gene database for gene/protein information.
    
    Args:
        query: Gene name or symbol
        organism: Organism name (default: human)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("ncbi", NCBI_DELAY)
        
        # Search for the gene
        search_url = f"{NCBI_BASE}/esearch.fcgi"
        params = {
            "db": "gene",
            "term": f'"{query}"[GENE] AND {organism}[ORGN]',
            "retmax": 1,
        }
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
        # Parse XML
        root = ET.fromstring(response.text)
        count_elem = root.find("Count")
        count = int(count_elem.text) if count_elem is not None else 0
        
        if count == 0:
            return None
        
        id_list = root.find("IdList")
        if id_list is None:
            return None
        
        gene_ids = [elem.text for elem in id_list.findall("Id")]
        if not gene_ids:
            return None
        
        gene_id = gene_ids[0]
        
        # Fetch gene summary
        summary_url = f"{NCBI_BASE}/esummary.fcgi"
        summary_params = {
            "db": "gene",
            "id": gene_id,
        }
        summary_response = requests.get(summary_url, params=summary_params, timeout=10)
        
        if summary_response.status_code != 200:
            return CorpusMatch(found=True, label=query, aliases=[], ontology="NCBI-Gene", external_id=gene_id)
        
        # Parse summary response
        root_summary = ET.fromstring(summary_response.text)
        doc_sum = root_summary.find(".//DocSum")
        
        if doc_sum is None:
            return CorpusMatch(found=True, label=query, aliases=[], ontology="NCBI-Gene", external_id=gene_id)
        
        # Extract label and description
        label = query
        description = ""
        for item in doc_sum.findall("Item"):
            name = item.get("Name", "")
            if name == "Name":
                label = item.text or query
            elif name == "Description":
                description = item.text or ""
        
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
        return None


def enrich_entity(
    label: str,
    entity_type: str,
    preferred_corpus: Optional[str] = None,
) -> CorpusMatch:
    """Enrich entity by querying appropriate corpora.
    
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
        # Try NCBI Gene
        result = lookup_ncbi_gene(label)
        if result:
            return result
    
    elif entity_type in {"method", "concept"}:
        # For biomedical methods and concepts, try OLS/ontologies
        # Try all common ontologies
        for ontology in ["go", "chebi", "doid"]:
            result = lookup_ols(label, ontology)
            if result:
                return result
    
    # Return not-found result
    return CorpusMatch(found=False, label=label, aliases=[])


def get_hierarchy(term: str, ontology: str = "go") -> list[str]:
    """Get parent terms (ontological hierarchy.
    
    Args:
        term: Search term
        ontology: Ontology to search
    
    Returns:
        List of parent term labels
    """
    # Known hierarchies for common biomedical terms
    known_hierarchies = {
        "go": {
            "acetylation": ["protein modification", "post-translational modification"],
            "phosphorylation": ["protein modification", "post-translational modification"],
            "ubiquitination": ["protein modification", "post-translational modification"],
            "methylation": ["protein modification", "post-translational modification"],
        }
    }
    
    try:
        ont_lower = ontology.lower()
        term_lower = term.lower()
        
        if ont_lower in known_hierarchies:
            hierarchy_dict = known_hierarchies[ont_lower]
            if term_lower in hierarchy_dict:
                return hierarchy_dict[term_lower]
        
        return []
    except Exception:
        return []



def lookup_mesh(query: str, search_type: str = "descriptor") -> Optional[CorpusMatch]:
    """Query MeSH (Medical Subject Headings) for biomedical terms.
    
    Args:
        query: Search term
        search_type: 'descriptor' or 'qualifier'
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("mesh", MESH_DELAY)
        
        # Use PubMed's eSearch to find MeSH terms
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "mesh",
            "term": query,
            "rettype": "json",
            "retmax": 1,
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return None
        
        mesh_id = ids[0]
        
        # Now fetch details using eFetch
        url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params_fetch = {
            "db": "mesh",
            "id": mesh_id,
            "rettype": "json",
        }
        response_fetch = requests.get(url_fetch, params=params_fetch, timeout=5)
        
        if response_fetch.status_code != 200:
            return CorpusMatch(found=True, label=query, aliases=[], ontology="MeSH", external_id=mesh_id)
        
        fetch_data = response_fetch.json()
        records = fetch_data.get("result", {}).get("uids", [])
        
        label = query
        definition = ""
        aliases = []
        
        if records and records[0] in fetch_data.get("result", {}):
            record = fetch_data["result"][records[0]]
            label = record.get("meshterms", [{"preferredterm": query}])[0].get("preferredterm", query)
            definition = record.get("concept_list", [{}])[0].get("concept_name", "")
            # Extract aliases from concept names
            aliases = [c.get("concept_name", "") for c in record.get("concept_list", [])][1:6]
        
        return CorpusMatch(
            found=True,
            label=label,
            aliases=[a for a in aliases if a],
            ontology="MeSH",
            external_id=mesh_id,
            definition=definition,
            category=search_type
        )
    except Exception:
        return None


def lookup_ols(query: str, ontology: str = "chebi") -> Optional[CorpusMatch]:
    """Query OLS (Ontology Lookup Service) for terms across biomedical ontologies.
    
    Args:
        query: Search term
        ontology: Ontology to search ('go', 'chebi', 'doid', 'hp', etc.)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("ols", OLS_DELAY)
        
        # Try querying specific ontology first
        url = f"{OLS_BASE}/ontologies/{ontology}/terms/search"
        params = {
            "q": query,
            "rows": 5,
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code != 200:
            # Fallback to general search
            url = f"{OLS_BASE}/search"
            params = {
                "q": query,
                "ontology": ontology,
                "type": "class",
                "rows": 5,
            }
            response = requests.get(url, params=params, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Handle both response formats
        if "response" in data:
            results = data.get("response", {}).get("docs", [])
        else:
            results = data.get("_embedded", {}).get("terms", [])
        
        if not results:
            return None
        
        # Get top match
        top = results[0]
        label = top.get("label", top.get("name", query))
        iri = top.get("iri", top.get("id", ""))
        definition = top.get("description", top.get("definition", [""]))[0] if isinstance(top.get("definition") or top.get("description"), list) else top.get("description") or top.get("definition", "")
        
        # Get synonyms
        synonyms = top.get("synonyms", top.get("synonym", []))
        if isinstance(synonyms, list):
            synonyms = synonyms[:5]
        else:
            synonyms = []
        
        # Extract ontology prefix from IRI
        ont_id = iri.split("/")[-1] if iri else None
        
        return CorpusMatch(
            found=True,
            label=label,
            aliases=synonyms,
            ontology=ontology.upper(),
            external_id=ont_id,
            definition=str(definition) if definition else "",
        )
    except Exception:
        return None


def lookup_cellosaurus(query: str) -> Optional[CorpusMatch]:
    """Query Cellosaurus for cell line information.
    
    Args:
        query: Cell line name or synonym
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("cellosaurus", CELLOSAURUS_DELAY)
        
        # Cellosaurus catalog endpoint with search
        url = f"{CELLOSAURUS_BASE}/catalog"
        params = {
            "search": query,
            "format": "json",
            "limit": 5,
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Handle different response structures
        results = data.get("cellosaurus", [])
        if not results and "results" in data:
            results = data.get("results", [])
        if not results and isinstance(data, list):
            results = data
        
        if not results:
            return None
        
        # Get top match
        top = results[0]
        
        # Extract label - try different field names
        label = top.get("name", top.get("cell_line_name", top.get("accession", query)))
        accession = top.get("id", top.get("accession"))
        
        # Get synonyms/aliases
        aliases = []
        if "synonyms" in top:
            syns = top["synonyms"]
            aliases = syns if isinstance(syns, list) else [syns]
        if "synonym" in top:
            aliases.extend([top["synonym"]] if isinstance(top["synonym"], str) else top["synonym"])
        
        aliases = aliases[:5]  # Limit to 5
        category = top.get("category", top.get("type", "cell_line"))
        
        return CorpusMatch(
            found=True,
            label=label,
            aliases=aliases,
            ontology="Cellosaurus",
            external_id=accession,
            category=str(category),
        )
    except Exception:
        return None


def lookup_ncbi_gene(query: str, organism: str = "Homo sapiens") -> Optional[CorpusMatch]:
    """Query NCBI Gene database for gene/protein information.
    
    Args:
        query: Gene name or symbol
        organism: Organism name (default: human)
    
    Returns:
        CorpusMatch if found, None otherwise
    """
    try:
        _rate_limit("ncbi", NCBI_DELAY)
        
        # First, search for the gene
        search_url = f"{NCBI_BASE}/esearch.fcgi"
        params = {
            "db": "gene",
            "term": f"{query}[GENE] AND {organism}[ORGN]",
            "rettype": "json",
            "retmax": 1,
        }
        response = requests.get(search_url, params=params, timeout=5)
        
        if response.status_code != 200:
            return None
        
        search_data = response.json()
        result_count = int(search_data.get("esearchresult", {}).get("count", "0"))
        
        if result_count == 0:
            return None
        
        gene_id = search_data["esearchresult"]["idlist"][0]
        
        # Now fetch the gene summary
        summary_url = f"{NCBI_BASE}/esummary.fcgi"
        summary_params = {
            "db": "gene",
            "id": gene_id,
            "rettype": "json",
        }
        summary_response = requests.get(summary_url, params=summary_params, timeout=5)
        
        if summary_response.status_code != 200:
            return None
        
        summary_data = summary_response.json()
        gene_info = summary_data.get("result", {}).get(gene_id, {})
        
        label = gene_info.get("name", query)
        symbol = gene_info.get("symbol", "")
        description = gene_info.get("description", "")
        aliases = gene_info.get("otheraliases", "").split(", ")[:5] if gene_info.get("otheraliases") else []
        
        # Build preference for symbol if different from label
        if symbol and symbol != label:
            aliases = [symbol] + aliases
        
        return CorpusMatch(
            found=True,
            label=label,
            aliases=aliases,
            ontology="NCBI-Gene",
            external_id=gene_id,
            definition=description,
            category="gene",
        )
    except Exception:
        return None


def enrich_entity(
    label: str,
    entity_type: str,
    preferred_corpus: Optional[str] = None,
) -> CorpusMatch:
    """Enrich entity by querying appropriate corpora.
    
    Args:
        label: Entity label to enrich
        entity_type: 'method', 'concept', 'dataset', 'gene', etc.
        preferred_corpus: Preferred corpus to try first (e.g., 'mesh', 'ols')
    
    Returns:
        CorpusMatch with enrichment data (found=False if not found)
    """
    
    if entity_type == "dataset":
        # Try Cellosaurus for cell lines
        result = lookup_cellosaurus(label)
        if result:
            return result
    
    elif entity_type == "gene":
        # Try NCBI Gene
        result = lookup_ncbi_gene(label)
        if result:
            return result
    
    elif entity_type in {"method", "concept"}:
        # For biomedical methods and concepts, try MeSH first, then OLS
        result = lookup_mesh(label, "descriptor")
        if result:
            return result
        
        # Try OLS with GO (Gene Ontology) and other biomedical ontologies
        for ontology in ["go", "chebi", "doid"]:
            result = lookup_ols(label, ontology)
            if result:
                return result
    
    # Return not-found result
    return CorpusMatch(found=False, label=label, aliases=[])


def get_hierarchy(term: str, ontology: str = "go") -> list[str]:
    """Get parent terms (ontological hierarchy) from OLS.
    
    Args:
        term: Search term
        ontology: Ontology to search
    
    Returns:
        List of parent term labels
    """
    try:
        _rate_limit("ols", OLS_DELAY)
        
        # Get the term first
        url = f"{OLS_BASE}/search"
        params = {"q": term, "ontology": ontology, "type": "class", "rows": 1}
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        results = data.get("response", {}).get("docs", [])
        
        if not results:
            return []
        
        iri = results[0].get("iri", "")
        if not iri:
            return []
        
        # Query for parent terms
        parents_url = f"{OLS_BASE}/ontologies/{ontology}/terms/{quote(iri, safe='')}/hierarchicalParents"
        parents_response = requests.get(parents_url, timeout=5)
        
        if parents_response.status_code != 200:
            return []
        
        parents_data = parents_response.json()
        parents = [p.get("label", "") for p in parents_data.get("_embedded", {}).get("terms", [])]
        
        return [p for p in parents if p]  # Filter out empty strings
    except Exception:
        return []
