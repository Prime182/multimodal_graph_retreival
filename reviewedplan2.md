
***

# Complete Fix Plan: 12-Item Master Checklist

## Phase 1 — Blockers (Do First, System Is Broken Without These)

### Fix 1: Reset Circuit Breaker + Add CLI Reset Command

**File: `backend/graphrag/cli.py`**

Add a new subcommand `reset-circuit-breaker` so you can clear stuck state without shell access:

```python
# In cli.py, add to the argparse subcommands section:
reset_cb_parser = subparsers.add_parser(
    "reset-circuit-breaker",
    help="Reset all circuit breaker failure counters (use after transient Gemini outages).",
)
reset_cb_parser.add_argument(
    "--service",
    default=None,
    help="Service name to reset (omit to reset all services).",
)

# In the dispatch block:
elif args.command == "reset-circuit-breaker":
    from .circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    cb.reset(args.service)  # reset() must accept None for all-services
    service_label = args.service or "all services"
    print(f"✓ Circuit breaker reset for {service_label}.")
```

**File: `backend/graphrag/circuit_breaker.py`**

Add a `reset()` method if not present:
```python
def reset(self, service: str | None = None) -> None:
    """Reset failure counts. Pass service=None to reset all."""
    with self._lock:
        if service is None:
            self._state.clear()
        elif service in self._state:
            del self._state[service]
        # also clear SQLite persistence
        self._db_reset(service)
```

**Immediate one-time fix:**
```bash
find . -name "circuit_breaker*.db" -o -name "cb_state*.db" | xargs rm -f
```

***

### Fix 2: Pydantic `value` Field Coercion in `extraction.py`

**File: `backend/graphrag/extraction.py`**

Gemini returns ranges like `"53–71"` or `"up to 16"` which crash chunk extraction entirely .

```python
# Change the _GeminiEntityPayload class field:
value: float | str | None = None

@field_validator("value", mode="before")
@classmethod
def _coerce_value(cls, v: Any) -> Any:
    if v is None or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        # Extract first numeric token from range strings like "53–71", "up to 16"
        cleaned = re.sub(r"[^\d.\-]", " ", v.replace("–", "-").replace("−", "-"))
        for token in cleaned.split():
            try:
                return float(token)
            except ValueError:
                continue
    return None  # Discard unparseable values rather than crashing the chunk
```

***

## Phase 2 — High Priority (Semantic Correctness Breaks Without These)

### Fix 3: Swap Embedding Model

**File: `backend/graphrag/embeddings.py`** 

```python
# Line ~80 in SentenceTransformerEmbedder:
# BEFORE:
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# AFTER:
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

Pre-download once before starting the server:
```bash
ALLOW_SBERT_MODEL_DOWNLOAD=1 python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

***

### Fix 4: Wire `EntityCanonicalizer` into `extract_layer2()` AND `build_search_bundle()`

**File: `backend/graphrag/extraction.py`**

After the entity merging loop inside `extract_layer2()`:
```python
from .canonicalization import EntityCanonicalizer
_canonicalizer = EntityCanonicalizer()

# After building global_entities dict, before returning Layer2DocumentRecord:
canonical_global: dict[str, Layer2EntityRecord] = {}
for key, entity in global_entities.items():
    canon_label = _canonicalizer.canonicalize(entity.label, entity.entity_type)
    if canon_label != entity.label:
        entity.label = canon_label
        key = _merge_key(entity.entity_type, canon_label)
    existing = canonical_global.get(key)
    if existing:
        existing.mention_chunk_ids = list(set(existing.mention_chunk_ids + entity.mention_chunk_ids))
        existing.confidence = max(existing.confidence, entity.confidence)
        existing.aliases = list(set(existing.aliases + entity.aliases))
    else:
        canonical_global[key] = entity
global_entities = canonical_global
```

**File: `backend/graphrag/search_service.py`** — inside `build_search_bundle()`, canonicalize the entity index:
```python
from .canonicalization import EntityCanonicalizer
_svc_canonicalizer = EntityCanonicalizer()

# Where the entity_index dict is built:
for doc in self.layer2_docs:
    for entity in doc.entities:
        canon_label = _svc_canonicalizer.canonicalize(entity.label, entity.entity_type)
        entity.label = canon_label  # normalize in-place before indexing
```

***

## Phase 3 — Medium Priority (Silent Data Corruption Without These)

### Fix 5: `_merge_key()` Case Collision + Entity Type Enum

**File: `backend/graphrag/entities.py`** 

Add validation in `__post_init__` (convert dataclass to use `__post_init__` by removing `slots=True` or using a classmethod):

```python
_VALID_ENTITY_TYPES = frozenset({
    "concept", "method", "claim", "result", "equation", "dataset"
})

# Since slots=True is used, enforce via a module-level factory or add field validator
# Option: change to a regular dataclass with __post_init__:
@dataclass  # remove slots=True to allow __post_init__
class Layer2EntityRecord:
    entity_id: str
    entity_type: str
    label: str
    source_chunk_id: str
    mention_chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    extractor_model: str = "heuristic-v1"
    embedding: list[float] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_type = self.entity_type.lower().strip()
        if self.entity_type not in _VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{self.entity_type}'. "
                f"Must be one of: {sorted(_VALID_ENTITY_TYPES)}"
            )
        if self.embedding and not isinstance(self.embedding[0], float):
            self.embedding = [float(v) for v in self.embedding]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

**File: `backend/graphrag/extraction.py`** — fix `_merge_key()`:
```python
def _merge_key(entity_type: str, label: str) -> str:
    # BEFORE: f"{entity_type}::{_normalize_key(label)}"
    # AFTER — always lowercase the type to prevent "Concept" vs "concept" collisions:
    return f"{entity_type.lower().strip()}::{_normalize_key(label)}"
```

***

### Fix 6: Embedding Dimension Enforcement (Two Places)

**File: `backend/graphrag/embeddings.py`** 

```python
def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: {len(left)} vs {len(right)}. "
            "All embedders must use the same configured dim. "
            "Check that SBERT fallback dim matches Gemini embedder dim."
        )
    return sum(a * b for a, b in zip(left, right))
```

**File: `backend/graphrag/extraction.py`** — in `_entity_base()` or wherever `Layer2EntityRecord` is constructed with an embedding, add a resize guard:
```python
from .embeddings import _resize_vector

# Before constructing Layer2EntityRecord:
if embedding and len(embedding) != embedder.dim:
    embedding = _resize_vector(embedding, embedder.dim)
```

***

### Fix 7: Restore `known_hierarchies` YAML Fallback in `edges.py`

**File: `backend/graphrag/edges.py`** 

`infer_is_a_edges()` silently returns zero edges when OLS is down. The `except Exception: pass` block on line ~175 swallows the failure completely. Fix with YAML fallback:

```python
from .domain_config import get_domain_knowledge

@lru_cache(maxsize=1)
def _known_hierarchies() -> dict[str, list[str]]:
    return get_domain_knowledge().get("extraction", {}).get("known_hierarchies", {})

# Inside infer_is_a_edges(), replace the bare `except Exception: pass` with:
except Exception:
    # OLS unavailable — fall back to YAML-backed known_hierarchies
    fallback = _known_hierarchies()
    for parent_label in fallback.get(entity.label, []):
        parent_norm = _normalize(parent_label)
        parent_entity = doc_entities.get(parent_norm)
        if parent_entity is None or parent_entity.entity_id == entity.entity_id:
            continue
        edge_id = _stable_id("is-a-fallback", entity.entity_id, parent_entity.entity_id)
        if edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)
        edges.append(Layer3EdgeRecord(
            edge_id=edge_id,
            relation_type="IS_A",
            source_node_id=entity.entity_id,
            source_node_type=_entity_node_type(entity.entity_type),
            source_label=entity.label,
            target_node_id=parent_entity.entity_id,
            target_node_type=_entity_node_type(parent_entity.entity_type),
            target_label=parent_entity.label,
            confidence=_clip(0.75),
            source_chunk_id=entity.source_chunk_id,
            extractor_model="ontology-yaml-fallback",
            evidence="From known_hierarchies in domain_knowledge.yaml",
            metadata={"paper_id": doc.paper_id, "hierarchy_type": "yaml_fallback"},
        ))
```

***

## Phase 4 — Infrastructure & Architectural Gaps

### Fix 8: Neo4j Connection Keep-Alive

**File: `backend/graphrag/graph_store.py`** 

```python
# In Neo4jGraphStore.__init__(), replace:
self._driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_user, settings.neo4j_password),
)
# With:
self._driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_user, settings.neo4j_password),
    keep_alive=True,
    max_connection_lifetime=300,           # recycle before AuraDB's 600s idle timeout
    connection_acquisition_timeout=60,
    max_connection_pool_size=10,
)
```

***

### Fix 9: Startup Fail-Fast for Missing SBERT

**File: `backend/graphrag/server.py`** 

```python
# After the probe_embedding_backends() call in create_app():
embedding_health = probe_embedding_backends(
    dim=settings.embedding_dim,
    prefer_remote=use_gemini,
)
app.state.embedding_health = embedding_health

import logging
_log = logging.getLogger(__name__)

if not embedding_health.get("remote_available") and not embedding_health.get("local_available"):
    raise RuntimeError(
        "FATAL: No embedding backend available at startup. "
        f"Remote error: {embedding_health.get('remote_error')}. "
        f"Local error: {embedding_health.get('local_error')}. "
        "Run with ALLOW_SBERT_MODEL_DOWNLOAD=1 once to cache the model."
    )

if not embedding_health.get("local_available"):
    _log.warning(
        "⚠ SBERT local fallback is NOT cached (error: %s). "
        "If Gemini becomes unavailable, ALL embedding calls will fail. "
        "Set ALLOW_SBERT_MODEL_DOWNLOAD=1 and restart to pre-cache.",
        embedding_health.get("local_error"),
    )
```

***

### Fix 10: Parser Abstraction — `ArticleParser` Protocol + JATS Support

**File: `backend/graphrag/parser.py`** — add at top, before existing functions:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ArticleParser(Protocol):
    def can_parse(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> PaperRecord: ...


class ElsevierParser:
    """Wraps the existing Elsevier ce: namespace parse_article()."""
    def can_parse(self, path: Path) -> bool:
        try:
            root = etree.parse(str(path), _PARSER).getroot()
            return root.tag in {"full-text-retrieval-response", "article"} and any(
                child.tag == "item-info" for child in root
            )
        except Exception:
            return False

    def parse(self, path: Path) -> PaperRecord:
        return parse_article(path)  # delegate to existing function


class JATSParser:
    """Handles PubMed / arXiv JATS XML (root tag: <article>)."""
    def can_parse(self, path: Path) -> bool:
        try:
            root = etree.parse(str(path), _PARSER).getroot()
            local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
            return local == "article" and root.get("article-type") is not None
        except Exception:
            return False

    def parse(self, path: Path) -> PaperRecord:
        root = etree.parse(str(path), _PARSER).getroot()
        ns = {"jats": "https://jats.nlm.nih.gov/ns/archiving/1.0/"}

        def _get(xpath: str) -> str:
            el = root.find(xpath)
            return _norm(" ".join(el.itertext())) if el is not None else ""

        title = _get(".//article-title") or path.stem
        abstract = _get(".//abstract")
        doi = _get(".//article-id[@pub-id-type='doi']") or None
        paper_id = _build_paper_id(doi=doi, journal_code=None, article_id=path.stem)

        # Build minimal PaperRecord — sections from body
        body = root.find(".//body")
        sections: list[SectionRecord] = []
        if abstract:
            sections.append(SectionRecord(
                section_id=_scoped_id(paper_id, "abstract", "abstract"),
                paper_id=paper_id, title="Abstract",
                section_type="abstract", level=1, ordinal=0,
                text=abstract, paragraphs=[abstract],
                key_sentence=_first_sentence(abstract),
            ))
        if body is not None:
            for i, sec in enumerate(body.findall(".//sec")):
                sec_title = _norm(" ".join(sec.find("title").itertext())) if sec.find("title") is not None else f"Section {i+1}"
                paras = [_norm(" ".join(p.itertext())) for p in sec.findall(".//p")]
                text = " ".join(paras)
                if text:
                    sections.append(SectionRecord(
                        section_id=_scoped_id(paper_id, f"sec-{i}", f"sec-{i}"),
                        paper_id=paper_id, title=sec_title,
                        section_type=_section_type(sec_title),
                        level=1, ordinal=i+1,
                        text=text, paragraphs=paras,
                        key_sentence=_first_sentence(text),
                    ))

        return PaperRecord(
            paper_id=paper_id, source_path=str(path),
            title=title, doi=doi,
            abstract=abstract, sections=sections,
            metadata={"source_format": "jats_xml"},
        )


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: list[ArticleParser] = []

    def register(self, parser: ArticleParser) -> None:
        self._parsers.append(parser)

    def parse(self, path: Path) -> PaperRecord:
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser.parse(path)
        raise ValueError(
            f"No registered parser can handle '{path}'. "
            "Supported formats: Elsevier XML, JATS/PubMed XML."
        )


# Default registry — used by cli.py and search_service.py
default_registry = ParserRegistry()
default_registry.register(ElsevierParser())
default_registry.register(JATSParser())
```

**File: `backend/graphrag/cli.py`** — replace `parse_article(path)` calls:
```python
from .parser import default_registry
# BEFORE: paper = parse_article(path)
# AFTER:
paper = default_registry.parse(Path(path))
```

***

## Execution Order

```
Phase 1 (blockers — run today):
  1. Delete circuit_breaker .db files
  2. Add circuit-breaker reset command to cli.py
  3. Fix _GeminiEntityPayload.value coercion in extraction.py

Phase 2 (run before next ingestion pass):
  4. Swap MODEL_NAME in embeddings.py → all-MiniLM-L6-v2
  5. Pre-download model with ALLOW_SBERT_MODEL_DOWNLOAD=1
  6. Wire EntityCanonicalizer in extract_layer2() and build_search_bundle()

Phase 3 (before merging to main):
  7. Add __post_init__ + _VALID_ENTITY_TYPES in entities.py
  8. Fix _merge_key() lowercase in extraction.py
  9. Add cosine_similarity() dim assertion in embeddings.py
 10. Add _resize_vector guard in extraction.py _entity_base()
 11. Restore known_hierarchies YAML fallback in edges.py

Phase 4 (infrastructure — can be a separate PR):
 12. Neo4j keep_alive in graph_store.py
 13. Startup fail-fast for SBERT in server.py
 14. Add ArticleParser protocol + JATSParser + ParserRegistry in parser.py
 15. Update cli.py to use default_registry
```