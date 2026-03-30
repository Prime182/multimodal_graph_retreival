# GraphRAG System: Production Engineering Review
**Classification: Principal Engineer Review — Implementation-Ready**

---

## SECTION 1: SYSTEM DESIGN GAP ANALYSIS

### Top 10 Architectural Failures

---

#### Failure 1 — Neo4j is structurally dead code

**Severity: Critical**

`graph_store.py` and `graph_retrieval.py` contain ~900 lines of Cypher queries, upsert logic, and vector index definitions. None of it is reachable from the live query path.

The actual call graph is:

```
HTTP GET /api/search
  → GraphRAGSearchService.search()
  → build_search_bundle()
  → LocalVectorIndex.search()      ← the only retrieval that runs
```

`GraphRetrieval` is never instantiated anywhere in the active path. The system is advertising itself as a Graph RAG system but is actually a local in-memory vector search with a regex entity extractor. The graph is written during `load-neo4j` CLI command but never read back. You could comment out the entire `graph_store.py`, `graph_retrieval.py`, and `indexing.py` and the API would behave identically.

**Root cause:** The search service was built against `LocalVectorIndex` and the Neo4j integration was added as a parallel path that was never wired in.

---

#### Failure 2 — HashingEmbedder destroys retrieval correctness

**Severity: Critical**

`build_entity_embedder()` returns `GeminiEmbedder` which silently falls back to `HashingEmbedder` on any API failure. `HashingEmbedder` uses Blake2b of individual tokens projected into a 256-dimensional space. The cosine similarity between two semantically related texts is not meaningfully higher than between two unrelated texts.

Concrete proof from the codebase: `test_local_index_prefers_the_relevant_article` queries `"renewable energy ev charging bangladesh"` and asserts the first hit is about sustainable transportation. This test passes only because that paper contains the exact tokens `sustainable`, `transportation`. If you query `"EV infrastructure adoption"` the test would fail.

There is **no semantic fallback**. The system has no sentence-transformers, no SBERT, no local embedding model of any kind. Without `GOOGLE_API_KEY` the system cannot do semantic retrieval.

**The correct fix is not "use a better model" — it is to bundle a local embedding model (e.g., `sentence-transformers/all-mpnet-base-v2`) as a mandatory non-optional dependency with zero API requirements.**

---

#### Failure 3 — Extraction is calibrated to exactly two papers

**Severity: Critical**

Three independent hardcoding problems compound each other:

**A. `_GENE_SYMBOL_RE`** lists 26 specific gene symbols from `BJ_100828.xml`. A climate science paper, a materials science paper, or even a different biomedical paper about oncology will extract zero genes.

**B. Dataset patterns (`_DATASET_DEFINITION_PATTERNS`, `_DATASET_DIRECT_PATTERNS`)** match `TBL20`, `BL20`, `V`, `A`, `Vm`, `Am` — labels from one specific Theileria bovine biology paper. Pattern `r"\b(V|A)\b"` will match the letter V in any sentence containing velocity, voltage, or volume.

**C. `_local_results()` gates on dataset detection:**
```python
datasets = _result_datasets(sentence, chunk.text) or chunk_datasets
if not datasets:
    continue  # SKIPS ALL NUMERIC EXTRACTION
```
For any paper where none of the 35+ hardcoded dataset patterns fire, the system extracts zero quantitative results regardless of how many numbers, percentages, or metrics appear in the text.

**`domain_knowledge.yaml` is not a general knowledge base.** It is a hand-crafted lookup table for the test corpus. It contains 14 canonical concept names (all from one paper), 11 method aliases (all from one paper), and 0 entries for physics, chemistry, climate science, economics, or any other domain.

---

#### Failure 4 — Chunk overlap is declared but never applied

**Severity: High**

`Phase1Settings.chunk_overlap_words = 30` exists. `chunk_article()` never reads `settings.chunk_overlap_words`. The implementation does:

```python
def _section_chunks(section, max_words):
    # builds chunks with no overlap at all
```

Chunk boundaries silently cut sentences, claims, and experimental results. A 3-sentence finding that spans a chunk boundary will have its first two sentences in chunk N and its conclusion in chunk N+1. Neither chunk contains a complete claim. The `prev_chunk_id`/`next_chunk_id` fields exist but the retrieval path never uses them to expand context.

---

#### Failure 5 — `lru_cache` on instance method is broken

**Severity: High**

In `rag.py`:
```python
@lru_cache(maxsize=1)
def _query_graph(self) -> Any | None:
```

`lru_cache` on an undecorated instance method does not hold a reference to `self` separately — it holds `self` as the first argument, preventing garbage collection of the instance and creating a memory leak. More critically, if `QuerySynthesizer` is ever instantiated twice (e.g., in tests, or after a reload), the second instance shares the cached graph from the first instance, which was compiled with the first instance's node methods bound to the wrong `self`. The graph nodes call `self._retrieve_node`, `self._synthesize_node` etc., so the cached graph from instance A will invoke methods on instance A even when called from instance B.

---

#### Failure 6 — `lookup_mesh()` routes to the wrong API

**Severity: High — Silent Correctness Failure**

```python
def lookup_mesh(query: str, search_type: str = "descriptor") -> Optional[CorpusMatch]:
    return lookup_ols(query, "go")
```

This calls the Gene Ontology (GO) API, not MeSH. The function signature says `search_type: str = "descriptor"` (a MeSH concept) but it ignores both the parameter and the intent. Any caller expecting MeSH disease/drug/procedure terms gets GO biological process terms instead, with no warning. This corrupts entity enrichment for any concept that has a MeSH entry but not a GO entry (e.g., `"hypertension"`, `"aspirin"`, `"computed tomography"`).

---

#### Failure 7 — No parser abstraction — format lock-in

**Severity: High**

`parse_article()` directly uses `lxml` and hard-codes every Elsevier XML tag (`ce:para`, `ce:section-title`, `ce:doi`, `ce:author`, `jid`, etc.). There is no `ArticleParser` interface. Adding PubMed XML, JATS XML, PDF, or arXiv XML requires rewriting `parser.py` from scratch and breaks all callers. The function returns an empty `PaperRecord` with no error when fed non-Elsevier XML — silent failure.

---

#### Failure 8 — No validation contract between extraction and retrieval

**Severity: High**

`Layer2EntityRecord` has `entity_type: str` — a raw string with no enforcement. The codebase uses `"concept"`, `"method"`, `"claim"`, `"result"`, `"equation"`, `"dataset"` but nothing prevents `"Concept"`, `"CONCEPT"`, or `"unknown"` from entering the graph. The `_merge_key()` function uses this raw string as a deduplication key, so `"concept::wtap"` and `"Concept::WTAP"` create two separate entity nodes. This happened in practice — `_normalize_key()` only lowercases the label, not the type.

Additionally, `Layer2EntityRecord.embedding` is `list[float]` with no dimension enforcement. If Gemini returns a 768-dim vector and the local index was built with 256-dim vectors, cosine similarity silently computes on mismatched dimensions (Python's `zip` truncates to the shorter one — no error, wrong result).

---

#### Failure 9 — GraphRAGSearchService re-parses everything on every restart

**Severity: High at Scale**

`GraphRAGSearchService.__post_init__()` calls `_load_corpus()` which calls `parse_article()`, `chunk_article()`, and `extract_layer2()` for every XML file, synchronously, on startup. For 100 papers this takes ~30 seconds. For 10,000 papers this is impossible. There is no checkpoint, no serialization, no cache. Every server restart triggers a full reparse. The `ingestion_status.py` SQLite store records what was processed but is never consulted to skip already-processed papers.

---

#### Failure 10 — Canonicalization is never applied to the live pipeline

**Severity: Medium — Architectural Inconsistency**

`EntityCanonicalizer` exists in `canonicalization.py` with real deduplication logic. It is exported in `__init__.py`. It is never called in `GraphRAGSearchService`, `extract_layer2()`, or `build_search_bundle()`. The entity deduplication that does happen is only within a single paper via `_merge_key()`. Across papers, `"CRISPR"` from paper A and `"CRISPR"` from paper B are two separate entity objects with no connection unless Neo4j is used — which it isn't.

---

### Component Status Matrix

| Component | Status | Reason |
|---|---|---|
| `parser.py` | **Partial** | Real code, Elsevier XML only, silent failure on other formats |
| `chunking.py` | **Partial** | Real code, overlap not implemented, context expansion not used |
| `extraction.py` (local heuristics) | **Partial** | Works for 2-paper corpus, fails on any other domain |
| `extraction.py` (Gemini path) | **Gated** | Real code, requires API key, self-correction loop is correct |
| `embeddings.py` (GeminiEmbedder) | **Gated** | Real code, requires API key |
| `embeddings.py` (HashingEmbedder) | **Misleading** | Non-semantic, cosine similarity is meaningless |
| `retrieval.py` (LocalVectorIndex) | **Partial** | Real retrieval logic, semantic quality depends on embedder |
| `graph_store.py` | **Dead** | Never called from active query path |
| `graph_retrieval.py` | **Dead** | Never called from active query path |
| `indexing.py` | **Dead** | Only callable via CLI, never integrated with search |
| `rag.py` (with Gemini) | **Partial** | Real synthesis loop, `lru_cache` bug on instance method |
| `rag.py` (without Gemini) | **Non-functional** | Returns null answer, no degraded mode |
| `corpus.py` (NCBI Gene) | **Partial** | Real API calls, rate limited correctly, no disk cache |
| `corpus.py` (lookup_mesh) | **Broken** | Routes to wrong API silently |
| `canonicalization.py` | **Dead** | Never called in live pipeline |
| `edges.py` | **Partial** | Citation edges work, IS_A edges hardcoded to biomedical |
| `domain_knowledge.yaml` | **Partial** | Right architecture, wrong data (two papers only) |
| `tracing.py` | **Gated** | Real implementation, LangFuse connection never closed |
| `server.py` | **Functional** | FastAPI setup is correct, minor type annotation bug |
| `ingestion_status.py` | **Functional** | SQLite tracking works, never consulted on restart |
| `circuit_breaker.py` | **Functional** | Correct implementation, in-memory only (resets on restart) |

---

### Hidden Couplings

1. **`_local_results()` → dataset patterns → paper-specific labels**: Any change to dataset patterns can silently break result extraction for ALL papers simultaneously.

2. **`LocalVectorIndex._build()` runs synchronously in `__init__`**: Embedding all chunks on startup couples startup time to corpus size with no parallelism.

3. **`get_tracing_manager()` is a global singleton**: `TracingManager._instance` persists across test runs, causing tests that check tracing behavior to bleed state.

4. **`_dataset_registry()` in extraction.py is populated from `domain_knowledge.yaml`** which is loaded via `get_domain_knowledge()` decorated with `@lru_cache(maxsize=1)`. Modifying the YAML at runtime has no effect — the stale cache is used forever.

5. **`search_service.py`'s `build_search_bundle()` takes `layer2_docs: list`** but the entity lookup `entities_by_chunk` is built from `entity.source_chunk_id` — which assumes chunks are never renumbered between extraction and retrieval. If chunking settings change after extraction, chunk IDs become stale and entity lookup silently returns empty lists.

---

## SECTION 2: CORRECT PRODUCTION ARCHITECTURE

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION                               │
│  Parser Registry → Document Normalizer → Validation Gate        │
└────────────────────────────┬────────────────────────────────────┘
                             │ NormalizedDocument
┌────────────────────────────▼────────────────────────────────────┐
│                         CHUNKING                                │
│  Overlap-aware Chunker → Chunk Validator → Salience Scorer      │
└────────────────────────────┬────────────────────────────────────┘
                             │ ChunkRecord[]
┌────────────────────────────▼────────────────────────────────────┐
│                        EXTRACTION                               │
│  Multi-pass Pipeline:                                           │
│    Pass 1: Schema-driven LLM extraction (primary)              │
│    Pass 2: Regex/NLP augmentation (always runs)                 │
│    Pass 3: Corpus enrichment (async, cached)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ ExtractionResult[]
┌────────────────────────────▼────────────────────────────────────┐
│                    ENTITY RESOLUTION                            │
│  Within-doc dedup → Cross-doc canonicalization → Alias merging  │
└────────────────────────────┬────────────────────────────────────┘
                             │ ResolvedEntity[]
┌────────────────────────────▼────────────────────────────────────┐
│                   GRAPH CONSTRUCTION                            │
│  Node writer → Edge inferencer → PageRank scorer               │
└────────────────────────────┬────────────────────────────────────┘
                             │ Persisted to Neo4j
┌────────────────────────────▼────────────────────────────────────┐
│                  EMBEDDING + INDEXING                           │
│  Local SBERT (always) → Optional Gemini upgrade                 │
│  Neo4j vector index + HNSW local fallback                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ Indexed
┌────────────────────────────▼────────────────────────────────────┐
│               RETRIEVAL ORCHESTRATION                           │
│  Query → entity extraction → graph expansion → vector search    │
│  Fusion: RRF(graph_hits, vector_hits, bm25_hits)               │
└────────────────────────────┬────────────────────────────────────┘
                             │ RankedPassage[]
┌────────────────────────────▼────────────────────────────────────┐
│                   RAG GENERATION                                │
│  Context builder → LLM synthesis → Grounding verifier          │
│  Degraded mode: return top passages with citation links         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Layer Specifications

#### Layer 1: Ingestion

**Responsibility:** Accept any scientific article format and produce a format-agnostic `NormalizedDocument`.

**Input:** File path or bytes + format hint  
**Output:** `NormalizedDocument` with guaranteed fields (title, abstract, sections, references)  
**Failure mode:** Raise `ParserError` with format identifier — never return a silent empty record  
**Validation:** Every `NormalizedDocument` must have `len(abstract) > 50` and `len(sections) > 0`

**Design:**
```python
class ArticleParser(Protocol):
    format_id: str  # "elsevier_xml", "pubmed_xml", "jats_xml", "arxiv_xml", "pdf"
    
    def can_parse(self, content: bytes) -> bool: ...
    def parse(self, path: Path) -> NormalizedDocument: ...

class ParserRegistry:
    def register(self, parser: ArticleParser) -> None: ...
    def parse(self, path: Path) -> NormalizedDocument: ...
    # Tries each parser in priority order, raises ParserError if none match
```

#### Layer 2: Chunking

**Responsibility:** Produce overlapping, size-bounded chunks with correct prev/next links.

**Input:** `NormalizedDocument`  
**Output:** `Chunk[]` with guaranteed `len(text) > 0`, `embedding_dim = None` (set later)  
**Failure mode:** If a section has no extractable text, log and skip — never produce empty chunks  
**Validation:** Assert `sum(chunk.word_count) >= 0.8 * document.word_count` (overlap may increase total)

**The overlap algorithm (currently missing):**

```
For each section:
  chunks = []
  buffer = []
  buffer_words = 0
  overlap_carry = []  # last N words from previous chunk

  for sentence in sentences:
    if buffer_words + len(sentence.words) > max_words AND buffer:
      chunk_text = join(overlap_carry + buffer)
      chunks.append(chunk_text)
      overlap_carry = buffer[-overlap_words:]  # carry forward
      buffer = [sentence]
      buffer_words = len(sentence.words)
    else:
      buffer.append(sentence)
      buffer_words += len(sentence.words)
```

#### Layer 3: Extraction

**Responsibility:** Extract typed entities and claims from each chunk.

**Input:** `Chunk` + `Document` context  
**Output:** `ExtractionResult` with validated entity list  
**Failure mode:** If LLM fails, run local extraction — never return empty result without logging  
**Validation:** Every `result` entity must have `value: float` and `metric: str`; every `claim` must have `text` length > 20

**Multi-pass design:**

```
Pass 1 (LLM, if available):
  - Send chunk to Gemini/Claude with structured output schema
  - Validate with Pydantic — reject invalid entities
  - Self-correct once if quality < threshold
  - Cache result in SQLite keyed by (chunk_id, model_version)

Pass 2 (Always runs, augments Pass 1):
  - Run domain-agnostic NER: quantities, percentages, p-values
  - Run schema-agnostic method detection (any phrase ending in assay/seq/cytometry/blot)
  - Merge with Pass 1 results, deduplicate by normalized label

Pass 3 (Async, corpus enrichment):
  - For each entity from Passes 1+2, query corpus in background
  - Update entity record with aliases and external IDs
  - Do NOT block serving on this
```

#### Layer 4: Entity Resolution

**Responsibility:** Merge duplicate entities within and across documents.

**Input:** `ExtractionResult[]` for a batch of documents  
**Output:** `ResolvedEntity[]` with canonical IDs and alias lists  
**Failure mode:** If similarity computation fails, default to exact-match dedup only — never skip  
**Validation:** No two `ResolvedEntity` objects should share a canonical label after normalization

**Algorithm:**
```
Within-document (fast, synchronous):
  key = f"{entity_type}::{normalize(label).lower()}"
  merge on exact key match

Cross-document (offline batch job):
  Group by entity_type
  For each group:
    Build embedding matrix (batch embed all labels)
    Run approximate nearest neighbor (HNSW) to find candidates sim > 0.88
    For candidates: apply string overlap filter
    Cluster: Union-Find on accepted pairs
    Elect canonical: most frequent label OR longest label
```

#### Layer 5: Graph Construction

**Responsibility:** Write resolved entities and typed edges into Neo4j.

**Input:** `ResolvedEntity[]`, `CitationEdge[]`, `SemanticEdge[]`  
**Output:** Persisted Neo4j graph with vector indexes  
**Failure mode:** All writes are idempotent (`MERGE` not `CREATE`); failed writes are queued for retry  
**Validation:** After batch write, query count of written nodes and compare to input count

#### Layer 6: Embedding + Indexing

**Responsibility:** Embed all text artifacts and maintain searchable indexes.

**Input:** `Chunk[]`, `ResolvedEntity[]`  
**Output:** Populated Neo4j vector indexes OR local HNSW index  
**Failure mode:** If Gemini unavailable, use local SBERT model (bundled, no API key required)  
**Validation:** Spot-check 10 random embedding pairs — semantically similar pairs must score > 0.6

**Model hierarchy (never fall through to hashing):**
```
Priority 1: Gemini text-embedding-004 (API key required)
Priority 2: sentence-transformers/all-mpnet-base-v2 (local SBERT fallback)
Priority 3: RAISE ERROR — do not fall back to HashingEmbedder
```

#### Layer 7: Retrieval Orchestration

**Responsibility:** Given a query, produce a ranked list of relevant passages with graph context.

**Input:** `str` query, `int` top_k  
**Output:** `RankedPassage[]` each with text, source, graph neighborhood, confidence  
**Failure mode:** If Neo4j unavailable, fall back to local HNSW — log the fallback  
**Validation:** Retrieved passages must have average relevance score > 0.3 (alert if below)

**Hybrid retrieval algorithm:**
```
Step 1 - Query entity extraction:
  Extract entities from query (same Pass 2 local extractor, < 100ms)

Step 2 - Graph expansion:
  For each query entity, fetch its 2-hop neighborhood from Neo4j
  Collect all chunk IDs that mention these entities or their aliases
  This gives graph_candidates[]

Step 3 - Vector search:
  Embed query
  Query Neo4j vector index (or local HNSW): top 3*k candidates
  This gives vector_candidates[]

Step 4 - BM25 lexical search:
  Run BM25 over chunk corpus
  This gives bm25_candidates[]

Step 5 - Reciprocal Rank Fusion:
  score(chunk) = Σ 1/(rank_i + 60) across all three lists
  Rank by fused score, return top_k

Step 6 - Context expansion:
  For each top chunk, fetch prev_chunk and next_chunk
  Include in context window if total tokens < budget
```

#### Layer 8: RAG Generation

**Responsibility:** Generate a grounded answer from retrieved passages.

**Input:** `str` query, `RankedPassage[]`  
**Output:** `GeneratedAnswer` with text, citations, confidence, grounding status  
**Failure mode (no LLM):** Return top 3 passages as structured citations — never return null answer  
**Validation:** Every sentence in the answer must be traceable to at least one passage (grounding check)

---

## SECTION 3: REWRITE PLAN

### Phase 1: Stabilize Current System
**Goal: Stop silent failures. Make the system honest about what it can do.**  
**Timeline: 1 week**

**Files to modify:**

`backend/graphrag/embeddings.py`
- DELETE `HashingEmbedder` as a default fallback
- ADD `SentenceTransformerEmbedder` using `sentence-transformers` package as mandatory fallback
- MODIFY `build_entity_embedder()` to never return `HashingEmbedder` in the live path
- KEEP `HashingEmbedder` only for unit tests that explicitly request it

```python
# NEW: mandatory local fallback
class SentenceTransformerEmbedder:
    MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self, dim: int = 384):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL)
        self.dim = dim
    
    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

def build_entity_embedder(dim: int = 384, *, prefer_remote: bool = True) -> TextEmbedder:
    if prefer_remote and gemini_available():
        return GeminiEmbedder(dim=dim)
    return SentenceTransformerEmbedder(dim=dim)  # never HashingEmbedder
```

`backend/graphrag/chunking.py`
- IMPLEMENT `chunk_overlap_words` parameter — it is declared in `Phase1Settings` but ignored
- MODIFY `_section_chunks()` to carry the last N words of each chunk into the next
- KEEP all other logic

`backend/graphrag/extraction.py`
- REMOVE `_GENE_SYMBOL_RE` entirely — replace with a call to the NER library or a corpus lookup
- REMOVE `_DATASET_DEFINITION_PATTERNS` and `_DATASET_DIRECT_PATTERNS` — replace with a generic quantity/dataset detector
- MODIFY `_local_results()` to remove the dataset gate: extract numerics from ALL chunks, attach dataset context when available but do not require it
- KEEP all Pydantic validation models, they are correct

`backend/graphrag/corpus.py`
- FIX `lookup_mesh()` — it must actually call the MeSH API (`https://id.nlm.nih.gov/mesh/lookup/descriptor`)
- KEEP all existing `lookup_ols()` and `lookup_ncbi_gene()` logic

`backend/graphrag/rag.py`
- FIX `_query_graph()` `lru_cache` bug — use a module-level cache dict keyed by instance id, or simply build the graph once in `__init__`
- ADD degraded mode: when `self.enabled = False`, return structured citation list from passages instead of `None`

`backend/graphrag/server.py`
- FIX `serve_app_js()` return type annotation (`-> FileResponse`)
- ADD startup health check that validates embedding model is loadable

**Files to keep unchanged:**
- `circuit_breaker.py` — correct implementation
- `ingestion_status.py` — correct implementation
- `models.py` — correct data models
- `tracing.py` — mostly correct, add `__del__` to flush/close

---

### Phase 2: Remove Hardcoded Logic
**Goal: Make extraction domain-agnostic.**  
**Timeline: 2 weeks**

**Files to modify:**

`backend/graphrag/domain_knowledge.yaml`
- REMOVE all paper-specific entries from `dataset_registry` (TBL20, BL20, Vm, Am, V, A, Ode)
- REMOVE paper-specific genes from any implicit lists
- ADD generic quantity patterns: `percentage`, `fold_change`, `p_value`, `concentration`
- ADD generic method suffix list: `assay`, `seq`, `cytometry`, `blot`, `imaging`, `microscopy`, `spectroscopy`, `chromatography`
- ADD generic concept patterns: `pathway`, `receptor`, `inhibitor`, `activator`, `complex`, `cascade`

`backend/graphrag/extraction.py`
- REWRITE `_extract_dataset_mentions()` to use a generic pattern: any named experimental group introduced with parenthetical abbreviation `Full Name (ABBREV)` is a dataset candidate
- REWRITE `_local_methods()` to use suffix-based detection from YAML rather than fixed biomedical list
- REWRITE `_local_results()` to: (1) extract all numeric values with surrounding context, (2) attempt to associate with a measurement unit or metric phrase, (3) attach experimental condition if detectable — but never gate on dataset detection
- ADD `_detect_experimental_groups()` that finds any `(A)`, `(Control)`, `(Group 1)` style abbreviations in the document abstract and uses those as the dataset registry for that document

`backend/graphrag/edges.py`
- REMOVE hardcoded `known_hierarchies` list from the live code — move to YAML and generalize
- MODIFY `infer_is_a_edges()` to use OLS hierarchy lookup for any entity type, not just biomedical

---

### Phase 3: Introduce Schema-Driven Extraction
**Goal: Decouple extraction schema from domain; make it configurable per domain.**  
**Timeline: 2 weeks**

**New files to create:**

`backend/graphrag/extraction_schema.py`
```python
class EntitySchema(BaseModel):
    type_name: str              # "concept", "method", "claim", "result"
    required_fields: list[str]  # fields that must be non-null
    optional_fields: list[str]
    validation_rules: list[ValidationRule]
    extraction_hints: list[str]  # added to LLM prompt

class ExtractionSchema(BaseModel):
    domain: str                  # "biomedical", "physics", "general"
    version: str
    entity_schemas: list[EntitySchema]
    relation_schemas: list[RelationSchema]
    
    @classmethod
    def load(cls, domain: str) -> "ExtractionSchema":
        # loads from schemas/{domain}.yaml
```

`backend/graphrag/schemas/general.yaml` — base schema for any domain  
`backend/graphrag/schemas/biomedical.yaml` — extends general  
`backend/graphrag/schemas/physics.yaml` — extends general  

**Files to modify:**

`backend/graphrag/extraction.py`
- MODIFY `_GEMINI_PROMPT` to be generated from `ExtractionSchema` rather than hardcoded
- MODIFY `_GeminiChunkExtractionPayload` validation to use schema rules rather than hardcoded type names
- ADD `detect_domain(paper: PaperRecord) -> str` function that uses journal name, keywords, and abstract to select the schema

---

### Phase 4: Enable Real Graph RAG Retrieval
**Goal: Wire Neo4j into the active query path.**  
**Timeline: 3 weeks**

This is the most critical phase — the system cannot be called "Graph RAG" until this is done.

**New files to create:**

`backend/graphrag/retriever.py` — unified retrieval interface  
`backend/graphrag/bm25_index.py` — BM25 implementation using `rank_bm25` library  
`backend/graphrag/context_builder.py` — builds LLM context from retrieved passages + graph subgraph

**Files to modify:**

`backend/graphrag/search_service.py`
- REPLACE `LocalVectorIndex.search()` as the sole retrieval method
- ADD `HybridRetriever` that orchestrates graph + vector + BM25
- MODIFY `build_search_bundle()` to use graph neighborhood expansion

`backend/graphrag/graph_retrieval.py`
- ADD `get_entity_neighborhood(entity_ids: list[str], hops: int) -> SubGraph`
- ADD `get_chunks_mentioning_entities(entity_ids: list[str]) -> list[str]`
- These are the two missing methods that connect the graph to retrieval

`backend/graphrag/server.py`
- ADD `GraphRAGSearchService` initialization check: if Neo4j is unavailable, log clearly and use `LocalVectorIndex` with real embeddings as fallback
- ADD `/api/graph/entity/{entity_id}` endpoint to expose graph neighborhood for debugging

---

### Phase 5: Production Hardening
**Goal: Persistence, caching, idempotency, scale.**  
**Timeline: 2 weeks**

**Files to modify:**

`backend/graphrag/search_service.py`
- ADD ingestion checkpoint: on startup, read `ingestion_status.py` and skip papers with `status="complete"` and matching file hash
- ADD async ingestion: move `_load_corpus()` to a background task, serve empty results while loading

`backend/graphrag/ingestion_status.py`
- ADD `file_hash: str` column to `ingestion_status` table
- ADD `model_version: str` column to detect when re-extraction is needed

`backend/graphrag/extraction.py`
- ADD `extraction_cache.py`: SQLite cache keyed by `(chunk_id, chunk_text_hash, model_version)` — skip LLM call if cache hit exists

`backend/graphrag/corpus.py`
- ADD TTL to SQLite cache: re-query OLS/NCBI after 30 days
- ADD circuit breaker reset on restart by reading last failure time from SQLite (currently in-memory only)

`backend/graphrag/tracing.py`
- ADD `__del__` to `TracingManager` that calls `self.langfuse.flush()` and `self.langfuse.shutdown()`
- ADD connection health check endpoint

---

## SECTION 4: IMPLEMENTATION-LEVEL DETAILS

### A. Interfaces

```python
# backend/graphrag/interfaces.py

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol


class ArticleParser(Protocol):
    """Parses a single article file into a NormalizedDocument."""
    format_id: str

    def can_parse(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> "NormalizedDocument": ...


class TextEmbedder(Protocol):
    """Produces normalized float vectors. Must NOT use hashing."""
    dim: int

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class EntityExtractor(Protocol):
    """Extracts typed entities from a single chunk."""

    def extract(
        self,
        chunk: "Chunk",
        document_context: "DocumentContext",
    ) -> "ExtractionResult": ...

    def supports_domain(self, domain: str) -> bool: ...


class GraphStore(Protocol):
    """Persists and queries the knowledge graph."""

    def upsert_nodes(self, nodes: list["GraphNode"]) -> None: ...
    def upsert_edges(self, edges: list["GraphEdge"]) -> None: ...
    def get_neighborhood(
        self,
        node_ids: list[str],
        hops: int = 2,
        max_nodes: int = 100,
    ) -> "SubGraph": ...
    def vector_search(
        self,
        embedding: list[float],
        node_type: str,
        top_k: int,
    ) -> list[tuple[float, "GraphNode"]]: ...
    def is_available(self) -> bool: ...


class Retriever(Protocol):
    """Retrieves ranked passages for a query."""

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list["RankedPassage"]: ...


class ContextBuilder(Protocol):
    """Builds LLM-ready context from retrieved passages."""

    def build(
        self,
        query: str,
        passages: list["RankedPassage"],
        max_tokens: int = 4000,
    ) -> "GenerationContext": ...


class AnswerGenerator(Protocol):
    """Generates a grounded answer. Must degrade gracefully."""

    def generate(
        self,
        context: "GenerationContext",
    ) -> "GeneratedAnswer": ...

    def is_available(self) -> bool: ...
```

---

### B. Data Models

```python
# backend/graphrag/schema.py

from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(str, Enum):
    CONCEPT = "concept"
    METHOD = "method"
    CLAIM = "claim"
    RESULT = "result"
    EQUATION = "equation"
    DATASET = "dataset"


class ClaimType(str, Enum):
    FINDING = "finding"
    HYPOTHESIS = "hypothesis"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"


class Entity(BaseModel):
    entity_id: str
    entity_type: EntityType                     # enum — never raw string
    label: str = Field(min_length=1)
    source_chunk_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] = Field(default_factory=list)
    embedding_dim: int | None = None            # validated separately
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    extractor_model: str
    
    @field_validator("embedding")
    @classmethod
    def embedding_must_not_be_hashed(cls, v: list[float]) -> list[float]:
        # Detect hashing embedder output: values concentrate near ±0.1
        if v and len(v) > 0:
            import math
            avg_abs = sum(abs(x) for x in v) / len(v)
            if avg_abs < 0.05:
                raise ValueError(
                    "Embedding looks like hash output (avg_abs < 0.05). "
                    "Use a semantic embedder."
                )
        return v

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> "Entity":
        p = self.properties
        if self.entity_type == EntityType.RESULT:
            if not isinstance(p.get("value"), (int, float)):
                raise ValueError("Result entity must have numeric 'value' property")
            if not p.get("metric"):
                raise ValueError("Result entity must have 'metric' property")
        if self.entity_type == EntityType.CLAIM:
            if len(self.label.split()) < 4:
                raise ValueError("Claim label must be at least 4 words")
        return self


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section_id: str
    ordinal: int
    text: str = Field(min_length=1)
    chunk_type: str
    word_count: int = Field(gt=0)
    token_count: int = Field(gt=0)
    salience_score: float = Field(ge=0.0, le=1.0)
    embedding: list[float] = Field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    overlap_prev_words: int = 0             # how many words overlapped from prev chunk
    overlap_next_words: int = 0


class GraphNode(BaseModel):
    node_id: str
    node_type: str                           # "Paper", "Chunk", "Concept", etc.
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    pagerank: float = 0.0


class GraphEdge(BaseModel):
    edge_id: str
    relation_type: str
    source_id: str
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class SubGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    seed_node_ids: list[str]                 # which nodes were the query seeds


class RankedPassage(BaseModel):
    rank: int
    score: float
    chunk: Chunk
    paper_id: str
    paper_title: str
    doi: str | None
    section_title: str
    retrieval_method: str                    # "vector", "graph", "bm25", "fusion"
    graph_context: SubGraph | None = None   # neighborhood for graph-retrieved passages
    entities_in_chunk: list[Entity] = Field(default_factory=list)


class GenerationContext(BaseModel):
    query: str
    passages: list[RankedPassage]
    formatted_context: str                   # ready-to-inject into LLM prompt
    passage_count: int
    total_tokens_estimate: int
    source_map: dict[int, str]              # passage_index -> citation string


class GeneratedAnswer(BaseModel):
    answer: str                              # never None — degraded mode returns citations
    is_synthesized: bool                     # False if just returning passage citations
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[dict[str, Any]]
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    refined_query: str | None = None


class ExtractionResult(BaseModel):
    chunk_id: str
    entities: list[Entity]
    salience_score: float = Field(ge=0.0, le=1.0)
    extractor_model: str
    pass_used: Literal["llm", "local", "hybrid"]
    quality_score: float = Field(ge=0.0, le=1.0)
    extraction_warnings: list[str] = Field(default_factory=list)
```

---

### C. Critical Algorithms

#### Algorithm 1: Multi-pass Entity Extraction

```python
def extract_chunk(
    chunk: Chunk,
    doc_context: DocumentContext,
    schema: ExtractionSchema,
    llm: AnswerGenerator | None,
    embedder: TextEmbedder,
    corpus_client: CorpusClient,
    cache: ExtractionCache,
) -> ExtractionResult:

    # Check cache first
    cache_key = f"{chunk.chunk_id}::{schema.version}::{schema.domain}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    entities: list[Entity] = []
    pass_used = "local"

    # PASS 1: LLM extraction (if available)
    if llm and llm.is_available():
        prompt = build_extraction_prompt(chunk, doc_context, schema)
        try:
            raw = llm.extract_json(prompt)
            llm_entities = validate_extraction_output(raw, schema, chunk)
            if quality_score(llm_entities) >= 0.6:
                entities = llm_entities
                pass_used = "llm"
        except (LLMError, ValidationError) as e:
            log.warning(f"LLM extraction failed for {chunk.chunk_id}: {e}")
            # Fall through to local

    # PASS 2: Local augmentation (always runs)
    local_entities = local_extract(chunk, doc_context, schema)
    
    if pass_used == "llm":
        # Merge: keep LLM entities, add local entities not already covered
        entities = merge_extractions(primary=entities, secondary=local_entities)
        pass_used = "hybrid"
    else:
        entities = local_entities
        pass_used = "local"

    # PASS 3: Embed all entities
    for entity in entities:
        entity.embedding = embedder.embed(entity.label)
        entity.embedding_dim = embedder.dim

    # Compute result quality
    q = quality_score(entities)
    result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=entities,
        salience_score=compute_salience(chunk, entities),
        extractor_model=schema.domain,
        pass_used=pass_used,
        quality_score=q,
    )
    
    cache.set(cache_key, result)
    return result


def local_extract(chunk: Chunk, doc_context: DocumentContext, schema: ExtractionSchema) -> list[Entity]:
    """Domain-agnostic local extraction based on schema hints."""
    entities = []
    
    # Extract quantities (universal)
    for match in QUANTITY_RE.finditer(chunk.text):
        value, unit, context = parse_quantity(match, chunk.text)
        metric = infer_metric_from_context(context)
        if metric:
            entities.append(Entity(
                entity_type=EntityType.RESULT,
                label=f"{value} {metric}",
                properties={"value": value, "unit": unit, "metric": metric},
                ...
            ))
    
    # Extract method candidates (schema-driven suffix matching)
    method_suffixes = schema.get_hints("method_suffixes")  # from YAML
    for phrase in find_method_phrases(chunk.text, method_suffixes):
        entities.append(Entity(entity_type=EntityType.METHOD, label=phrase, ...))
    
    # Extract experimental groups (universal pattern: "Name (ABBREV)")
    for full_name, abbrev in find_defined_abbreviations(chunk.text):
        if looks_like_experimental_group(full_name):
            entities.append(Entity(entity_type=EntityType.DATASET, label=full_name, aliases=[abbrev], ...))
    
    # Extract claims (universal: sentence with claim verb + subject)
    for sentence in split_sentences(chunk.text):
        if has_claim_structure(sentence):
            entities.append(Entity(entity_type=EntityType.CLAIM, label=sentence, ...))
    
    return entities
```

#### Algorithm 2: Entity Resolution (Cross-document)

```python
def resolve_entities_across_corpus(
    extraction_results: list[ExtractionResult],
    embedder: TextEmbedder,
    similarity_threshold: float = 0.88,
) -> dict[str, str]:  # old_entity_id -> canonical_entity_id
    
    canonical_map = {}
    
    # Group by type (only resolve within same type)
    by_type: dict[str, list[Entity]] = defaultdict(list)
    for result in extraction_results:
        for entity in result.entities:
            by_type[entity.entity_type.value].append(entity)
    
    for entity_type, entities in by_type.items():
        if len(entities) < 2:
            continue
        
        # Batch embed all labels that don't have embeddings yet
        without_embedding = [e for e in entities if not e.embedding]
        if without_embedding:
            labels = [e.label for e in without_embedding]
            embeddings = embedder.embed_batch(labels)  # single batched call
            for entity, embedding in zip(without_embedding, embeddings):
                entity.embedding = embedding
        
        # Build HNSW index for fast ANN search
        index = HNSWIndex(dim=embedder.dim)
        id_to_entity = {}
        for entity in entities:
            index.add(entity.entity_id, entity.embedding)
            id_to_entity[entity.entity_id] = entity
        
        # Find candidate pairs
        union_find = UnionFind(entity.entity_id for entity in entities)
        
        for entity in entities:
            candidates = index.search(entity.embedding, k=10)
            for candidate_id, similarity in candidates:
                if similarity < similarity_threshold:
                    break
                candidate = id_to_entity[candidate_id]
                if candidate_id == entity.entity_id:
                    continue
                # Secondary filter: string overlap
                if label_compatible(entity.label, candidate.label):
                    union_find.union(entity.entity_id, candidate_id)
        
        # Build canonical map: elect most frequent label as canonical
        for cluster in union_find.get_clusters():
            label_counts = Counter(id_to_entity[eid].label for eid in cluster)
            canonical_label = label_counts.most_common(1)[0][0]
            canonical_id = next(
                eid for eid in cluster
                if id_to_entity[eid].label == canonical_label
            )
            for eid in cluster:
                canonical_map[eid] = canonical_id
    
    return canonical_map
```

#### Algorithm 3: Hybrid Retrieval with RRF

```python
def hybrid_retrieve(
    query: str,
    top_k: int,
    graph_store: GraphStore,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    embedder: TextEmbedder,
    entity_extractor: EntityExtractor,
) -> list[RankedPassage]:
    
    RRF_K = 60  # standard RRF constant
    
    # Extract entities from query for graph expansion
    query_entities = entity_extractor.extract_from_query(query)
    query_embedding = embedder.embed(query)
    
    # Source 1: Graph-expanded retrieval
    graph_chunk_ids = set()
    if graph_store.is_available() and query_entities:
        entity_ids = [e.entity_id for e in query_entities]
        # Also search by label in case entity wasn't extracted
        label_entity_ids = graph_store.find_entities_by_label(
            [e.label for e in query_entities], fuzzy=True
        )
        all_entity_ids = list(set(entity_ids + label_entity_ids))
        subgraph = graph_store.get_neighborhood(all_entity_ids, hops=2)
        graph_chunk_ids = graph_store.get_chunks_mentioning_entities(all_entity_ids)
    
    graph_hits = rank_by_embedding_in_set(
        query_embedding, graph_chunk_ids, vector_index
    )  # re-rank graph candidates by semantic score
    
    # Source 2: Vector search
    vector_hits = vector_index.search(query_embedding, top_k=top_k * 3)
    
    # Source 3: BM25
    bm25_hits = bm25_index.search(query, top_k=top_k * 3)
    
    # Reciprocal Rank Fusion
    scores: dict[str, float] = defaultdict(float)
    
    for rank, hit in enumerate(graph_hits[:top_k * 3]):
        scores[hit.chunk_id] += 1.0 / (rank + RRF_K)
    
    for rank, hit in enumerate(vector_hits):
        scores[hit.chunk_id] += 1.0 / (rank + RRF_K)
    
    for rank, hit in enumerate(bm25_hits):
        scores[hit.chunk_id] += 1.0 / (rank + RRF_K)
    
    # Sort by fused score
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    
    # Materialize passages with context expansion
    passages = []
    for rank, chunk_id in enumerate(ranked_ids[:top_k]):
        chunk = fetch_chunk_with_neighbors(chunk_id)  # include prev/next
        graph_context = None
        if chunk_id in graph_chunk_ids and graph_store.is_available():
            graph_context = fetch_local_subgraph(chunk_id, graph_store)
        
        passages.append(RankedPassage(
            rank=rank,
            score=scores[chunk_id],
            chunk=chunk,
            retrieval_method="fusion",
            graph_context=graph_context,
            ...
        ))
    
    return passages


def build_context(
    query: str,
    passages: list[RankedPassage],
    max_tokens: int = 4000,
) -> GenerationContext:
    
    formatted_parts = []
    source_map = {}
    token_budget = max_tokens
    
    for i, passage in enumerate(passages):
        # Include graph context summary before the passage text
        graph_summary = ""
        if passage.graph_context:
            graph_summary = summarize_subgraph(passage.graph_context)
        
        # Include neighboring chunks if budget allows
        full_text = passage.chunk.text
        if passage.chunk.prev_chunk_id and token_budget > 200:
            prev = fetch_chunk(passage.chunk.prev_chunk_id)
            full_text = prev.text[-200:] + " [...] " + full_text  # overlap context
        
        citation = f"[{i+1}] {passage.paper_title} — {passage.section_title}"
        part = f"{citation}\n{graph_summary}{full_text}"
        
        part_tokens = estimate_tokens(part)
        if part_tokens > token_budget:
            break
        
        formatted_parts.append(part)
        source_map[i] = citation
        token_budget -= part_tokens
    
    return GenerationContext(
        query=query,
        passages=passages[:len(formatted_parts)],
        formatted_context="\n\n---\n\n".join(formatted_parts),
        passage_count=len(formatted_parts),
        total_tokens_estimate=max_tokens - token_budget,
        source_map=source_map,
    )
```

#### Algorithm 4: Degraded Answer Generation (No LLM)

```python
def generate_degraded_answer(context: GenerationContext) -> GeneratedAnswer:
    """
    When no LLM is available, return a structured citation response.
    Never return null. Never return empty string.
    """
    if not context.passages:
        return GeneratedAnswer(
            answer="No relevant passages found in the corpus for this query.",
            is_synthesized=False,
            confidence=0.0,
            citations=[],
            grounded=True,  # trivially grounded — we're not claiming anything
        )
    
    lines = [
        f"Based on {context.passage_count} passages retrieved from the corpus:\n"
    ]
    citations = []
    
    for i, passage in enumerate(context.passages[:5]):
        # Extract the most relevant sentence from the passage
        best_sentence = extract_best_sentence(
            passage.chunk.text, context.query
        )
        lines.append(f"[{i+1}] {best_sentence}")
        lines.append(f"    Source: {passage.paper_title}, {passage.section_title}")
        citations.append({
            "index": i + 1,
            "paper_title": passage.paper_title,
            "doi": passage.doi,
            "section": passage.section_title,
            "score": passage.score,
        })
    
    lines.append(
        "\n(Full LLM synthesis unavailable — configure GOOGLE_API_KEY or "
        "ANTHROPIC_API_KEY for generated answers.)"
    )
    
    return GeneratedAnswer(
        answer="\n".join(lines),
        is_synthesized=False,
        confidence=min(context.passages[0].score, 0.7) if context.passages else 0.0,
        citations=citations,
        grounded=True,
    )
```

---

### D. Removal List

**Files to DELETE entirely:**

| File | Reason |
|---|---|
| `backend/graphrag/webapp.py` | 3-line wrapper that re-exports from `server.py` — just update imports |

**Functions/classes to DELETE within files:**

| Location | Item | Reason |
|---|---|---|
| `embeddings.py` | `HashingEmbedder` (as live fallback) | Non-semantic; corrupts retrieval |
| `embeddings.py` | `build_entity_embedder()` returning HashingEmbedder | Misleading default |
| `extraction.py` | `_GENE_SYMBOL_RE` (26 hardcoded genes) | Paper-specific, never generalizes |
| `extraction.py` | `_DATASET_DEFINITION_PATTERNS` | Paper-specific Theileria patterns |
| `extraction.py` | `_DATASET_DIRECT_PATTERNS` | Paper-specific TBL20/BL20/V/A |
| `extraction.py` | `_DATASET_COMPARISON_RE` | Paper-specific |
| `extraction.py` | `_DATASET_PAIR_RE` | Paper-specific |
| `corpus.py` | `lookup_mesh()` body | Routes to wrong API |
| `rag.py` | `@lru_cache` on `_query_graph()` | Broken on instance methods |
| `domain_knowledge.yaml` | Entire `dataset_registry` section | Theileria-specific |
| `domain_knowledge.yaml` | `concept_canonical_map` entries | From 2 test papers only |
| `domain_knowledge.yaml` | `concept_alias_map` entries for METTL3, WTAP etc. | From 2 test papers only |

**Tests to DELETE or rewrite:**

| Test | Reason |
|---|---|
| `test_local_index_prefers_the_relevant_article` | Passes only due to lexical overlap, not semantic retrieval |
| `test_paper_search_uses_abstract_embeddings` | Same issue — "theileria" appears verbatim in query |
| `test_biomedical_datasets_results_and_aliases_are_specific` | Tests paper-specific patterns that should be removed |

---

## SECTION 5: FAILURE MODES & TESTING STRATEGY

### What will break in real-world usage

#### At ingestion time:

| Scenario | Current behavior | Correct behavior |
|---|---|---|
| PubMed XML file | Silent empty PaperRecord | `ParserError` with format hint |
| Corrupted XML | `lxml` recover=True silently produces partial parse | Log partial parse, validate minimum fields |
| PDF input | Not handled | Queue for PDF parser, not silent skip |
| 10,000 paper corpus | Server OOM on startup | Async incremental ingestion with progress |
| Duplicate paper (same DOI) | Creates duplicate PaperRecord | Idempotent upsert by DOI |

#### At extraction time:

| Scenario | Current behavior | Correct behavior |
|---|---|---|
| Physics paper | Zero methods, zero results, zero datasets | Domain-agnostic patterns extract quantities and methods |
| Paper with no BL20/TBL20 | Zero result entities | Numeric extraction runs on all chunks |
| Gemini rate limit | Silent fallback to heuristics | Log degradation, cache and retry |
| Non-English text | Partially extracted (regex fires on any language) | Detect language, skip or route to multilingual model |

#### At retrieval time:

| Scenario | Current behavior | Correct behavior |
|---|---|---|
| Query with no exact token overlap | Poor results (HashingEmbedder) | Semantic similarity via SBERT |
| Query about entity from a different paper | Entity not connected to graph | Cross-paper entity resolution applied |
| Neo4j down | Results silently come from local index only | Log fallback, return `retrieval_source: "local"` in response |

#### At generation time:

| Scenario | Current behavior | Correct behavior |
|---|---|---|
| No API key | `{"answer": null}` returned | Structured citation list returned |
| Hallucinated answer | No detection | Grounding check with citation verification |

---

### Metrics to Track

**Extraction quality (per-paper, stored in SQLite):**
- `entity_count_by_type` — if any paper has 0 result entities, flag for review
- `avg_entity_confidence` — alert if below 0.6
- `chunk_coverage` — fraction of chunks with at least 1 entity — alert if below 0.5
- `extraction_pass_used` — track LLM vs local vs hybrid ratio

**Graph completeness (query Neo4j after each ingestion batch):**
```cypher
MATCH (p:Paper) WHERE p.pagerank IS NULL RETURN count(p) AS unscored_papers
MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS unindexed_chunks
MATCH (e:Entity) WHERE e.canonical_id IS NULL RETURN count(e) AS unresolved_entities
```

**Retrieval accuracy (offline evaluation, weekly):**
- Build a labeled evaluation set: 50 questions with known relevant papers
- Measure Recall@5 and Recall@10
- Alert if Recall@5 drops below 0.7
- Track separately: `recall_graph_only`, `recall_vector_only`, `recall_fusion` — fusion must be best

**Answer grounding (per query, production):**
- Track `grounded: bool` from `GeneratedAnswer`
- Track `unsupported_claims` count
- Alert if `grounded=False` rate exceeds 10% over a 1-hour window
- Log all `unsupported_claims` for offline review

---

### Test Strategy

**Unit tests (fast, no external dependencies):**
```python
# Test extraction is domain-agnostic
def test_extract_physics_quantities():
    chunk = Chunk(text="The measured resistivity was 2.3 × 10⁻⁸ Ω·m at 300 K.")
    result = local_extract(chunk, ...)
    assert any(e.entity_type == EntityType.RESULT for e in result.entities)
    result_entity = next(e for e in result.entities if e.entity_type == EntityType.RESULT)
    assert result_entity.properties["value"] == pytest.approx(2.3e-8, rel=0.01)

# Test chunking overlap is applied
def test_chunks_have_overlap():
    settings = Phase1Settings(chunk_size_words=50, chunk_overlap_words=10)
    paper = make_paper_with_long_section(words=200)
    chunks = chunk_article(paper, settings).chunks
    for i in range(1, len(chunks)):
        assert chunks[i].overlap_prev_words == 10

# Test parser fails loudly on wrong format
def test_parser_raises_on_pubmed_xml():
    with pytest.raises(ParserError, match="elsevier_xml"):
        parse_article("test_fixtures/pubmed_sample.xml")
```

**Integration tests (require SBERT model, no API keys):**
```python
def test_semantic_retrieval_with_sbert():
    # Must work without ANY API key
    embedder = SentenceTransformerEmbedder()
    index = LocalVectorIndex(documents=[paper_ev, paper_theileria], embedder=embedder)
    hits = index.search("battery charging infrastructure policy")
    assert hits[0].paper_id == paper_ev.paper_id
    # This should pass on semantic similarity, not just lexical overlap
```

**Regression tests (catch paper-specific hardcoding):**
```python
def test_extraction_finds_results_in_non_biomedical_paper():
    # A climate science paper — no TBL20, no METTL3, no known datasets
    paper = chunk_article(parse_article("test_fixtures/climate_science.xml"))
    extraction = extract_layer2(paper, use_gemini=False)
    result_entities = [e for e in extraction.entities if e.entity_type == "result"]
    assert len(result_entities) > 0, "No results extracted — dataset gate is blocking"
```

---

## SECTION 6: PRIORITY EXECUTION ORDER

```
WEEK 1 — Stop the bleeding (Phase 1)
  Day 1-2: Replace HashingEmbedder fallback with SBERT
  Day 3:   Implement chunk overlap
  Day 4:   Remove paper-specific dataset patterns, fix _local_results() gate
  Day 5:   Fix lru_cache bug, fix lookup_mesh(), add degraded answer mode

WEEK 2-3 — Domain generalization (Phase 2)
  Rewrite domain_knowledge.yaml as a general base
  Rewrite _local_methods() and _local_results() as schema-driven
  Add domain auto-detection

WEEK 4-5 — Schema-driven extraction (Phase 3)
  Build ExtractionSchema system
  Create general.yaml, biomedical.yaml, physics.yaml schemas
  Migrate extraction to use schema-selected prompts

WEEK 6-8 — Wire the graph (Phase 4)  ← MOST IMPORTANT
  Implement HybridRetriever
  Implement BM25Index
  Implement ContextBuilder with graph expansion
  Wire GraphRetrieval into search_service.py
  Add /api/graph/* endpoints

WEEK 9-10 — Production hardening (Phase 5)
  Async ingestion with incremental checkpointing
  Extraction cache (SQLite, keyed by chunk hash + model version)
  Circuit breaker persistence across restarts
  Evaluation harness + weekly metric jobs
```

---

*This system is architecturally sound in its structure but operationally non-functional in its graph layer, semantically broken in its embedding layer, and domain-locked in its extraction layer. All three failures must be addressed before this can be called a production Graph RAG system.*
