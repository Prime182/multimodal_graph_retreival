
**What is the end goal?**

A system where a researcher asks a question like *"What methods are used to study m6A methylation and what do they find?"* and gets a accurate, cited, trustworthy answer synthesized across dozens of papers — with the knowledge graph staying correct as more papers are added.

That means the system must:
- Ingest any domain paper correctly
- Extract entities that are actually right
- Build a graph where edges mean something real
- Retrieve the right chunks for a query
- Synthesize a grounded answer
- Know when it's wrong and fix it

---

**Does your pipeline do this today?**

```
Ingest any domain paper       → NO  (hardcoded to one domain)
Extract entities correctly    → PARTIALLY (Gemini extracts, but no validation)
Graph edges mean something    → NO  (HashingEmbedder breaks similarity edges)
Retrieve right chunks         → PARTIALLY (hashing embedder limits quality)
Synthesize grounded answer    → PARTIALLY (rag.py does this but no fact-checking)
Know when it's wrong          → NO  (zero self-correction anywhere)
```

---

**Is LangGraph used?**

No. Looking at your entire codebase — there is no LangGraph, no LangChain, no agent loop, no state machine. You have:

- A linear pipeline: parse → chunk → extract → store
- A single Gemini call per chunk with no retry logic
- A single synthesis call with no verification
- No feedback loops anywhere

This is the biggest gap. A production RAG system is not a pipeline — it's an **agentic loop with checkpoints**.

---

**The architecture you need, working backwards**

```
USER QUERY
    ↓
Query Understanding Agent
    ↓
Retrieval (hybrid: vector + graph traversal)
    ↓
Answer Synthesis
    ↓
Self-Verification Agent ← this is missing entirely
    ↓
If verification fails → Re-retrieve with refined query
    ↓
FINAL ANSWER with citations + confidence
```

And for ingestion:

```
NEW PAPER
    ↓
Parse + Chunk
    ↓
Extraction Agent (Gemini)
    ↓
Validation Agent ← missing
    ↓
If validation fails → Re-extract with corrections
    ↓
Canonicalization
    ↓
Graph Storage
    ↓
Consistency Check ← missing
```

---

**LangGraph is the right tool here — here's why**

LangGraph lets you define these loops as a state machine with explicit nodes, edges, and conditional routing. Your current pipeline is a straight line — LangGraph makes it a graph with cycles.

Here's what your ingestion graph should look like:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ExtractionState(TypedDict):
    paper: PaperRecord
    chunk: ChunkRecord
    section: SectionRecord
    raw_entities: list[dict]
    validated_entities: list[Layer2EntityRecord]
    validation_errors: list[str]
    retry_count: int
    extraction_quality_score: float
    domain: str

# Define the graph
extraction_graph = StateGraph(ExtractionState)

# Nodes
extraction_graph.add_node("detect_domain", detect_domain_node)
extraction_graph.add_node("extract", gemini_extraction_node)
extraction_graph.add_node("validate", validation_node)
extraction_graph.add_node("self_correct", self_correction_node)
extraction_graph.add_node("enrich", corpus_enrichment_node)
extraction_graph.add_node("finalize", finalize_node)

# Edges with conditional routing
extraction_graph.set_entry_point("detect_domain")
extraction_graph.add_edge("detect_domain", "extract")
extraction_graph.add_edge("extract", "validate")

# THIS is the self-correction loop
extraction_graph.add_conditional_edges(
    "validate",
    should_retry,  # returns "self_correct" or "enrich"
    {
        "self_correct": "self_correct",
        "enrich": "enrich",
    }
)
extraction_graph.add_edge("self_correct", "extract")  # loop back
extraction_graph.add_edge("enrich", "finalize")
extraction_graph.add_edge("finalize", END)
```

---

**The self-correction mechanism — what it actually looks like**

```python
def validation_node(state: ExtractionState) -> ExtractionState:
    """
    Validates extracted entities against rules.
    Sets validation_errors and extraction_quality_score.
    """
    errors = []
    entities = state["raw_entities"]
    
    for entity in entities:
        # Rule 1: Results must have a numeric value
        if entity.get("type") == "result":
            if not isinstance(entity.get("value"), (int, float)):
                errors.append(f"Result '{entity.get('name')}' missing numeric value")
        
        # Rule 2: Methods must not be generic words
        if entity.get("type") == "method":
            name = entity.get("name", "").lower()
            if name in {"analysis", "approach", "method", "system", "model"}:
                errors.append(f"Method '{name}' is too generic")
        
        # Rule 3: Claims must be complete sentences
        if entity.get("type") == "claim":
            text = entity.get("text", "")
            if len(text.split()) < 5:
                errors.append(f"Claim too short: '{text}'")
        
        # Rule 4: Concepts should not be longer than 5 words
        if entity.get("type") == "concept":
            if len(entity.get("name", "").split()) > 6:
                errors.append(f"Concept name suspiciously long: '{entity.get('name')}'")
        
        # Rule 5: Confidence must be in range
        conf = entity.get("confidence", 0)
        if not 0 <= conf <= 1:
            errors.append(f"Invalid confidence {conf} for '{entity.get('name')}'")
    
    # Quality score: what fraction of entities passed
    total = len(entities)
    if total == 0:
        quality = 0.0
    else:
        # Unique entities with errors
        error_entities = len(set(
            e.split("'")[1] for e in errors if "'" in e
        ))
        quality = max(0.0, 1.0 - (error_entities / total))
    
    return {
        **state,
        "validation_errors": errors,
        "extraction_quality_score": quality,
    }


def should_retry(state: ExtractionState) -> str:
    """Route to self-correction or proceed."""
    if state["retry_count"] >= 2:
        # Never retry more than twice — proceed with what we have
        return "enrich"
    
    if state["extraction_quality_score"] < 0.6:
        return "self_correct"
    
    if len(state["validation_errors"]) > 3:
        return "self_correct"
    
    return "enrich"


def self_correction_node(state: ExtractionState) -> ExtractionState:
    """
    Sends errors back to Gemini with correction instructions.
    This is the key node — it makes the system learn from its mistakes.
    """
    errors_text = "\n".join(f"- {e}" for e in state["validation_errors"])
    
    correction_prompt = f"""You previously extracted entities from this text but made errors.

ERRORS IN YOUR PREVIOUS EXTRACTION:
{errors_text}

ORIGINAL ENTITIES (with problems):
{json.dumps(state["raw_entities"], indent=2)}

TEXT:
{state["chunk"].text}

Please re-extract, fixing all the listed errors. 
- Remove entities that are too generic
- Add numeric values to results that are missing them
- Shorten concept names that are too long
- Ensure all claims are complete sentences

Return corrected JSON:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(correction_prompt)
    
    try:
        corrected = json.loads(response.text)
        return {
            **state,
            "raw_entities": corrected.get("entities", state["raw_entities"]),
            "retry_count": state["retry_count"] + 1,
        }
    except Exception:
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
        }
```

---

**The RAG query loop with self-correction**

Same pattern for query answering:

```python
class QueryState(TypedDict):
    question: str
    retrieved_chunks: list[dict]
    answer: str | None
    citations: list[str]
    verification_result: dict
    retry_count: int
    refined_query: str | None

query_graph = StateGraph(QueryState)

query_graph.add_node("retrieve", retrieval_node)
query_graph.add_node("synthesize", synthesis_node)
query_graph.add_node("verify", verification_node)       # is answer grounded?
query_graph.add_node("refine_query", query_refinement_node)  # fix the query
query_graph.add_node("respond", final_response_node)

query_graph.set_entry_point("retrieve")
query_graph.add_edge("retrieve", "synthesize")
query_graph.add_edge("synthesize", "verify")
query_graph.add_conditional_edges(
    "verify",
    should_refine,
    {
        "refine": "refine_query",
        "respond": "respond",
    }
)
query_graph.add_edge("refine_query", "retrieve")  # loop with better query
query_graph.add_edge("respond", END)


def verification_node(state: QueryState) -> QueryState:
    """
    Checks if the answer is actually grounded in retrieved chunks.
    This prevents hallucination.
    """
    if not state["answer"] or not state["retrieved_chunks"]:
        return {**state, "verification_result": {"grounded": False, "reason": "no answer or chunks"}}
    
    chunk_texts = "\n\n".join(
        f"[{i+1}] {c['text']}" 
        for i, c in enumerate(state["retrieved_chunks"][:5])
    )
    
    verify_prompt = f"""Given these source passages and an answer, determine if the answer is fully grounded.

SOURCES:
{chunk_texts}

ANSWER:
{state["answer"]}

Answer these questions:
1. Is every factual claim in the answer supported by at least one source? (yes/no)
2. Are there any statements in the answer NOT found in the sources? List them.
3. Confidence the answer is fully grounded (0.0-1.0)

Return JSON: {{"grounded": bool, "unsupported_claims": [...], "confidence": float}}"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(verify_prompt)
    
    try:
        result = json.loads(response.text)
        return {**state, "verification_result": result}
    except Exception:
        return {**state, "verification_result": {"grounded": True, "confidence": 0.5}}


def should_refine(state: QueryState) -> str:
    if state["retry_count"] >= 2:
        return "respond"
    
    verification = state["verification_result"]
    if not verification.get("grounded") or verification.get("confidence", 1.0) < 0.6:
        return "refine"
    
    return "respond"
```

---

**Fail-safe mechanisms — the full checklist**

**Ingestion failures**

```python
# Every paper ingestion needs a status record
class PaperIngestionStatus(TypedDict):
    paper_id: str
    status: Literal["pending", "parsing", "chunking", "extracting", "storing", "complete", "failed"]
    error: str | None
    retry_count: int
    last_attempted: float
    extraction_quality: float

# Store this in a simple SQLite table
# If status is "failed" after 3 retries, alert and skip — don't block the pipeline
```

**Circuit breaker for external APIs**

```python
class CircuitBreaker:
    """
    Stops calling a failing API after N consecutive failures.
    Resets after a cooldown period.
    """
    def __init__(self, failure_threshold: int = 5, cooldown: int = 300):
        self._failures: dict[str, int] = {}
        self._last_failure: dict[str, float] = {}
        self._threshold = failure_threshold
        self._cooldown = cooldown
    
    def is_open(self, service: str) -> bool:
        failures = self._failures.get(service, 0)
        if failures < self._threshold:
            return False
        # Check if cooldown has passed
        last = self._last_failure.get(service, 0)
        if time.time() - last > self._cooldown:
            self._failures[service] = 0  # reset
            return False
        return True
    
    def record_failure(self, service: str):
        self._failures[service] = self._failures.get(service, 0) + 1
        self._last_failure[service] = time.time()
    
    def record_success(self, service: str):
        self._failures[service] = 0

_circuit_breaker = CircuitBreaker()

# Use in CorpusClient:
def lookup_gene(self, symbol: str) -> CorpusMatch:
    if _circuit_breaker.is_open("ncbi"):
        return CorpusMatch(found=False, label=symbol)  # degrade gracefully
    try:
        result = self._call_ncbi(symbol)
        _circuit_breaker.record_success("ncbi")
        return result
    except Exception:
        _circuit_breaker.record_failure("ncbi")
        return CorpusMatch(found=False, label=symbol)
```

**Gemini failure handling**

```python
# Current code:
except Exception:
    return None  # SILENT FAILURE

# Should be:
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    retry=tenacity.retry_if_exception_type((requests.Timeout, json.JSONDecodeError)),
    reraise=False,
)
def _call_gemini_with_retry(prompt: str, model) -> dict:
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
        request_options={"timeout": 30},
    )
    # Strip markdown fences Gemini sometimes adds
    text = re.sub(r"```json\s*|\s*```", "", response.text).strip()
    return json.loads(text)
```

---

**What the production-ready system looks like end to end**

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                          │
│  XML Paper                                               │
│     ↓                                                    │
│  Parser (lxml) ──────────────────────── status: parsing  │
│     ↓                                                    │
│  Chunker ────────────────────────────── status: chunking │
│     ↓                                                    │
│  Domain Detector (Gemini, fast)                          │
│     ↓                                                    │
│  For each chunk:                                         │
│    LangGraph Extraction Loop                             │
│      extract → validate → [self_correct →] enrich        │
│      max 2 retries, circuit breaker on Gemini            │
│     ↓                                                    │
│  SQLite Cache (corpus enrichment results)                │
│     ↓                                                    │
│  Entity Canonicalization (real embeddings)               │
│     ↓                                                    │
│  Neo4j Storage ─────────────────────── status: complete  │
│     ↓                                                    │
│  Post-ingestion consistency check                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                        │
│                                                          │
│  User Question                                           │
│     ↓                                                    │
│  Query Classifier (what type of question is this?)       │
│     ↓                                                    │
│  LangGraph Query Loop                                    │
│    retrieve → synthesize → verify → [refine →]           │
│    max 2 retries                                         │
│     ↓                                                    │
│  Answer + Citations + Confidence                         │
│     ↓                                                    │
│  Confidence < 0.5 → "I don't have enough information"    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                         │
│                                                          │
│  LangFuse: every LLM call logged                        │
│  /api/corpus-misses: what enrichment is failing          │
│  /api/ingestion-status: per-paper pipeline status        │
│  /api/extraction-quality: quality scores over time       │
│  /api/graph-health: entity counts, edge counts,          │
│                     null alias rate, orphan nodes         │
└─────────────────────────────────────────────────────────┘
```

---

**What to build in what order**

Starting from today's codebase:

**Week 1 — Stop the bleeding**
- Replace `HashingEmbedder` with Gemini embeddings for canonicalization and claim edges
- Add Pydantic validation on Gemini output
- Add SQLite cache to `CorpusClient` with real API calls
- Add retry with `tenacity` on Gemini calls

**Week 2 — Self-correction**
- Install LangGraph
- Wrap per-chunk extraction in the LangGraph extraction loop
- Add validation node + self-correction node
- Wire retry routing

**Week 3 — Query quality**
- Wrap query synthesis in LangGraph query loop
- Add verification node
- Add query refinement on low-confidence answers

**Week 4 — Observability + domain expansion**
- Expose `/api/corpus-misses` and `/api/extraction-quality`
- Add per-paper ingestion status table
- Add circuit breakers on all external APIs
- Replace hardcoded dicts with YAML configs loaded at startup

---

