# STEM Knowledge Graph & Hybrid RAG — Full Implementation Plan
### Stack: Neo4j · Gemini · LangGraph · React

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Phase 1 — Document Spine (Layer 1)](#3-phase-1--document-spine-layer-1)
4. [Phase 2 — Entity Extraction (Layer 2)](#4-phase-2--entity-extraction-layer-2)
5. [Phase 3 — Semantic Edges & Validation (Layer 3)](#5-phase-3--semantic-edges--validation-layer-3)
6. [Phase 4 — Indexing Strategy](#6-phase-4--indexing-strategy)
7. [Phase 5 — Hybrid Retrieval Engine](#7-phase-5--hybrid-retrieval-engine)
8. [Phase 6 — LangGraph Agentic Pipeline](#8-phase-6--langgraph-agentic-pipeline)
9. [Phase 7 — React Frontend](#9-phase-7--react-frontend)
10. [Testing & Validation Strategy](#10-testing--validation-strategy)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Risk Mitigations & Known Pitfalls](#12-risk-mitigations--known-pitfalls)

---

## 1. Project Overview & Architecture

### What We Are Building

An end-to-end system that:
- Ingests STEM PDFs and XML documents
- Parses and embeds multimodal content (text, equations, tables, figures)
- Structures knowledge into a 3-layer Neo4j graph
- Uses LangGraph agents for autonomous extraction, canonicalization, and validation
- Executes hybrid retrieval combining dense vector similarity with graph-theoretic authority scoring
- Exposes a React frontend for querying and graph exploration

### 3-Layer Architecture at a Glance

```
Layer 1 — Document Spine
  Paper → Section → Chunk (with NEXT edges)
  Paper → Author, Journal

Layer 2 — Knowledge Entities
  Chunk → Concept, Method, Claim, Result, Equation, Figure/Table
  Result → Dataset, Metric

Layer 3 — Semantic & Provenance Edges
  MENTIONS, GROUNDED_IN, IS_A, IMPROVES, SOLVES, USES
  SUPPORTS, CONTRADICTS, CITES, MAPS_TO
  Every edge carries: confidence, source_chunk_id, extractor_model
```

### Technology Choices

| Concern | Tool | Reason |
|---|---|---|
| Graph database | Neo4j 5.x (AuraDB or self-hosted) | Native vector indexes + GDS plugin for PageRank |
| Embeddings | `gemini-embedding-2-preview` | Natively multimodal, MRL support |
| LLM extraction | `gemini-2.0-flash` | Fast, cheap per-chunk extraction |
| LLM synthesis | `gemini-2.5-pro` | Complex reasoning for canonicalization |
| PDF parsing | `unstructured-io` | Best-in-class multimodal partitioning |
| Agentic orchestration | `langgraph` | Cyclic graph execution with state |
| Backend API | FastAPI | Async, easy Neo4j driver integration |
| Frontend | React + Vite | Simple query UI + optional graph explorer |

---

## 2. Prerequisites & Environment Setup

### 2.1 System Requirements

- Python 3.11+
- Node.js 20+
- Docker (for local Neo4j)
- 16 GB RAM minimum (32 GB recommended for large corpora)

### 2.2 Python Dependencies

```bash
# Core
pip install neo4j==5.20.0
pip install google-generativeai==0.8.0
pip install langgraph==0.2.0
pip install langchain-google-genai==2.0.0

# Parsing
pip install unstructured[all-docs]==0.16.0
pip install pdf2image pillow pytesseract

# Backend
pip install fastapi uvicorn python-multipart
pip install pydantic-settings python-dotenv

# Utilities
pip install tenacity tqdm rich loguru
```

### 2.3 Neo4j Setup (Docker)

```bash
docker run \
  --name neo4j-stem \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.* \
  -v $PWD/neo4j/data:/data \
  -v $PWD/neo4j/logs:/logs \
  neo4j:5.20.0-enterprise
```

> **Important:** The Graph Data Science (GDS) plugin is required for PageRank in Phase 5. Always start Neo4j with it enabled.

### 2.4 Environment Variables

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Model selection
EMBED_MODEL=models/gemini-embedding-2-preview
EMBED_DIM=768           # 768 | 1536 | 3072 — tune per cost/quality needs
EXTRACT_MODEL=gemini-2.0-flash
REASON_MODEL=gemini-2.5-pro

# Ingest limits
MAX_RETRY_CYCLES=2      # Self-correction loop cap
MIN_CONFIDENCE=0.70     # Below this, trigger retry
CHUNK_SIZE=512          # Tokens per chunk
CHUNK_OVERLAP=64
```

### 2.5 Project Structure

```
stem-kg-rag/
├── ingest/
│   ├── parser.py           # Unstructured partitioning
│   ├── chunker.py          # Text splitting + salience scoring
│   ├── embedder.py         # Gemini embedding wrapper
│   └── multimodal.py       # VLM summaries for tables/figures
├── graph/
│   ├── schema.py           # Constraint + index creation
│   ├── layer1.py           # Document spine ingestion
│   ├── layer2.py           # Entity node creation
│   └── layer3.py           # Edge creation + provenance
├── agents/
│   ├── extractor.py        # LangGraph extraction agent
│   ├── canonicalizer.py    # Alias merge + ontology link agent
│   └── validator.py        # Self-correction + consensus agent
├── retrieval/
│   ├── query_engine.py     # Hybrid Cypher query builder
│   └── reranker.py         # Score fusion (cosine + PageRank)
├── api/
│   ├── main.py             # FastAPI app
│   └── routers/
│       ├── ingest.py
│       └── query.py
├── frontend/               # React app (Phase 7)
├── tests/
└── .env
```

---

## 3. Phase 1 — Document Spine (Layer 1)

Build this first. A working spine + basic vector retrieval gives you a usable system before any agentic complexity is added.

### 3.1 Neo4j Schema — Constraints & Indexes

Run once on a fresh database.

```cypher
// ── Uniqueness constraints ──────────────────────────────────────────────────
CREATE CONSTRAINT paper_doi    IF NOT EXISTS FOR (p:Paper)   REQUIRE p.doi    IS UNIQUE;
CREATE CONSTRAINT author_orcid IF NOT EXISTS FOR (a:Author)  REQUIRE a.orcid  IS UNIQUE;
CREATE CONSTRAINT journal_issn IF NOT EXISTS FOR (j:Journal) REQUIRE j.issn   IS UNIQUE;
CREATE CONSTRAINT chunk_id     IF NOT EXISTS FOR (c:Chunk)   REQUIRE c.id     IS UNIQUE;
CREATE CONSTRAINT concept_id   IF NOT EXISTS FOR (c:Concept) REQUIRE c.id     IS UNIQUE;
CREATE CONSTRAINT method_id    IF NOT EXISTS FOR (m:Method)  REQUIRE m.id     IS UNIQUE;
CREATE CONSTRAINT claim_id     IF NOT EXISTS FOR (c:Claim)   REQUIRE c.id     IS UNIQUE;
CREATE CONSTRAINT result_id    IF NOT EXISTS FOR (r:Result)  REQUIRE r.id     IS UNIQUE;
CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name   IS UNIQUE;
CREATE CONSTRAINT metric_name  IF NOT EXISTS FOR (m:Metric)  REQUIRE m.name   IS UNIQUE;
CREATE CONSTRAINT equation_id  IF NOT EXISTS FOR (e:Equation)REQUIRE e.id     IS UNIQUE;
```

### 3.2 PDF Parsing

```python
# ingest/parser.py
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Text, Title, NarrativeText, Table, Image, Formula
)
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

@dataclass
class ParsedDocument:
    paper_id: str
    doi: Optional[str]
    title: str
    sections: List[dict] = field(default_factory=list)
    tables: List[dict] = field(default_factory=list)
    figures: List[dict] = field(default_factory=list)
    equations: List[dict] = field(default_factory=list)

def parse_pdf(path: str | Path) -> ParsedDocument:
    elements = partition(
        filename=str(path),
        strategy="hi_res",          # OCR + layout model
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir="./tmp/images",
        languages=["eng"],
    )

    doc = ParsedDocument(
        paper_id=str(uuid.uuid4()),
        doi=None,
        title="",
    )

    current_section = None
    section_order = 0

    for el in elements:
        el_type = type(el).__name__
        text = el.text.strip()
        metadata = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}

        if el_type == "Title":
            if not doc.title:
                doc.title = text
            else:
                current_section = {
                    "id": str(uuid.uuid4()),
                    "title": text,
                    "order": section_order,
                    "section_type": infer_section_type(text),
                    "chunks": [],
                }
                doc.sections.append(current_section)
                section_order += 1

        elif el_type in ("NarrativeText", "Text"):
            if current_section is not None:
                current_section["chunks"].append({
                    "id": str(uuid.uuid4()),
                    "raw_text": text,
                    "chunk_type": "narrative",
                    "page_number": metadata.get("page_number"),
                })

        elif el_type == "Table":
            doc.tables.append({
                "id": str(uuid.uuid4()),
                "html": metadata.get("text_as_html", ""),
                "caption": text,
                "page_number": metadata.get("page_number"),
                "section_id": current_section["id"] if current_section else None,
            })

        elif el_type == "Image":
            doc.figures.append({
                "id": str(uuid.uuid4()),
                "path": metadata.get("image_path", ""),
                "caption": text,
                "page_number": metadata.get("page_number"),
                "section_id": current_section["id"] if current_section else None,
            })

        elif el_type == "Formula":
            doc.equations.append({
                "id": str(uuid.uuid4()),
                "latex": text,
                "page_number": metadata.get("page_number"),
                "section_id": current_section["id"] if current_section else None,
            })

    return doc


SECTION_KEYWORDS = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "related_work": ["related work", "prior work", "literature"],
    "methods": ["method", "approach", "methodology", "framework"],
    "experiments": ["experiment", "setup", "implementation"],
    "results": ["result", "evaluation", "performance", "benchmark"],
    "discussion": ["discussion", "analysis", "ablation"],
    "conclusion": ["conclusion", "future work", "summary"],
    "supplementary": ["supplement", "appendix"],
}

def infer_section_type(title: str) -> str:
    t = title.lower()
    for section_type, keywords in SECTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return section_type
    return "other"
```

### 3.3 Text Chunking

```python
# ingest/chunker.py
import re
from typing import List

def chunk_section(section: dict, max_tokens: int = 512, overlap: int = 64) -> List[dict]:
    """
    Splits section text into overlapping chunks, preserving sentence boundaries.
    Returns chunks with sequential order indices.
    """
    raw_chunks = section.get("chunks", [])
    # Flatten all narrative text from pre-split chunks
    full_text = " ".join(c["raw_text"] for c in raw_chunks)

    sentences = split_sentences(full_text)
    chunks = []
    current_tokens = []
    current_text_parts = []
    order = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)

        if sum(estimate_tokens(p) for p in current_text_parts) + sent_tokens > max_tokens and current_text_parts:
            chunk_text = " ".join(current_text_parts)
            chunks.append({
                "id": f"{section['id']}_chunk_{order}",
                "raw_text": chunk_text,
                "order": order,
                "chunk_type": classify_chunk(chunk_text),
                "salience_score": 0.0,   # Filled by LLM in Phase 2
                "sentence_count": len(current_text_parts),
                "lang": "en",
            })
            order += 1
            # Overlap: keep last N tokens worth of sentences
            overlap_parts = []
            overlap_count = 0
            for part in reversed(current_text_parts):
                t = estimate_tokens(part)
                if overlap_count + t > overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_count += t
            current_text_parts = overlap_parts

        current_text_parts.append(sent)

    # Final chunk
    if current_text_parts:
        chunk_text = " ".join(current_text_parts)
        chunks.append({
            "id": f"{section['id']}_chunk_{order}",
            "raw_text": chunk_text,
            "order": order,
            "chunk_type": classify_chunk(chunk_text),
            "salience_score": 0.0,
            "sentence_count": len(current_text_parts),
            "lang": "en",
        })

    return chunks

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3  # Rough approximation

CHUNK_TYPE_PATTERNS = {
    "definition":    r"\b(is defined as|refers to|means that|is a type of)\b",
    "procedure":     r"\b(we first|then we|next,|finally,|algorithm|step \d)\b",
    "result_report": r"\b(achieves|outperforms|accuracy of|f1 score|bleu|improvement)\b",
    "limitation":    r"\b(limitation|drawback|however,|cannot|fails to)\b",
    "hypothesis":    r"\b(we hypothesize|we propose|we conjecture|we expect)\b",
}

def classify_chunk(text: str) -> str:
    t = text.lower()
    for chunk_type, pattern in CHUNK_TYPE_PATTERNS.items():
        if re.search(pattern, t):
            return chunk_type
    return "narrative"
```

### 3.4 Gemini Embeddings

```python
# ingest/embedder.py
import os
import time
from typing import List, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-2-preview")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "768"))

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
) -> List[List[float]]:
    """
    Batch embed texts using Gemini's embedding model.
    task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | SEMANTIC_SIMILARITY
    """
    dim = output_dimensionality or EMBED_DIM
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type=task_type,
        output_dimensionality=dim,
    )
    return result["embedding"] if isinstance(texts, str) else result["embedding"]

def embed_query(text: str) -> List[float]:
    """For query-time embedding — uses RETRIEVAL_QUERY task type."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]

def embed_image_summary(summary: str) -> List[float]:
    """Embed a VLM-generated image/table summary for multimodal retrieval."""
    return embed_texts([summary], task_type="RETRIEVAL_DOCUMENT")[0]
```

### 3.5 VLM Summaries for Tables and Figures

```python
# ingest/multimodal.py
import base64
import os
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def summarize_figure(image_path: str, caption: str = "") -> str:
    """Generate a retrieval-optimized natural language summary of a figure."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lstrip(".")
    mime = f"image/{ext if ext in ('png','jpg','jpeg','webp') else 'png'}"

    prompt = (
        "You are summarizing a scientific figure for a retrieval system. "
        "Describe what this figure shows in 2-4 sentences. "
        "Be specific about axes, trends, comparisons, and key values. "
        f"{'Caption: ' + caption if caption else ''} "
        "Output only the summary text, no preamble."
    )

    response = model.generate_content([
        {"mime_type": mime, "data": img_data},
        prompt,
    ])
    return response.text.strip()

def summarize_table(html: str, caption: str = "") -> str:
    """Generate a retrieval-optimized summary of a table from its HTML."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = (
        "You are summarizing a scientific table for a retrieval system. "
        "The table is given as HTML. Describe its structure, column meanings, "
        "and key values or trends in 3-5 sentences. Be specific about numbers. "
        f"{'Caption: ' + caption if caption else ''}\n\nTable HTML:\n{html}\n\n"
        "Output only the summary text, no preamble."
    )
    response = model.generate_content(prompt)
    return response.text.strip()
```

### 3.6 Layer 1 Graph Ingestion

```python
# graph/layer1.py
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def ingest_paper(doc, abstract_embedding: list, session=None):
    """Merge Paper node. Uses MERGE on doi to be idempotent."""
    with driver.session() as s:
        s.run("""
            MERGE (p:Paper {doi: $doi})
            SET p.id            = $id,
                p.title         = $title,
                p.year          = $year,
                p.venue         = $venue,
                p.field_of_study = $field_of_study,
                p.abstract_embedding = $embedding,
                p.open_access_url    = $url
        """, doi=doc.doi or doc.paper_id, id=doc.paper_id,
             title=doc.title, year=doc.year, venue=doc.venue,
             field_of_study=doc.field_of_study,
             embedding=abstract_embedding, url=doc.url)

def ingest_authors(paper_doi: str, authors: list):
    with driver.session() as s:
        for author in authors:
            s.run("""
                MERGE (a:Author {orcid: $orcid})
                SET a.name        = $name,
                    a.affiliation = $affiliation,
                    a.h_index     = $h_index,
                    a.country     = $country
                WITH a
                MATCH (p:Paper {doi: $doi})
                MERGE (p)-[:AUTHORED_BY]->(a)
            """, orcid=author["orcid"], name=author["name"],
                 affiliation=author.get("affiliation", ""),
                 h_index=author.get("h_index", 0),
                 country=author.get("country", ""),
                 doi=paper_doi)

def ingest_journal(paper_doi: str, journal: dict):
    with driver.session() as s:
        s.run("""
            MERGE (j:Journal {issn: $issn})
            SET j.name          = $name,
                j.publisher     = $publisher,
                j.impact_factor = $impact_factor,
                j.open_access   = $open_access
            WITH j
            MATCH (p:Paper {doi: $doi})
            MERGE (p)-[:PUBLISHED_IN]->(j)
        """, issn=journal["issn"], name=journal["name"],
             publisher=journal.get("publisher", ""),
             impact_factor=journal.get("impact_factor", 0.0),
             open_access=journal.get("open_access", False),
             doi=paper_doi)

def ingest_sections_and_chunks(paper_doi: str, sections: list, chunk_embeddings: dict):
    """
    chunk_embeddings: {chunk_id: [float, ...]}
    Creates Section nodes, Chunk nodes, CONTAINS edges, and NEXT edges between chunks.
    """
    with driver.session() as s:
        for section in sections:
            # Create Section
            s.run("""
                MATCH (p:Paper {doi: $doi})
                MERGE (sec:Section {id: $sec_id})
                SET sec.title         = $title,
                    sec.order         = $order,
                    sec.section_type  = $section_type,
                    sec.key_sentence  = $key_sentence,
                    sec.summary_embedding = $embedding
                MERGE (p)-[:HAS_SECTION]->(sec)
            """, doi=paper_doi,
                 sec_id=section["id"],
                 title=section["title"],
                 order=section["order"],
                 section_type=section["section_type"],
                 key_sentence=section.get("key_sentence", ""),
                 embedding=chunk_embeddings.get(f"sec_{section['id']}", []))

            # Create Chunks + CONTAINS edges
            chunks = section.get("flattened_chunks", [])
            for i, chunk in enumerate(chunks):
                s.run("""
                    MATCH (sec:Section {id: $sec_id})
                    MERGE (c:Chunk {id: $cid})
                    SET c.raw_text       = $text,
                        c.text_vector    = $vector,
                        c.chunk_type     = $ctype,
                        c.salience_score = $salience,
                        c.sentence_count = $scount,
                        c.lang           = $lang,
                        c.order          = $order
                    MERGE (sec)-[:CONTAINS]->(c)
                """, sec_id=section["id"],
                     cid=chunk["id"],
                     text=chunk["raw_text"],
                     vector=chunk_embeddings.get(chunk["id"], []),
                     ctype=chunk["chunk_type"],
                     salience=chunk.get("salience_score", 0.5),
                     scount=chunk.get("sentence_count", 1),
                     lang=chunk.get("lang", "en"),
                     order=chunk["order"])

                # NEXT edge
                if i > 0:
                    prev_id = chunks[i - 1]["id"]
                    s.run("""
                        MATCH (prev:Chunk {id: $prev_id}), (curr:Chunk {id: $curr_id})
                        MERGE (prev)-[:NEXT]->(curr)
                    """, prev_id=prev_id, curr_id=chunk["id"])
```

---

## 4. Phase 2 — Entity Extraction (Layer 2)

### 4.1 Extraction Prompt Design

```python
# agents/extractor.py — Prompt templates

EXTRACTION_SYSTEM_PROMPT = """
You are a scientific knowledge extractor. Given a text chunk from a research paper,
extract all scientific entities. Return ONLY valid JSON, no preamble.

Entity types to extract:
- concept: A scientific idea or phenomenon (e.g., "attention mechanism", "oxidative stress")
- method: A procedure or technique someone performed (e.g., "gradient descent", "X-ray crystallography")
- claim: One falsifiable statement (e.g., "ResNet-50 achieves 76.1% top-1 accuracy on ImageNet")
- result: A quantitative claim — must have a value, a dataset, and a metric
- equation: A mathematical formula (as LaTeX)

For each entity output:
{
  "entities": [
    {
      "type": "concept|method|claim|result|equation",
      "name": "...",                      // for concept/method
      "text": "...",                      // for claim (full statement)
      "claim_type": "finding|hypothesis|limitation|future_work",  // for claim
      "value": 0.0,                       // for result
      "unit": "...",                      // for result
      "dataset": "...",                   // for result
      "metric": "...",                    // for result
      "condition": "...",                 // for result (hardware, splits, etc.)
      "latex": "...",                     // for equation
      "plain_desc": "...",               // for equation (natural language description)
      "is_loss_fn": false,               // for equation
      "aliases": [],                      // known alternate names
      "confidence": 0.0                   // your confidence 0.0–1.0
    }
  ]
}
"""

SALIENCE_PROMPT = """
Rate the information density of this text chunk from 0.0 to 1.0.
0.0 = pure boilerplate ("In this paper we present our approach...")
1.0 = dense novel claims, specific measurements, key definitions
Return only a JSON object: {"salience_score": 0.0}
"""
```

### 4.2 Extraction Pipeline

```python
# agents/extractor.py
import json
import os
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
extract_model = genai.GenerativeModel(os.getenv("EXTRACT_MODEL", "gemini-2.0-flash"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def extract_entities_from_chunk(chunk_text: str, chunk_id: str) -> dict:
    """
    Returns extracted entities with confidence scores.
    Falls back to empty list on parse failure (logged, not raised).
    """
    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nChunk ID: {chunk_id}\n\nText:\n{chunk_text}"
    response = extract_model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Attempt to extract JSON from response
        import re
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"entities": [], "parse_error": True}

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def score_salience(chunk_text: str) -> float:
    response = extract_model.generate_content(
        f"{SALIENCE_PROMPT}\n\nText:\n{chunk_text}",
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    data = json.loads(response.text)
    return float(data.get("salience_score", 0.5))
```

### 4.3 Equation LaTeX Description

```python
# agents/extractor.py (continued)

EQUATION_DESC_PROMPT = """
Given this LaTeX equation: {latex}

Write a natural language description suitable for retrieval by non-experts.
Focus on: what the equation computes, what its variables mean, and where it's used.
Output only the description text, 1-3 sentences.
"""

def describe_equation(latex: str) -> str:
    response = extract_model.generate_content(
        EQUATION_DESC_PROMPT.format(latex=latex),
        generation_config={"temperature": 0.2},
    )
    return response.text.strip()
```

### 4.4 Layer 2 Graph Ingestion

```python
# graph/layer2.py
from neo4j import GraphDatabase
import os, uuid

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def ingest_entities(chunk_id: str, entities: list, entity_embeddings: dict):
    """
    entity_embeddings: {entity_id: [float, ...]}
    Creates all entity nodes and MENTIONS edges from the source chunk.
    """
    with driver.session() as s:
        for ent in entities:
            etype = ent["type"]
            eid = str(uuid.uuid4())

            if etype == "concept":
                s.run("""
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.id       = $id,
                                  c.aliases  = $aliases,
                                  c.ontology = $ontology,
                                  c.embedding = $embedding
                    WITH c
                    MATCH (chunk:Chunk {id: $chunk_id})
                    MERGE (chunk)-[m:MENTIONS]->(c)
                    SET m.confidence     = $confidence,
                        m.source_chunk_id = $chunk_id,
                        m.extractor_model = $model
                """, name=ent["name"], id=eid,
                     aliases=ent.get("aliases", []),
                     ontology=ent.get("ontology", ""),
                     embedding=entity_embeddings.get(eid, []),
                     chunk_id=chunk_id,
                     confidence=ent.get("confidence", 0.5),
                     model=os.getenv("EXTRACT_MODEL"))

            elif etype == "method":
                s.run("""
                    MERGE (m:Method {name: $name})
                    ON CREATE SET m.id            = $id,
                                  m.aliases       = $aliases,
                                  m.type          = $mtype,
                                  m.first_paper_id = $first_paper,
                                  m.embedding     = $embedding
                    WITH m
                    MATCH (chunk:Chunk {id: $chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(m)
                    SET mn.confidence     = $confidence,
                        mn.source_chunk_id = $chunk_id,
                        mn.extractor_model = $model
                """, name=ent["name"], id=eid,
                     aliases=ent.get("aliases", []),
                     mtype=ent.get("method_type", "algorithmic"),
                     first_paper=ent.get("first_paper_id", ""),
                     embedding=entity_embeddings.get(eid, []),
                     chunk_id=chunk_id,
                     confidence=ent.get("confidence", 0.5),
                     model=os.getenv("EXTRACT_MODEL"))

            elif etype == "claim":
                s.run("""
                    CREATE (cl:Claim {id: $id})
                    SET cl.text        = $text,
                        cl.claim_type  = $ctype,
                        cl.confidence  = $confidence,
                        cl.embedding   = $embedding
                    WITH cl
                    MATCH (chunk:Chunk {id: $chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(cl)
                    SET mn.confidence     = $confidence,
                        mn.source_chunk_id = $chunk_id,
                        mn.extractor_model = $model
                    MERGE (cl)-[g:GROUNDED_IN]->(chunk)
                    SET g.confidence      = $confidence,
                        g.extractor_model = $model
                """, id=eid,
                     text=ent["text"],
                     ctype=ent.get("claim_type", "finding"),
                     confidence=ent.get("confidence", 0.5),
                     embedding=entity_embeddings.get(eid, []),
                     chunk_id=chunk_id,
                     model=os.getenv("EXTRACT_MODEL"))

            elif etype == "result":
                # Create Result, Dataset, Metric, and their relationships
                s.run("""
                    CREATE (r:Result {id: $id})
                    SET r.value     = $value,
                        r.unit      = $unit,
                        r.condition = $condition

                    MERGE (d:Dataset {name: $dataset})
                    MERGE (m:Metric  {name: $metric})
                    SET m.higher_is_better = $higher_is_better,
                        m.category         = $metric_cat

                    MERGE (r)-[:MEASURED_ON]->(d)
                    MERGE (r)-[:USING_METRIC]->(m)

                    WITH r
                    MATCH (chunk:Chunk {id: $chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(r)
                    SET mn.confidence     = $confidence,
                        mn.source_chunk_id = $chunk_id,
                        mn.extractor_model = $model
                """, id=eid,
                     value=ent.get("value", 0.0),
                     unit=ent.get("unit", ""),
                     condition=ent.get("condition", ""),
                     dataset=ent.get("dataset", "unknown"),
                     metric=ent.get("metric", "unknown"),
                     higher_is_better=ent.get("higher_is_better", True),
                     metric_cat=ent.get("metric_category", "classification"),
                     chunk_id=chunk_id,
                     confidence=ent.get("confidence", 0.5),
                     model=os.getenv("EXTRACT_MODEL"))

            elif etype == "equation":
                s.run("""
                    CREATE (eq:Equation {id: $id})
                    SET eq.latex      = $latex,
                        eq.plain_desc = $plain_desc,
                        eq.domain     = $domain,
                        eq.is_loss_fn = $is_loss_fn,
                        eq.desc_embedding = $embedding
                    WITH eq
                    MATCH (chunk:Chunk {id: $chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(eq)
                    SET mn.confidence     = $confidence,
                        mn.source_chunk_id = $chunk_id,
                        mn.extractor_model = $model
                """, id=eid,
                     latex=ent.get("latex", ""),
                     plain_desc=ent.get("plain_desc", ""),
                     domain=ent.get("domain", "mathematics"),
                     is_loss_fn=ent.get("is_loss_fn", False),
                     embedding=entity_embeddings.get(eid, []),
                     chunk_id=chunk_id,
                     confidence=ent.get("confidence", 0.5),
                     model=os.getenv("EXTRACT_MODEL"))
```

---

## 5. Phase 3 — Semantic Edges & Validation (Layer 3)

### 5.1 Scientific Relationship Edges

```python
# graph/layer3.py
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def create_semantic_edges(edges: list):
    """
    edges: list of dicts with keys:
      from_id, from_label, to_id, to_label,
      rel_type, confidence, source_chunk_id, extractor_model
    
    Supported rel_types: IMPROVES, SOLVES, USES, IS_A, CITES, MAPS_TO
    """
    with driver.session() as s:
        for edge in edges:
            cypher = f"""
                MATCH (a:{edge['from_label']} {{id: $from_id}})
                MATCH (b:{edge['to_label']}  {{id: $to_id}})
                MERGE (a)-[r:{edge['rel_type']}]->(b)
                SET r.confidence      = $confidence,
                    r.source_chunk_id = $source_chunk_id,
                    r.extractor_model = $extractor_model
            """
            s.run(cypher,
                  from_id=edge["from_id"],
                  to_id=edge["to_id"],
                  confidence=edge.get("confidence", 0.5),
                  source_chunk_id=edge.get("source_chunk_id", ""),
                  extractor_model=edge.get("extractor_model", ""))

def create_citation_edge(citing_doi: str, cited_doi: str):
    with driver.session() as s:
        s.run("""
            MATCH (a:Paper {doi: $citing}), (b:Paper {doi: $cited})
            MERGE (a)-[:CITES]->(b)
        """, citing=citing_doi, cited=cited_doi)

def create_consensus_edge(claim_a_id: str, claim_b_id: str, rel_type: str,
                          confidence: float, source_chunk_id: str):
    """rel_type must be SUPPORTS or CONTRADICTS."""
    assert rel_type in ("SUPPORTS", "CONTRADICTS"), "Invalid consensus edge type"
    with driver.session() as s:
        s.run(f"""
            MATCH (a:Claim {{id: $aid}}), (b:Claim {{id: $bid}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.confidence      = $confidence,
                r.source_chunk_id = $chunk_id,
                r.extractor_model = 'consensus_agent'
        """, aid=claim_a_id, bid=claim_b_id,
             confidence=confidence, chunk_id=source_chunk_id)
```

---

## 6. Phase 4 — Indexing Strategy

Run these after all nodes are ingested.

### 6.1 Vector Indexes

```cypher
-- Chunk text vectors (primary retrieval surface)
CREATE VECTOR INDEX chunk_text_vector IF NOT EXISTS
FOR (c:Chunk) ON (c.text_vector)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};

-- Equation description vectors
CREATE VECTOR INDEX equation_desc_vector IF NOT EXISTS
FOR (e:Equation) ON (e.desc_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};

-- Concept / Method embedding vectors (for entity-level search)
CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};

-- Figure/Table clip embeddings
CREATE VECTOR INDEX figure_clip_embedding IF NOT EXISTS
FOR (f:Figure) ON (f.clip_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};

-- Claim embeddings (for cross-paper similarity)
CREATE VECTOR INDEX claim_embedding IF NOT EXISTS
FOR (cl:Claim) ON (cl.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};
```

### 6.2 Property Indexes (Metadata Filters)

```cypher
CREATE INDEX paper_year        IF NOT EXISTS FOR (p:Paper)   ON (p.year);
CREATE INDEX paper_field       IF NOT EXISTS FOR (p:Paper)   ON (p.field_of_study);
CREATE INDEX journal_impact    IF NOT EXISTS FOR (j:Journal) ON (j.impact_factor);
CREATE INDEX section_type      IF NOT EXISTS FOR (s:Section) ON (s.section_type);
CREATE INDEX chunk_type        IF NOT EXISTS FOR (c:Chunk)   ON (c.chunk_type);
CREATE INDEX chunk_salience    IF NOT EXISTS FOR (c:Chunk)   ON (c.salience_score);
CREATE INDEX claim_type        IF NOT EXISTS FOR (cl:Claim)  ON (cl.claim_type);
CREATE INDEX result_dataset    IF NOT EXISTS FOR (r:Result)  ON (r.dataset);
CREATE INDEX method_type       IF NOT EXISTS FOR (m:Method)  ON (m.type);
CREATE INDEX author_h_index    IF NOT EXISTS FOR (a:Author)  ON (a.h_index);
```

### 6.3 Pre-compute PageRank

Run this after ingesting citations. Schedule to re-run weekly.

```python
# retrieval/pagerank.py
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def compute_and_store_pagerank():
    with driver.session() as s:
        # Project citation graph
        s.run("""
            CALL gds.graph.project(
              'citation_graph',
              'Paper',
              'CITES'
            )
        """)
        # Run PageRank and write scores back to Paper nodes
        s.run("""
            CALL gds.pageRank.write(
              'citation_graph',
              {
                maxIterations: 20,
                dampingFactor: 0.85,
                writeProperty: 'pagerank'
              }
            )
            YIELD nodePropertiesWritten
        """)
        # Drop projection after use
        s.run("CALL gds.graph.drop('citation_graph')")

if __name__ == "__main__":
    compute_and_store_pagerank()
    print("PageRank scores written to Paper nodes.")
```

---

## 7. Phase 5 — Hybrid Retrieval Engine

### 7.1 Query Engine

```python
# retrieval/query_engine.py
from neo4j import GraphDatabase
from ingest.embedder import embed_query
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

HYBRID_QUERY = """
// ── 1. Dense vector search on chunks ────────────────────────────────────────
CALL db.index.vector.queryNodes(
  'chunk_text_vector', $top_k, $query_vector
) YIELD node AS chunk, score AS cosine_score

// ── 2. Metadata filter (applied concurrently with vector search) ─────────────
WHERE ($field_filter IS NULL OR EXISTS {
  MATCH (chunk)<-[:CONTAINS]-(:Section)<-[:HAS_SECTION]-(p:Paper)
  WHERE p.field_of_study = $field_filter
})
AND ($year_min IS NULL OR EXISTS {
  MATCH (chunk)<-[:CONTAINS]-(:Section)<-[:HAS_SECTION]-(p:Paper)
  WHERE p.year >= $year_min
})

// ── 3. Traverse upward to get Paper metadata ─────────────────────────────────
MATCH (chunk)<-[:CONTAINS]-(sec:Section)<-[:HAS_SECTION]-(paper:Paper)
OPTIONAL MATCH (paper)-[:PUBLISHED_IN]->(journal:Journal)
OPTIONAL MATCH (paper)-[:AUTHORED_BY]->(author:Author)

// ── 4. Authority score from graph metrics ────────────────────────────────────
WITH chunk, cosine_score, sec, paper, journal,
     COALESCE(paper.pagerank, 0.0)    AS pr_score,
     COALESCE(journal.impact_factor, 0.0) AS impact,
     MAX(COALESCE(author.h_index, 0)) AS max_h_index

// ── 5. Hybrid score fusion ────────────────────────────────────────────────────
WITH chunk, sec, paper,
     (cosine_score * 0.6)
   + (pr_score    * 0.2)
   + (CASE WHEN impact > 5 THEN 0.1 ELSE impact / 50.0 END)
   + (CASE WHEN max_h_index > 20 THEN 0.1 ELSE max_h_index / 200.0 END)
   AS hybrid_score,
     cosine_score

// ── 6. Return top results with context ───────────────────────────────────────
ORDER BY hybrid_score DESC
LIMIT $final_k

// ── 7. Add neighboring chunks for context window ─────────────────────────────
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(chunk)
OPTIONAL MATCH (chunk)-[:NEXT]->(nxt:Chunk)

RETURN
  chunk.id           AS chunk_id,
  chunk.raw_text     AS text,
  sec.title          AS section_title,
  sec.section_type   AS section_type,
  paper.doi          AS doi,
  paper.title        AS paper_title,
  paper.year         AS year,
  hybrid_score,
  cosine_score,
  prev.raw_text      AS prev_chunk,
  nxt.raw_text       AS next_chunk
"""

def hybrid_search(
    query: str,
    top_k: int = 20,
    final_k: int = 5,
    field_filter: str = None,
    year_min: int = None,
) -> list[dict]:
    query_vector = embed_query(query)
    with driver.session() as s:
        results = s.run(
            HYBRID_QUERY,
            query_vector=query_vector,
            top_k=top_k,
            final_k=final_k,
            field_filter=field_filter,
            year_min=year_min,
        )
        return [dict(r) for r in results]
```

### 7.2 Multimodal Resolution

```python
# retrieval/query_engine.py (continued)

EQUATION_SEARCH_QUERY = """
CALL db.index.vector.queryNodes(
  'equation_desc_vector', $top_k, $query_vector
) YIELD node AS eq, score
WHERE score > $min_score
RETURN eq.latex AS latex, eq.plain_desc AS description,
       eq.domain AS domain, score
ORDER BY score DESC
LIMIT $final_k
"""

def search_equations(query: str, top_k: int = 5, min_score: float = 0.75) -> list:
    query_vector = embed_query(query)
    with driver.session() as s:
        results = s.run(
            EQUATION_SEARCH_QUERY,
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
            final_k=top_k,
        )
        return [dict(r) for r in results]
```

### 7.3 Answer Synthesis

```python
# retrieval/query_engine.py (continued)
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
synth_model = genai.GenerativeModel(os.getenv("REASON_MODEL", "gemini-2.5-pro"))

SYNTHESIS_PROMPT = """
You are a scientific research assistant. Answer the user's question using ONLY
the provided context chunks. Cite every claim with [Paper Title, Year].

If chunks from different papers contradict each other, explicitly note the contradiction.
If the evidence is insufficient, say so — do not speculate.

Context:
{context}

Question: {question}

Answer:"""

def synthesize_answer(question: str, chunks: list[dict]) -> dict:
    context_parts = []
    for i, c in enumerate(chunks):
        ctx = (
            f"[{i+1}] {c['paper_title']} ({c['year']}) — {c['section_type']}\n"
            f"{c['text']}"
        )
        if c.get("prev_chunk"):
            ctx = f"[context before]: {c['prev_chunk']}\n" + ctx
        if c.get("next_chunk"):
            ctx = ctx + f"\n[context after]: {c['next_chunk']}"
        context_parts.append(ctx)

    context_str = "\n\n---\n\n".join(context_parts)
    prompt = SYNTHESIS_PROMPT.format(context=context_str, question=question)

    response = synth_model.generate_content(
        prompt,
        generation_config={"temperature": 0.3},
    )

    return {
        "answer": response.text,
        "sources": [
            {"doi": c["doi"], "title": c["paper_title"], "year": c["year"],
             "section": c["section_title"], "score": c["hybrid_score"]}
            for c in chunks
        ],
    }
```

---

## 8. Phase 6 — LangGraph Agentic Pipeline

### 8.1 Full Ingestion Graph (State Machine)

```python
# agents/pipeline.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
import operator

class IngestState(TypedDict):
    paper_path: str
    paper_id: str
    parsed_doc: Optional[dict]
    chunks: List[dict]
    extracted_entities: List[dict]
    low_confidence_chunks: List[str]    # chunk IDs needing re-extraction
    retry_count: int
    canonicalization_done: bool
    errors: Annotated[List[str], operator.add]

def build_ingest_graph():
    g = StateGraph(IngestState)

    g.add_node("parse",            parse_node)
    g.add_node("chunk_and_embed",  chunk_embed_node)
    g.add_node("extract_entities", extract_node)
    g.add_node("self_correct",     self_correct_node)
    g.add_node("canonicalize",     canonicalize_node)
    g.add_node("write_graph",      write_graph_node)

    g.set_entry_point("parse")
    g.add_edge("parse", "chunk_and_embed")
    g.add_edge("chunk_and_embed", "extract_entities")

    # Conditional: re-extract low-confidence chunks, up to MAX_RETRY_CYCLES
    g.add_conditional_edges(
        "extract_entities",
        should_self_correct,
        {"retry": "self_correct", "proceed": "canonicalize"},
    )
    g.add_edge("self_correct", "extract_entities")  # Loop back
    g.add_edge("canonicalize", "write_graph")
    g.add_edge("write_graph", END)

    return g.compile()

MAX_RETRY = int(os.getenv("MAX_RETRY_CYCLES", "2"))
MIN_CONF  = float(os.getenv("MIN_CONFIDENCE", "0.70"))

def should_self_correct(state: IngestState) -> str:
    has_low_conf = len(state["low_confidence_chunks"]) > 0
    under_limit  = state["retry_count"] < MAX_RETRY
    if has_low_conf and under_limit:
        return "retry"
    return "proceed"
```

### 8.2 Canonicalization Agent

```python
# agents/canonicalizer.py
import json
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
reason_model = genai.GenerativeModel(os.getenv("REASON_MODEL", "gemini-2.5-pro"))

CANON_PROMPT = """
You are checking whether two scientific terms refer to the same concept.
Be CONSERVATIVE — only merge if you are highly certain (>0.90 confidence).
A false merge corrupts the knowledge graph permanently.

Term A: "{term_a}" with aliases: {aliases_a}
Term B: "{term_b}" with aliases: {aliases_b}

Output JSON only:
{{
  "should_merge": true|false,
  "confidence": 0.0,
  "reason": "..."
}}
"""

def should_merge_concepts(concept_a: dict, concept_b: dict) -> dict:
    prompt = CANON_PROMPT.format(
        term_a=concept_a["name"],
        aliases_a=concept_a.get("aliases", []),
        term_b=concept_b["name"],
        aliases_b=concept_b.get("aliases", []),
    )
    response = reason_model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    return json.loads(response.text)

def merge_concepts_in_graph(keep_id: str, absorb_id: str, session):
    """
    Redirects all relationships from the absorbed node to the kept node,
    then deletes the absorbed node. Uses APOC for safe relationship migration.
    """
    session.run("""
        MATCH (keep:Concept {id: $keep_id})
        MATCH (absorb:Concept {id: $absorb_id})
        // Redirect all incoming relationships
        CALL apoc.refactor.mergeNodes([keep, absorb], {
            properties: 'combine',
            mergeRels: true
        }) YIELD node
        SET node.aliases = node.aliases + absorb.aliases
    """, keep_id=keep_id, absorb_id=absorb_id)
```

### 8.3 Consensus Agent (SUPPORTS / CONTRADICTS)

```python
# agents/validator.py
import json
import os
import google.generativeai as genai
from neo4j import GraphDatabase

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
reason_model = genai.GenerativeModel(os.getenv("REASON_MODEL", "gemini-2.5-pro"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

CONSENSUS_PROMPT = """
Two claims from different scientific papers are given below.
Determine if claim B supports, contradicts, or is neutral to claim A.
Only mark as SUPPORTS or CONTRADICTS if the relationship is explicit and unambiguous.
When in doubt, return NEUTRAL.

Claim A (from paper {paper_a}): "{claim_a}"
Claim B (from paper {paper_b}): "{claim_b}"

Output JSON only:
{{
  "relationship": "SUPPORTS|CONTRADICTS|NEUTRAL",
  "confidence": 0.0,
  "reason": "..."
}}
"""

def evaluate_claim_pair(claim_a: dict, claim_b: dict) -> dict:
    prompt = CONSENSUS_PROMPT.format(
        paper_a=claim_a.get("paper_title", "unknown"),
        claim_a=claim_a["text"],
        paper_b=claim_b.get("paper_title", "unknown"),
        claim_b=claim_b["text"],
    )
    response = reason_model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    return json.loads(response.text)

def find_and_link_claim_consensus(new_claim_id: str, similarity_threshold: float = 0.85):
    """
    For a newly ingested claim, finds semantically similar existing claims
    and evaluates SUPPORTS/CONTRADICTS relationships.
    """
    with driver.session() as s:
        # Find similar claims via vector search
        similar = s.run("""
            MATCH (new:Claim {id: $cid})
            CALL db.index.vector.queryNodes(
              'claim_embedding', 10, new.embedding
            ) YIELD node AS other, score
            WHERE other.id <> $cid AND score > $threshold
            MATCH (other)<-[:MENTIONS]-(:Chunk)<-[:CONTAINS]-(:Section)<-[:HAS_SECTION]-(p:Paper)
            MATCH (new)<-[:MENTIONS]-(:Chunk)<-[:CONTAINS]-(:Section)<-[:HAS_SECTION]-(pn:Paper)
            WHERE p.doi <> pn.doi  // Cross-paper only
            RETURN other.id AS other_id, other.text AS other_text,
                   p.title AS paper_title, score
        """, cid=new_claim_id, threshold=similarity_threshold)

        new_claim = s.run("MATCH (c:Claim {id: $id}) RETURN c.text AS text",
                          id=new_claim_id).single()

        for row in similar:
            result = evaluate_claim_pair(
                {"text": new_claim["text"], "paper_title": "current"},
                {"text": row["other_text"], "paper_title": row["paper_title"]},
            )
            if result["relationship"] in ("SUPPORTS", "CONTRADICTS") \
               and result["confidence"] >= 0.75:
                from graph.layer3 import create_consensus_edge
                create_consensus_edge(
                    new_claim_id, row["other_id"],
                    result["relationship"],
                    result["confidence"],
                    new_claim_id,
                )
```

---

## 9. Phase 7 — React Frontend

### 9.1 FastAPI Backend

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import ingest, query

app = FastAPI(title="STEM Knowledge Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(query.router,  prefix="/query",  tags=["retrieval"])
```

```python
# api/routers/query.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from retrieval.query_engine import hybrid_search, synthesize_answer, search_equations

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 20
    final_k: int = 5
    field_filter: Optional[str] = None
    year_min: Optional[int] = None
    include_equations: bool = False

@router.post("/")
async def query_knowledge_graph(req: QueryRequest):
    chunks  = hybrid_search(req.question, req.top_k, req.final_k,
                            req.field_filter, req.year_min)
    answer  = synthesize_answer(req.question, chunks)
    result  = {"answer": answer["answer"], "sources": answer["sources"]}

    if req.include_equations:
        result["equations"] = search_equations(req.question)

    return result
```

### 9.2 React Frontend

```bash
# Bootstrap the frontend
npm create vite@latest frontend -- --template react
cd frontend
npm install axios @neo4j-devtools/react-resizable-panels lucide-react
```

```jsx
// frontend/src/App.jsx
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [question, setQuestion]   = useState("");
  const [fieldFilter, setField]   = useState("");
  const [yearMin, setYearMin]     = useState("");
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  async function handleSearch() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post(`${API}/query/`, {
        question,
        field_filter: fieldFilter || null,
        year_min:     yearMin     ? parseInt(yearMin) : null,
        include_equations: true,
      });
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>STEM Knowledge Graph</h1>

      {/* Query Input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSearch()}
          placeholder="Ask a scientific question..."
          style={{ flex: 1, padding: "10px 14px", fontSize: 15, borderRadius: 8,
                   border: "1px solid #ccc" }}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          style={{ padding: "10px 20px", borderRadius: 8, background: "#2563eb",
                   color: "#fff", border: "none", cursor: "pointer", fontWeight: 600 }}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {/* Filters */}
      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        <input
          value={fieldFilter}
          onChange={e => setField(e.target.value)}
          placeholder="Field of study (optional)"
          style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd", flex: 1 }}
        />
        <input
          value={yearMin}
          onChange={e => setYearMin(e.target.value)}
          placeholder="Published after year"
          type="number"
          style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #ddd", width: 160 }}
        />
      </div>

      {error && (
        <div style={{ padding: 12, background: "#fef2f2", borderRadius: 8,
                      color: "#b91c1c", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {result && (
        <div>
          {/* Answer */}
          <div style={{ padding: 20, background: "#f0f9ff", borderRadius: 12,
                        marginBottom: 20, lineHeight: 1.7 }}>
            <h3 style={{ marginTop: 0 }}>Answer</h3>
            <p style={{ whiteSpace: "pre-wrap" }}>{result.answer}</p>
          </div>

          {/* Sources */}
          <h3>Sources ({result.sources.length})</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {result.sources.map((src, i) => (
              <div key={i} style={{ padding: "12px 16px", border: "1px solid #e5e7eb",
                                    borderRadius: 10, background: "#fff" }}>
                <div style={{ fontWeight: 600 }}>{src.title} ({src.year})</div>
                <div style={{ color: "#6b7280", fontSize: 13, marginTop: 2 }}>
                  {src.section} · DOI: {src.doi}
                  &nbsp;·&nbsp;
                  <span style={{ color: "#2563eb" }}>
                    Score: {src.score?.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Equations (if any) */}
          {result.equations?.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <h3>Related Equations</h3>
              {result.equations.map((eq, i) => (
                <div key={i} style={{ padding: 14, background: "#fafafa",
                                      borderRadius: 8, marginBottom: 8,
                                      fontFamily: "monospace" }}>
                  <div style={{ fontSize: 13, color: "#374151" }}>{eq.latex}</div>
                  <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4,
                                fontFamily: "sans-serif" }}>
                    {eq.description}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

### 9.3 Running the Full Stack

```bash
# Terminal 1 — Start Neo4j
docker start neo4j-stem

# Terminal 2 — Start FastAPI
cd stem-kg-rag
uvicorn api.main:app --reload --port 8000

# Terminal 3 — Start React frontend
cd frontend
npm run dev
# → http://localhost:5173
```

---

## 10. Testing & Validation Strategy

### 10.1 Unit Tests

```python
# tests/test_chunker.py
from ingest.chunker import chunk_section, classify_chunk

def test_chunk_does_not_exceed_token_limit():
    section = {"id": "s1", "title": "Methods", "chunks": [
        {"raw_text": "word " * 600}
    ]}
    chunks = chunk_section(section, max_tokens=512)
    for c in chunks:
        assert len(c["raw_text"].split()) <= 700  # generous buffer

def test_chunk_types_are_classified():
    assert classify_chunk("We define attention as a weighted sum") == "definition"
    assert classify_chunk("The model achieves 93% accuracy on CIFAR") == "result_report"

# tests/test_retrieval.py
from unittest.mock import patch, MagicMock
from retrieval.query_engine import hybrid_search

def test_hybrid_search_returns_results():
    with patch("retrieval.query_engine.embed_query", return_value=[0.1] * 768):
        with patch("retrieval.query_engine.driver") as mock_driver:
            mock_session = MagicMock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = []
            results = hybrid_search("attention mechanism", top_k=5, final_k=3)
            assert isinstance(results, list)
```

### 10.2 Graph Integrity Checks

```cypher
-- Chunks without NEXT edges (except last chunks in sections) — should be low
MATCH (c:Chunk)
WHERE NOT (c)-[:NEXT]->() AND NOT ()<-[:NEXT]-(c)
  AND c.order > 0
RETURN COUNT(c) AS orphaned_non_first_chunks;

-- Claims without GROUNDED_IN edges — a data quality violation
MATCH (cl:Claim)
WHERE NOT (cl)-[:GROUNDED_IN]->(:Chunk)
RETURN COUNT(cl) AS ungrounded_claims;

-- Entities without any MENTIONS edges — floating entities
MATCH (n)
WHERE (n:Concept OR n:Method OR n:Claim)
  AND NOT ()-[:MENTIONS]->(n)
RETURN LABELS(n)[0] AS type, COUNT(n) AS floating_count;
```

### 10.3 Retrieval Quality Metrics

```python
# tests/eval_retrieval.py
"""
Evaluate retrieval quality using a small labelled test set.
golden_set: list of {question, expected_dois: [str, ...]}
"""

from retrieval.query_engine import hybrid_search

def recall_at_k(golden_set: list, k: int = 5) -> float:
    hits = 0
    for item in golden_set:
        results = hybrid_search(item["question"], final_k=k)
        returned_dois = {r["doi"] for r in results}
        if any(doi in returned_dois for doi in item["expected_dois"]):
            hits += 1
    return hits / len(golden_set)
```

---

## 11. Deployment Architecture

### 11.1 Recommended Production Setup

```
┌─────────────────────────────────────────────────────────┐
│  Ingestion pipeline (batch / async)                      │
│  Celery worker + Redis queue                             │
│  Triggered by: file upload API endpoint                  │
└────────────────────────┬────────────────────────────────┘
                         │ writes
                         ▼
┌──────────────────────────────┐     ┌──────────────────────┐
│  Neo4j AuraDB (managed)      │◄────│  GDS + APOC plugins   │
│  Vector indexes live here    │     │  PageRank batch job    │
└──────────────┬───────────────┘     └──────────────────────┘
               │ reads
               ▼
┌─────────────────────────────┐
│  FastAPI (2+ replicas)       │
│  Gunicorn + Uvicorn workers  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  React (Vite build → Nginx) │
│  Or deploy to Vercel        │
└─────────────────────────────┘
```

### 11.2 Docker Compose (Development)

```yaml
# docker-compose.yml
version: "3.9"
services:
  neo4j:
    image: neo4j:5.20.0-enterprise
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/your_password
      NEO4J_PLUGINS: '["graph-data-science", "apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: gds.*,apoc.*
    volumes:
      - neo4j_data:/data

  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [neo4j]
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports: ["5173:80"]
    depends_on: [api]

volumes:
  neo4j_data:
```

---

## 12. Risk Mitigations & Known Pitfalls

| Risk | Mitigation |
|---|---|
| Canonicalization false merges | Use confidence ≥ 0.90 threshold; log all merges; run integrity check after each batch |
| Unbounded self-correction cost | Hard cap `MAX_RETRY_CYCLES=2`; only retry chunks with confidence < 0.70 |
| Neo4j GDS plugin missing | Always start container with `NEO4J_PLUGINS='["graph-data-science"]'`; check on startup |
| NEXT edge gaps after parallel ingest | Use sequential chunk ordering; add integrity check query post-ingest |
| Gemini API rate limits | Use `tenacity` exponential backoff; batch embed up to 100 texts per call |
| SUPPORTS/CONTRADICTS false positives | Use conservative prompt ("when in doubt return NEUTRAL"); set confidence ≥ 0.75 |
| Vector index dimension mismatch | Set `EMBED_DIM` once in `.env` and never change after first ingest; add startup assertion |
| Memory pressure on large PDFs | Stream chunks in batches of 50; never hold all embeddings in RAM simultaneously |

### Build Order Recommendation

```
Week 1–2:  Phase 1 (spine) + Phase 4 (indexes) + Phase 5 basic vector search
Week 3–4:  Phase 2 (entity extraction) + Phase 5 full hybrid query
Week 5–6:  Phase 6 (LangGraph agents) — extractor → canonicalizer → validator
Week 7:    Phase 3 (SUPPORTS/CONTRADICTS consensus edges)
Week 8:    Phase 7 (React frontend) + Phase 10 (evaluation)
```

---

*Document version 1.0 — covers Neo4j 5.20, Gemini 2.x, LangGraph 0.2, React 18*