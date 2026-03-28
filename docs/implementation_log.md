# Phase 1 Implementation Log

## Delivered

- Created a new `backend/graphrag` package for the Phase 1 document spine.
- Implemented a tolerant Elsevier XML parser using `lxml` recovery mode because the source XML uses publisher-prefixed tags without usable namespace declarations.
- Parsed Layer 1 entities: `Paper`, `Author`, `Journal`, `Section`, and `Chunk`.
- Added section typing heuristics for `abstract`, `highlights`, `introduction`, `methods`, `results`, `discussion`, `conclusion`, and related structural sections.
- Added chunk generation with per-section `NEXT` links and heuristic salience scores.
- Added deterministic local embeddings and chunk/paper search so retrieval works without Gemini or Neo4j.
- Added optional Neo4j persistence with Layer 1 constraints and vector index creation.
- Added a CLI for building a JSON spine artifact, running local search, and loading data into Neo4j.
- Added unit tests for parsing, chunk sequencing, and retrieval relevance.

## Phase 2 Delivered

- Added Layer 2 entity records for `Concept`, `Method`, `Claim`, `Result`, and `Equation`.
- Added a deterministic Layer 2 extractor that derives concepts, methods, claims, results, and equations from the parsed chunk text.
- Added optional Gemini-backed extraction hooks behind `USE_GEMINI_EXTRACTION` and `EXTRACT_MODEL`, with a local heuristic fallback when Gemini is unavailable.
- Enriched chunk salience during Layer 2 extraction so retrieval ranking can use entity density and confidence.
- Added Neo4j Layer 2 ingestion for `MENTIONS`, `GROUNDED_IN`, `MEASURED_ON`, and `USING_METRIC` relationships.
- Added a `build-layer2` CLI command and extended `load-neo4j` to ingest Layer 2 entities after Layer 1.
- Added tests covering Layer 2 extraction on the existing article corpus.

## Phase 3 Delivered

- Added bibliographic reference parsing to the paper model so citations can be resolved from the XML bibliography.
- Added Layer 3 edge records for `CITES`, `IS_A`, `SUPPORTS`, and `CONTRADICTS`.
- Added citation resolution against the local corpus using DOI and title matching.
- Added conservative claim-to-claim semantic relation inference using embedding similarity plus lexical cues.
- Added a `build-layer3` CLI command and extended `load-neo4j` to write Layer 3 edges.

## Phase 4 Delivered

- Added a dedicated index manager for Neo4j property indexes.
- Added a PageRank batch job over the citation graph using GDS.
- Added CLI commands for building property indexes and computing PageRank.

## Frontend Delivered

- Added table extraction from Elsevier XML so tables can be searched as first-class artifacts.
- Added a browser search service that returns passage hits, table hits, paper-level matches, structured entities, and resolved citations in one payload.
- Added a lightweight single-page frontend with a search bar, result cards, entity panels, and citation panels.
- Added a `serve` CLI command to launch the browser UI against the local XML corpus.

## Assumptions

- The current corpus is Elsevier XML, so Phase 1 was implemented against that format instead of the PDF-oriented examples in `plan.md`.
- `Journal` metadata in the XML is sparse. The implementation uses the journal code from `<jid>` as the stable journal identity when ISSN and full journal name are absent.
- ORCID values are not present in the sampled articles, so `author_id` is used as the operational merge key while still leaving `orcid` available for later enrichment.
- The local hashing embedder is a Phase 1 fallback. It is deterministic and testable, but it is not intended to replace Gemini embeddings in later phases.

## Usage

```bash
python3 -m backend.graphrag.cli build-spine --input-dir articles --output /tmp/phase1_spine.json
python3 -m backend.graphrag.cli build-layer2 --input-dir articles --output /tmp/phase2_entities.json
python3 -m backend.graphrag.cli build-layer3 --input-dir articles --output /tmp/phase3_edges.json
python3 -m backend.graphrag.cli build-indexes
python3 -m backend.graphrag.cli compute-pagerank
python3 -m backend.graphrag.cli serve --input-dir articles --port 8000
python3 -m backend.graphrag.cli search "district heating neighborhood decision model" --input-dir articles
python3 -m backend.graphrag.cli load-neo4j --input-dir articles
```

## Remaining Gaps

- No FastAPI surface yet.
- No frontend yet.
