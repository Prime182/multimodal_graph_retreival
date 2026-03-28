"""Search service for the browser frontend and API endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunking import chunk_article
from .config import Phase1Settings
from .corpus import get_corpus_client
from .domain_config import preload_domain_knowledge
from .edges import Layer3CorpusRecord, build_layer3
from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .extraction import extract_layer2
from .ingestion_status import IngestionStatusStore, get_ingestion_status_store
from .models import ChunkRecord, PaperRecord, SectionRecord
from .parser import parse_article
from .retrieval import LocalVectorIndex
from .tracing import get_tracing_manager


def _xml_paths(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).glob("*.xml"))


def _load_corpus(
    input_dir: str | Path,
    settings: Phase1Settings | None = None,
    use_gemini: bool = False,
    ingestion_store: IngestionStatusStore | None = None,
) -> tuple[list[PaperRecord], list[Layer2DocumentRecord], Layer3CorpusRecord]:
    settings = settings or Phase1Settings.from_env()
    ingestion_store = ingestion_store or get_ingestion_status_store()
    papers: list[PaperRecord] = []
    layer2_docs: list[Layer2DocumentRecord] = []
    for path in _xml_paths(input_dir):
        fallback_paper_id = path.stem
        fallback_title = path.name
        source_path = str(path)
        ingestion_store.upsert_status(
            paper_id=fallback_paper_id,
            paper_title=fallback_title,
            source_path=source_path,
            status="parsing",
        )
        try:
            parsed = parse_article(path)
            fallback_paper_id = parsed.paper_id
            fallback_title = parsed.title
            ingestion_store.upsert_status(
                paper_id=parsed.paper_id,
                paper_title=parsed.title,
                source_path=source_path,
                status="chunking",
            )
            paper = chunk_article(parsed, settings=settings)
            ingestion_store.upsert_status(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source_path=source_path,
                status="extracting",
                chunk_count=len(paper.chunks),
            )
            layer2_doc = extract_layer2(paper, settings=settings, use_gemini=use_gemini)
            avg_confidence = (
                round(sum(entity.confidence for entity in layer2_doc.entities) / len(layer2_doc.entities), 3)
                if layer2_doc.entities
                else 0.0
            )
            chunk_coverage = (
                sum(1 for entity_ids in layer2_doc.chunk_entity_ids.values() if entity_ids) / max(len(paper.chunks), 1)
            )
            avg_salience = (
                sum(layer2_doc.chunk_salience_scores.values()) / max(len(layer2_doc.chunk_salience_scores), 1)
                if layer2_doc.chunk_salience_scores
                else 0.0
            )
            extraction_quality = round((0.45 * avg_confidence) + (0.35 * avg_salience) + (0.20 * chunk_coverage), 3)
            ingestion_store.upsert_status(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source_path=source_path,
                status="storing",
                extraction_quality=extraction_quality,
                avg_confidence=avg_confidence,
                entity_count=len(layer2_doc.entities),
                chunk_count=len(paper.chunks),
            )
            papers.append(paper)
            layer2_docs.append(layer2_doc)
            ingestion_store.upsert_status(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source_path=source_path,
                status="complete",
                extraction_quality=extraction_quality,
                avg_confidence=avg_confidence,
                entity_count=len(layer2_doc.entities),
                chunk_count=len(paper.chunks),
            )
        except Exception as exc:
            existing = ingestion_store.get_status(fallback_paper_id)
            retry_count = int(existing["retry_count"]) + 1 if existing else 1
            ingestion_store.upsert_status(
                paper_id=fallback_paper_id,
                paper_title=fallback_title,
                source_path=source_path,
                status="failed",
                error=str(exc),
                retry_count=retry_count,
            )
            continue
    layer3 = build_layer3(papers, layer2_docs)
    return papers, layer2_docs, layer3


def _paper_summary(paper: PaperRecord) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "doi": paper.doi,
        "published_year": paper.published_year,
        "journal": paper.journal.to_dict() if paper.journal else None,
        "authors": [author.full_name for author in paper.authors],
        "abstract": paper.abstract,
        "reference_count": len(paper.references),
        "table_count": len(paper.tables),
        "figure_count": len(paper.figures),
        "section_count": len(paper.sections),
        "chunk_count": len(paper.chunks),
    }


def _entity_summary(
    entity: Layer2EntityRecord,
    paper: PaperRecord,
    section_title: str | None = None,
    chunk_text: str | None = None,
) -> dict[str, Any]:
    payload = {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "label": entity.label,
        "confidence": entity.confidence,
        "source_chunk_id": entity.source_chunk_id,
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "section_title": section_title,
        "properties": entity.properties,
    }
    if chunk_text:
        payload["chunk_text"] = chunk_text
    return payload


def _chunk_context(chunks: list[ChunkRecord], target_index: int) -> dict[str, str | None]:
    if target_index < 0 or target_index >= len(chunks):
        return {
            "previous": None,
            "current": None,
            "next": None,
        }
    previous_text = chunks[target_index - 1].text if target_index > 0 else None
    current_text = chunks[target_index].text if 0 <= target_index < len(chunks) else None
    next_text = chunks[target_index + 1].text if target_index + 1 < len(chunks) else None
    return {
        "previous": previous_text,
        "current": current_text,
        "next": next_text,
    }


def _citation_summary(edge: dict[str, Any], papers_by_id: dict[str, PaperRecord]) -> dict[str, Any]:
    source_paper = papers_by_id.get(edge["source_node_id"])
    target_paper = papers_by_id.get(edge["target_node_id"])
    return {
        "relation_type": edge["relation_type"],
        "confidence": edge["confidence"],
        "evidence": edge["evidence"],
        "source_paper_id": edge["source_node_id"],
        "source_paper_title": source_paper.title if source_paper else edge["source_label"],
        "target_paper_id": edge["target_node_id"],
        "target_paper_title": target_paper.title if target_paper else edge["target_label"],
        "metadata": edge.get("metadata", {}),
    }


def build_search_bundle(
    query: str,
    index: LocalVectorIndex,
    papers: list[PaperRecord],
    layer2_docs: list[Layer2DocumentRecord],
    layer3: Layer3CorpusRecord,
    top_k: int = 5,
) -> dict[str, Any]:
    if not papers:
        return {
            "query": query,
            "text_hits": [],
            "table_hits": [],
            "figure_hits": [],
            "papers": [],
            "entities": {},
            "citations": [],
            "stats": {
                "paper_count": 0,
                "chunk_count": 0,
                "table_count": 0,
                "figure_count": 0,
                "entity_count": 0,
                "citation_count": 0,
                "matched_papers": 0,
            },
        }
    
    if index is None:
        raise RuntimeError("Search index not initialized")
    
    papers_by_id = {paper.paper_id: paper for paper in papers}
    text_hits = index.search(query=query, top_k=top_k) or []
    table_hits = index.search_tables(query=query, top_k=top_k) or []
    figure_hits = index.search_figures(query=query, top_k=top_k) or []
    paper_hits = index.search_papers(query=query, top_k=top_k) or []
    paper_scores: dict[str, float] = {}

    chunk_lookup: dict[tuple[str, str], tuple[PaperRecord, ChunkRecord, int]] = {}
    section_lookup: dict[tuple[str, str], SectionRecord] = {}
    for paper in papers:
        for section in paper.sections:
            section_lookup[(paper.paper_id, section.section_id)] = section
        for position, chunk in enumerate(paper.chunks):
            chunk_lookup[(paper.paper_id, chunk.chunk_id)] = (paper, chunk, position)

    entities_by_chunk: dict[str, list[Layer2EntityRecord]] = {}
    for doc in layer2_docs:
        for entity in doc.entities:
            entities_by_chunk.setdefault(entity.source_chunk_id, []).append(entity)

    chunk_results: list[dict[str, Any]] = []
    matched_paper_ids: set[str] = set()
    for hit in text_hits:
        paper = papers_by_id.get(hit.paper_id)
        if paper is None:
            continue
        chunk_info = chunk_lookup.get((hit.paper_id, hit.chunk_id))
        section = section_lookup.get((hit.paper_id, hit.section_id))
        entities = entities_by_chunk.get(hit.chunk_id, [])
        matched_paper_ids.add(hit.paper_id)
        paper_scores[hit.paper_id] = max(paper_scores.get(hit.paper_id, 0.0), hit.score)
        chunk_results.append(
            {
                "score": hit.score,
                "paper_id": hit.paper_id,
                "paper_title": hit.paper_title,
                "doi": hit.doi,
                "section_id": hit.section_id,
                "section_title": hit.section_title,
                "section_type": section.section_type if section else None,
                "chunk_id": hit.chunk_id,
                "chunk_type": chunk_info[1].chunk_type if chunk_info else None,
                "text": hit.text,
                "context": _chunk_context(paper.chunks, chunk_info[2]) if chunk_info else {},
                "entities": [
                    _entity_summary(
                        entity,
                        paper=paper,
                        section_title=section.title if section else None,
                        chunk_text=hit.text,
                    )
                    for entity in entities
                ],
            }
        )

    table_results = [
        {
            **table_hit.to_dict(),
            "paper_title": table_hit.paper_title,
            "type": "table",
        }
        for table_hit in table_hits
    ]
    for table_hit in table_hits:
        matched_paper_ids.add(table_hit.paper_id)
        paper_scores[table_hit.paper_id] = max(paper_scores.get(table_hit.paper_id, 0.0), table_hit.score)

    figure_results = [
        {
            **figure_hit.to_dict(),
            "type": "figure",
        }
        for figure_hit in figure_hits
    ]
    for figure_hit in figure_hits:
        matched_paper_ids.add(figure_hit.paper_id)
        paper_scores[figure_hit.paper_id] = max(paper_scores.get(figure_hit.paper_id, 0.0), figure_hit.score)

    for score, paper in paper_hits:
        paper_scores[paper.paper_id] = max(paper_scores.get(paper.paper_id, 0.0), score)

    for _, paper in paper_hits:
        matched_paper_ids.add(paper.paper_id)

    paper_results = [
        {
            "score": round(paper_scores.get(paper.paper_id, 0.0), 4),
            **_paper_summary(paper),
        }
        for paper in sorted(papers, key=lambda item: paper_scores.get(item.paper_id, 0.0), reverse=True)
        if paper.paper_id in matched_paper_ids
    ]

    matched_entities: dict[str, list[dict[str, Any]]] = {}
    for doc in layer2_docs:
        if doc.paper_id not in matched_paper_ids:
            continue
        paper = papers_by_id.get(doc.paper_id)
        if paper is None:
            continue
        chunk_title_lookup = {chunk.chunk_id: chunk.text for chunk in paper.chunks}
        section_title_lookup = {section.section_id: section.title for section in paper.sections}
        for entity in doc.entities:
            section_title = None
            for chunk in paper.chunks:
                if chunk.chunk_id == entity.source_chunk_id:
                    section_title = section_title_lookup.get(chunk.section_id)
                    break
            summary = _entity_summary(
                entity,
                paper=paper,
                section_title=section_title,
                chunk_text=chunk_title_lookup.get(entity.source_chunk_id),
            )
            matched_entities.setdefault(entity.entity_type, []).append(summary)

    for entity_list in matched_entities.values():
        entity_list.sort(key=lambda item: item["confidence"], reverse=True)

    citation_results = [
        _citation_summary(edge.to_dict(), papers_by_id)
        for edge in layer3.citation_edges
        if edge.source_node_id in matched_paper_ids or edge.target_node_id in matched_paper_ids
    ]

    stats = {
        "paper_count": len(papers),
        "chunk_count": sum(len(paper.chunks) for paper in papers),
        "table_count": sum(len(paper.tables) for paper in papers),
        "figure_count": sum(len(paper.figures) for paper in papers),
        "entity_count": sum(len(doc.entities) for doc in layer2_docs),
        "citation_count": len(layer3.citation_edges),
        "matched_papers": len(matched_paper_ids),
    }

    return {
        "query": query,
        "text_hits": chunk_results,
        "table_hits": table_results,
        "figure_hits": figure_results,
        "papers": paper_results,
        "entities": matched_entities,
        "citations": citation_results,
        "stats": stats,
    }


@dataclass(slots=True)
class GraphRAGSearchService:
    """In-memory search service for the local XML corpus."""

    input_dir: str | Path = "articles"
    settings: Phase1Settings | None = None
    use_gemini: bool = False
    papers: list[PaperRecord] = field(default_factory=list)
    layer2_docs: list[Layer2DocumentRecord] = field(default_factory=list)
    layer3: Layer3CorpusRecord = field(default_factory=Layer3CorpusRecord)
    index: LocalVectorIndex | None = None
    ingestion_store: IngestionStatusStore | None = None

    def __post_init__(self) -> None:
        preload_domain_knowledge()
        settings = self.settings or Phase1Settings.from_env()
        self.ingestion_store = get_ingestion_status_store()
        self.papers, self.layer2_docs, self.layer3 = _load_corpus(
            self.input_dir,
            settings=settings,
            use_gemini=self.use_gemini,
            ingestion_store=self.ingestion_store,
        )
        self.index = LocalVectorIndex(self.papers)

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        tracer = get_tracing_manager()
        
        if self.index is None:
            raise RuntimeError("Search index is not initialized")
        
        with tracer.trace(
            name="document_search",
            input_data={"query": query, "top_k": top_k},
        ) as trace_ctx:
            result = build_search_bundle(
                query=query,
                index=self.index,
                papers=self.papers,
                layer2_docs=self.layer2_docs,
                layer3=self.layer3,
                top_k=top_k,
            )
            
            # Log retrieval metrics
            tracer.log_retrieval(
                query=query,
                results=result.get("text_hits", [])[:3],
                result_count=len(result.get("text_hits", [])),
                metadata={
                    "table_hits": len(result.get("table_hits", [])),
                    "figure_hits": len(result.get("figure_hits", [])),
                    "paper_hits": len(result.get("papers", [])),
                    "entity_hits": len(result.get("entities", [])),
                },
            )
            
            trace_ctx["output"] = {
                "text_hits": len(result.get("text_hits", [])),
                "figure_hits": len(result.get("figure_hits", [])),
                "papers": len(result.get("papers", [])),
                "entities": len(result.get("entities", [])),
            }
            
            return result

    def corpus_misses(self, limit: int = 100) -> list[dict[str, object]]:
        return get_corpus_client().list_misses(limit=limit)

    def extraction_quality_report(self, limit: int = 100) -> dict[str, Any]:
        store = self.ingestion_store or get_ingestion_status_store()
        return store.extraction_quality_report(limit=limit)

    def ingestion_status_report(self, limit: int = 100) -> list[dict[str, Any]]:
        store = self.ingestion_store or get_ingestion_status_store()
        return store.list_statuses(limit=limit)
