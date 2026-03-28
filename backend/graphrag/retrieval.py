"""Local vector retrieval for Phase 1 chunk search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .chunking import chunk_article
from .config import Phase1Settings
from .embeddings import HashingEmbedder, cosine_similarity
from .models import ChunkRecord, FigureRecord, PaperRecord, SearchHit, TableRecord
from .parser import parse_article


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(slots=True)
class _IndexedChunk:
    paper: PaperRecord
    chunk: ChunkRecord
    section_title: str


@dataclass(slots=True)
class _IndexedTable:
    paper: PaperRecord
    table: TableRecord


@dataclass(slots=True)
class _IndexedFigure:
    paper: PaperRecord
    figure: FigureRecord


@dataclass(slots=True)
class TableHit:
    score: float
    paper_id: str
    paper_title: str
    table_id: str
    label: str | None
    caption: str | None
    text: str
    doi: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "table_id": self.table_id,
            "label": self.label,
            "caption": self.caption,
            "text": self.text,
            "doi": self.doi,
        }


@dataclass(slots=True)
class FigureHit:
    score: float
    paper_id: str
    paper_title: str
    figure_id: str
    label: str | None
    caption: str | None
    alt_text: str | None
    text: str
    placeholder_uri: str | None
    source_ref: str | None
    doi: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "figure_id": self.figure_id,
            "label": self.label,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "text": self.text,
            "placeholder_uri": self.placeholder_uri,
            "source_ref": self.source_ref,
            "doi": self.doi,
        }


class LocalVectorIndex:
    """A deterministic local vector index for the Layer 1 corpus."""

    def __init__(self, documents: list[PaperRecord], embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.documents = documents
        self._chunks: list[_IndexedChunk] = []
        self._tables: list[_IndexedTable] = []
        self._figures: list[_IndexedFigure] = []
        self._build()

    @classmethod
    def from_xml_paths(
        cls,
        paths: list[str | Path],
        settings: Phase1Settings | None = None,
        embedder: HashingEmbedder | None = None,
    ) -> "LocalVectorIndex":
        settings = settings or Phase1Settings.from_env()
        documents: list[PaperRecord] = []
        for path in paths:
            article = parse_article(path)
            documents.append(chunk_article(article, settings=settings))
        return cls(documents=documents, embedder=embedder)

    def _build(self) -> None:
        section_lookup: dict[tuple[str, str], str] = {}
        for paper in self.documents:
            paper.abstract_embedding = self.embedder.embed(paper.abstract or paper.title)
            for section in paper.sections:
                section_lookup[(paper.paper_id, section.section_id)] = section.title
            for chunk in paper.chunks:
                chunk.embedding = self.embedder.embed(chunk.text)
                self._chunks.append(
                    _IndexedChunk(
                        paper=paper,
                        chunk=chunk,
                        section_title=section_lookup.get((paper.paper_id, chunk.section_id), chunk.section_id),
                    )
                )
            for table in paper.tables:
                table.embedding = self.embedder.embed(f"{table.caption or ''} {table.text}".strip())
                self._tables.append(_IndexedTable(paper=paper, table=table))
            for figure in paper.figures:
                figure.embedding = self.embedder.embed(f"{figure.caption or ''} {figure.alt_text or ''}".strip() or figure.text)
                self._figures.append(_IndexedFigure(paper=paper, figure=figure))

    @staticmethod
    def _lexical_overlap(query: str, text: str) -> float:
        query_terms = {
            token.lower()
            for token in _TOKEN_RE.findall(query)
            if token.lower() not in _STOPWORDS and len(token) > 2
        }
        if not query_terms:
            return 0.0
        text_terms = {token.lower() for token in _TOKEN_RE.findall(text)}
        matches = len(query_terms & text_terms)
        return matches / len(query_terms)

    @staticmethod
    def _normalized_label(text: str) -> str:
        return " ".join(_TOKEN_RE.findall(text.lower()))

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_type: str | None = None,
    ) -> list[SearchHit]:
        query_embedding = self.embedder.embed(query)
        scored: list[tuple[float, _IndexedChunk]] = []
        for indexed in self._chunks:
            if section_type and indexed.chunk.chunk_type != section_type:
                continue
            similarity = cosine_similarity(query_embedding, indexed.chunk.embedding)
            overlap = self._lexical_overlap(query, indexed.chunk.text)
            score = (0.65 * similarity) + (0.25 * overlap) + (0.10 * indexed.chunk.salience_score)
            scored.append((score, indexed))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[SearchHit] = []
        for score, indexed in scored[:top_k]:
            hits.append(
                SearchHit(
                    score=round(score, 4),
                    paper_id=indexed.paper.paper_id,
                    paper_title=indexed.paper.title,
                    section_id=indexed.chunk.section_id,
                    chunk_id=indexed.chunk.chunk_id,
                    section_title=indexed.section_title,
                    text=indexed.chunk.text,
                    doi=indexed.paper.doi,
                )
            )
        return hits

    def search_papers(self, query: str, top_k: int = 5) -> list[tuple[float, PaperRecord]]:
        query_embedding = self.embedder.embed(query)
        scored = [
            (cosine_similarity(query_embedding, paper.abstract_embedding), paper)
            for paper in self.documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(round(score, 4), paper) for score, paper in scored[:top_k]]

    def search_tables(self, query: str, top_k: int = 5) -> list[TableHit]:
        query_embedding = self.embedder.embed(query)
        scored: list[tuple[float, _IndexedTable]] = []
        for indexed in self._tables:
            text = f"{indexed.table.caption or ''} {indexed.table.text}".strip()
            similarity = cosine_similarity(query_embedding, indexed.table.embedding)
            overlap = self._lexical_overlap(query, text)
            label_overlap = self._lexical_overlap(query, indexed.table.label or "")
            query_label = self._normalized_label(query)
            table_label = self._normalized_label(indexed.table.label or "")
            exact_label_bonus = 0.25 if query_label and table_label and query_label == table_label else 0.0
            score = (
                (0.45 * similarity)
                + (0.20 * overlap)
                + (0.15 * label_overlap)
                + exact_label_bonus
                + (0.10 * min(len(text.split()) / 120.0, 1.0))
            )
            scored.append((score, indexed))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            TableHit(
                score=round(score, 4),
                paper_id=indexed.paper.paper_id,
                paper_title=indexed.paper.title,
                table_id=indexed.table.table_id,
                label=indexed.table.label,
                caption=indexed.table.caption,
                text=indexed.table.text,
                doi=indexed.paper.doi,
            )
            for score, indexed in scored[:top_k]
        ]

    def search_figures(self, query: str, top_k: int = 5) -> list[FigureHit]:
        query_embedding = self.embedder.embed(query)
        scored: list[tuple[float, _IndexedFigure]] = []
        for indexed in self._figures:
            text = indexed.figure.text or indexed.figure.caption or indexed.figure.alt_text or indexed.figure.label or ""
            similarity = cosine_similarity(query_embedding, indexed.figure.embedding)
            overlap = self._lexical_overlap(query, text)
            label_overlap = self._lexical_overlap(query, indexed.figure.label or "")
            query_label = self._normalized_label(query)
            figure_label = self._normalized_label(indexed.figure.label or "")
            exact_label_bonus = 0.25 if query_label and figure_label and query_label == figure_label else 0.0
            score = (0.48 * similarity) + (0.22 * overlap) + (0.20 * label_overlap) + exact_label_bonus
            scored.append((score, indexed))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            FigureHit(
                score=round(score, 4),
                paper_id=indexed.paper.paper_id,
                paper_title=indexed.paper.title,
                figure_id=indexed.figure.figure_id,
                label=indexed.figure.label,
                caption=indexed.figure.caption,
                alt_text=indexed.figure.alt_text,
                text=indexed.figure.text,
                placeholder_uri=indexed.figure.placeholder_uri,
                source_ref=indexed.figure.source_ref,
                doi=indexed.paper.doi,
            )
            for score, indexed in scored[:top_k]
        ]
