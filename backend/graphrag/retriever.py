"""Hybrid retrieval orchestration for local vector, BM25, and graph hits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .bm25_index import BM25Index
from .context_builder import ContextBuilder, GenerationContext


class _SearchBackend(Protocol):
    def search(self, query: str, top_k: int = 5, section_type: str | None = None) -> list[Any]: ...


@dataclass(slots=True)
class RetrievedPassage:
    rank: int
    score: float
    chunk_id: str
    paper_id: str
    paper_title: str
    section_id: str
    section_title: str
    text: str
    doi: str | None = None
    retrieval_method: str = "fusion"
    graph_context: Any | None = None
    previous_text: str | None = None
    next_text: str | None = None
    source_scores: dict[str, float] = field(default_factory=dict)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _hit_key(hit: Any) -> str:
    chunk_id = _get_field(hit, "chunk_id")
    if chunk_id:
        return str(chunk_id)
    paper_id = _get_field(hit, "paper_id", "")
    section_id = _get_field(hit, "section_id", "")
    text = _get_field(hit, "text", "")
    return f"{paper_id}::{section_id}::{text[:40]}"


def _hit_to_passage(hit: Any, *, retrieval_method: str) -> RetrievedPassage:
    return RetrievedPassage(
        rank=int(_get_field(hit, "rank", 0) or 0),
        score=float(_get_field(hit, "score", 0.0) or 0.0),
        chunk_id=str(_get_field(hit, "chunk_id", "")),
        paper_id=str(_get_field(hit, "paper_id", "")),
        paper_title=str(_get_field(hit, "paper_title", "")),
        section_id=str(_get_field(hit, "section_id", "")),
        section_title=str(_get_field(hit, "section_title", "")),
        text=str(_get_field(hit, "text", "")),
        doi=_get_field(hit, "doi"),
        retrieval_method=retrieval_method,
        graph_context=_get_field(hit, "graph_context"),
        previous_text=_get_field(hit, "previous_text"),
        next_text=_get_field(hit, "next_text"),
    )


class HybridRetriever:
    """Fuses vector, lexical, and optional graph-backed retrieval with RRF."""

    def __init__(
        self,
        vector_index: _SearchBackend,
        bm25_index: BM25Index,
        *,
        graph_backend: Any | None = None,
        context_builder: ContextBuilder | None = None,
        rrf_k: int = 60,
        source_multiplier: int = 3,
    ) -> None:
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.graph_backend = graph_backend
        self.context_builder = context_builder or ContextBuilder()
        self.rrf_k = rrf_k
        self.source_multiplier = max(1, source_multiplier)

    def _query_embedding(self, query: str) -> list[float]:
        embedder = getattr(self.vector_index, "embedder", None)
        if embedder is None or not hasattr(embedder, "embed"):
            if self.graph_backend is not None and hasattr(self.graph_backend, "embedder"):
                embedder = getattr(self.graph_backend, "embedder")
        if embedder is None or not hasattr(embedder, "embed"):
            raise RuntimeError("A query embedder is required for graph-backed retrieval")
        return list(embedder.embed(query))

    def _vector_hits(self, query: str, top_k: int, section_type: str | None = None) -> list[Any]:
        search = getattr(self.vector_index, "search", None)
        if not callable(search):
            return []
        try:
            return list(search(query, top_k=top_k, section_type=section_type))
        except TypeError:
            return list(search(query, top_k=top_k))

    def _bm25_hits(self, query: str, top_k: int, section_type: str | None = None) -> list[Any]:
        return list(self.bm25_index.search(query, top_k=top_k, section_type=section_type))

    def _graph_hits(self, query: str, top_k: int) -> list[Any]:
        if self.graph_backend is None:
            return []

        search_chunks = getattr(self.graph_backend, "search_chunks", None)
        if not callable(search_chunks):
            return []

        embedding = self._query_embedding(query)
        try:
            return list(search_chunks(embedding, top_k=top_k))
        except TypeError:
            return list(search_chunks(query, top_k=top_k))

    def retrieve(self, query: str, top_k: int = 10, section_type: str | None = None) -> list[RetrievedPassage]:
        candidate_limit = max(top_k * self.source_multiplier, top_k)
        source_hits: list[tuple[str, list[Any]]] = [
            ("graph", self._graph_hits(query, candidate_limit)),
            ("vector", self._vector_hits(query, candidate_limit, section_type=section_type)),
            ("bm25", self._bm25_hits(query, candidate_limit, section_type=section_type)),
        ]

        fused: dict[str, dict[str, Any]] = {}
        for source_name, hits in source_hits:
            for rank, hit in enumerate(hits[:candidate_limit], start=1):
                if section_type and _get_field(hit, "chunk_type") not in (None, section_type):
                    # The existing vector index already filters by section type when it knows how.
                    # This guard just keeps the merged result consistent across backends.
                    continue
                key = _hit_key(hit)
                payload = fused.setdefault(
                    key,
                    {
                        "passage": _hit_to_passage(hit, retrieval_method="fusion"),
                        "rrf_score": 0.0,
                        "source_scores": {},
                    },
                )
                payload["rrf_score"] += 1.0 / (self.rrf_k + rank)
                payload["source_scores"][source_name] = float(_get_field(hit, "score", 0.0) or 0.0)
                if source_name == "graph" and payload["passage"].graph_context is None:
                    payload["passage"].graph_context = _get_field(hit, "graph_context")

        ranked_payloads = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                -len(item["source_scores"]),
                -max(item["source_scores"].values()) if item["source_scores"] else 0.0,
                item["passage"].paper_title,
                item["passage"].section_title,
                item["passage"].chunk_id,
            ),
        )

        passages: list[RetrievedPassage] = []
        for rank, payload in enumerate(ranked_payloads[:top_k], start=1):
            passage = payload["passage"]
            passage.rank = rank
            passage.score = round(payload["rrf_score"], 6)
            passage.source_scores = payload["source_scores"]
            passages.append(passage)
        return passages

    def build_context(self, query: str, top_k: int = 10, max_tokens: int = 4000) -> GenerationContext:
        passages = self.retrieve(query, top_k=top_k)
        return self.context_builder.build(query, passages, max_tokens=max_tokens)
