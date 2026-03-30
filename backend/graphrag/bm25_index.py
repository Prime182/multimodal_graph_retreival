"""Lightweight BM25 ranking over the chunk corpus."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
import re
from typing import Any, Iterable, Sequence

from .models import ChunkRecord, PaperRecord, SectionRecord

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
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
class BM25Hit:
    score: float
    paper_id: str
    paper_title: str
    section_id: str
    chunk_id: str
    section_title: str
    text: str
    doi: str | None = None
    chunk_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section_id": self.section_id,
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "text": self.text,
            "doi": self.doi,
            "chunk_type": self.chunk_type,
        }


@dataclass(slots=True)
class _BM25Document:
    paper_id: str
    paper_title: str
    section_id: str
    section_title: str
    chunk_id: str
    text: str
    doi: str | None = None
    chunk_type: str | None = None


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _section_title_lookup(paper: PaperRecord) -> dict[str, str]:
    return {section.section_id: section.title for section in paper.sections}


def _chunk_documents_from_papers(papers: Sequence[PaperRecord]) -> list[_BM25Document]:
    documents: list[_BM25Document] = []
    for paper in papers:
        section_titles = _section_title_lookup(paper)
        for chunk in paper.chunks:
            documents.append(
                _BM25Document(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    section_id=chunk.section_id,
                    section_title=section_titles.get(chunk.section_id, chunk.section_id),
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    doi=paper.doi,
                    chunk_type=chunk.chunk_type,
                )
            )
    return documents


class BM25Index:
    """BM25 ranking over chunk texts with a pure-Python fallback."""

    def __init__(self, documents: Sequence[_BM25Document]) -> None:
        self.documents = list(documents)
        self._tokens = [_tokenize(doc.text) for doc in self.documents]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        self._idf = self._build_idf()
        self._bm25 = BM25Okapi(self._tokens) if BM25Okapi is not None and self._tokens else None

    @classmethod
    def from_papers(cls, papers: Sequence[PaperRecord]) -> "BM25Index":
        return cls(_chunk_documents_from_papers(papers))

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[ChunkRecord],
        *,
        paper_lookup: dict[str, PaperRecord] | None = None,
        section_lookup: dict[str, SectionRecord] | None = None,
    ) -> "BM25Index":
        documents: list[_BM25Document] = []
        for chunk in chunks:
            paper = paper_lookup.get(chunk.paper_id) if paper_lookup else None
            section = section_lookup.get(chunk.section_id) if section_lookup else None
            documents.append(
                _BM25Document(
                    paper_id=chunk.paper_id,
                    paper_title=paper.title if paper else chunk.paper_id,
                    section_id=chunk.section_id,
                    section_title=section.title if section else chunk.section_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    doi=paper.doi if paper else None,
                    chunk_type=chunk.chunk_type,
                )
            )
        return cls(documents)

    @staticmethod
    def _build_idf_documents(tokens: Sequence[list[str]]) -> dict[str, float]:
        df: dict[str, int] = {}
        for doc_tokens in tokens:
            for term in set(doc_tokens):
                df[term] = df.get(term, 0) + 1
        total_docs = max(len(tokens), 1)
        return {
            term: log(1.0 + ((total_docs - freq + 0.5) / (freq + 0.5)))
            for term, freq in df.items()
        }

    def _build_idf(self) -> dict[str, float]:
        return self._build_idf_documents(self._tokens)

    def _score_document(self, query_tokens: list[str], doc_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        doc_tokens = self._tokens[doc_index]
        if not query_tokens or not doc_tokens:
            return 0.0

        freqs: dict[str, int] = {}
        for token in doc_tokens:
            freqs[token] = freqs.get(token, 0) + 1

        doc_len = self._doc_lengths[doc_index] or 1
        norm = k1 * (1.0 - b + b * (doc_len / self._avgdl if self._avgdl else 1.0))

        score = 0.0
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if not tf:
                continue
            idf = self._idf.get(token)
            if idf is None:
                continue
            score += idf * (tf * (k1 + 1.0)) / (tf + norm)
        return score

    def search(self, query: str, top_k: int = 5, section_type: str | None = None) -> list[BM25Hit]:
        query_tokens = _tokenize(query)
        scored: list[tuple[float, int]] = []
        library_scores = None
        if self._bm25 is not None and query_tokens:
            library_scores = list(self._bm25.get_scores(query_tokens))
            # `rank_bm25` can flatten tiny corpora to all-zero scores when every
            # matching term appears in exactly half of the documents. Fall back
            # to the local scorer so ranking still reflects lexical matches.
            if library_scores and max(abs(float(score)) for score in library_scores) == 0.0:
                library_scores = None

        for index, document in enumerate(self.documents):
            if section_type and document.chunk_type != section_type:
                continue
            score = (
                float(library_scores[index])
                if library_scores is not None
                else self._score_document(query_tokens, index)
            )
            scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[BM25Hit] = []
        for score, index in scored[:top_k]:
            document = self.documents[index]
            hits.append(
                BM25Hit(
                    score=round(score, 4),
                    paper_id=document.paper_id,
                    paper_title=document.paper_title,
                    section_id=document.section_id,
                    chunk_id=document.chunk_id,
                    section_title=document.section_title,
                    text=document.text,
                    doi=document.doi,
                    chunk_type=document.chunk_type,
                )
            )
        return hits
