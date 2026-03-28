"""Data models for the Phase 1 document spine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ReferenceRecord:
    reference_id: str
    label: str | None = None
    title: str | None = None
    doi: str | None = None
    url: str | None = None
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuthorRecord:
    author_id: str
    full_name: str
    given_name: str | None = None
    surname: str | None = None
    email: str | None = None
    affiliations: list[str] = field(default_factory=list)
    orcid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JournalRecord:
    journal_id: str
    code: str | None = None
    name: str | None = None
    issn: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SectionRecord:
    section_id: str
    paper_id: str
    title: str
    section_type: str
    level: int
    ordinal: int
    text: str
    paragraphs: list[str] = field(default_factory=list)
    key_sentence: str | None = None
    parent_section_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TableRecord:
    table_id: str
    paper_id: str
    ordinal: int
    label: str | None = None
    caption: str | None = None
    text: str = ""
    rows: int | None = None
    columns: int | None = None
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FigureRecord:
    figure_id: str
    paper_id: str
    ordinal: int
    label: str | None = None
    caption: str | None = None
    alt_text: str | None = None
    text: str = ""
    placeholder_uri: str | None = None
    source_ref: str | None = None
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    section_id: str
    ordinal: int
    text: str
    chunk_type: str
    word_count: int
    token_count: int
    salience_score: float
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    source_path: str
    title: str
    doi: str | None = None
    pii: str | None = None
    article_number: str | None = None
    published_year: int | None = None
    abstract: str = ""
    highlights: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    journal: JournalRecord | None = None
    authors: list[AuthorRecord] = field(default_factory=list)
    references: list[ReferenceRecord] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    figures: list[FigureRecord] = field(default_factory=list)
    sections: list[SectionRecord] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    abstract_embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    score: float
    paper_id: str
    paper_title: str
    section_id: str
    chunk_id: str
    section_title: str
    text: str
    doi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
