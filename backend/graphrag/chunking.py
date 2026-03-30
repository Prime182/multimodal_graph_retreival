"""Chunking utilities for Section -> Chunk conversion."""

from __future__ import annotations

import re

from .config import Phase1Settings
from .models import ChunkRecord, PaperRecord, SectionRecord


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+")


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _word_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _salience(text: str) -> float:
    words = max(_word_count(text), 1)
    citations = text.count("[")
    numerics = sum(token.isdigit() for token in _TOKEN_RE.findall(text))
    uppercase_tokens = sum(token.isupper() and len(token) > 1 for token in _TOKEN_RE.findall(text))
    score = 0.35
    score += min(citations, 5) * 0.05
    score += min(numerics / words, 0.2)
    score += min(uppercase_tokens / words, 0.15)
    return round(min(score, 1.0), 3)


def _paragraph_segments(paragraph: str, max_words: int) -> list[str]:
    if _word_count(paragraph) <= max_words:
        return [paragraph]
    segments: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in _split_sentences(paragraph):
        sentence_words = _word_count(sentence)
        if current and current_words + sentence_words > max_words:
            segments.append(" ".join(current).strip())
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words
    if current:
        segments.append(" ".join(current).strip())
    return [segment for segment in segments if segment]


def _tail_words(text: str, overlap_words: int) -> str:
    if overlap_words <= 0:
        return ""
    words = _TOKEN_RE.findall(text)
    if not words:
        return ""
    return " ".join(words[-overlap_words:])


def _section_chunks(section: SectionRecord, max_words: int, overlap_words: int) -> list[str]:
    paragraphs = section.paragraphs or ([section.text] if section.text else [])
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    overlap_words = max(0, min(overlap_words, max(max_words - 1, 0)))
    for paragraph in paragraphs:
        for segment in _paragraph_segments(paragraph, max_words=max_words):
            segment_words = _word_count(segment)
            if buffer and buffer_words + segment_words > max_words:
                chunk_text = " ".join(buffer).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                carry = _tail_words(chunk_text, overlap_words)
                buffer = [carry] if carry else []
                buffer_words = _word_count(carry)
                buffer.append(segment)
                buffer_words += segment_words
            else:
                buffer.append(segment)
                buffer_words += segment_words
    if buffer:
        chunks.append(" ".join(buffer).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_article(article: PaperRecord, settings: Phase1Settings | None = None) -> PaperRecord:
    """Populate article chunks in place and return the same article for chaining."""

    settings = settings or Phase1Settings.from_env()
    chunks: list[ChunkRecord] = []
    for section in article.sections:
        if not section.text:
            continue
        section_chunks = _section_chunks(
            section,
            max_words=settings.chunk_size_words,
            overlap_words=settings.chunk_overlap_words,
        )
        for ordinal, chunk_text in enumerate(section_chunks):
            chunk_id = f"{section.section_id}-chunk-{ordinal}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    paper_id=article.paper_id,
                    section_id=section.section_id,
                    ordinal=ordinal,
                    text=chunk_text,
                    chunk_type=section.section_type,
                    word_count=_word_count(chunk_text),
                    token_count=_word_count(chunk_text),
                    salience_score=_salience(chunk_text),
                )
            )

    chunks_by_section: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(chunk.section_id, []).append(chunk)
    for section_chunks in chunks_by_section.values():
        for index, chunk in enumerate(section_chunks):
            chunk.prev_chunk_id = section_chunks[index - 1].chunk_id if index > 0 else None
            chunk.next_chunk_id = section_chunks[index + 1].chunk_id if index + 1 < len(section_chunks) else None

    article.chunks = chunks
    return article
