"""Embedding utilities for retrieval and entity similarity."""

from __future__ import annotations

from hashlib import blake2b
import math
import re
from typing import Protocol

from .gemini import embed_text, gemini_available


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")


class TextEmbedder(Protocol):
    """Minimal embedder interface used across the pipeline."""

    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for the input text."""


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic local fallback embedder with cosine-normalized output."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 12) / 12.0
            vector[index] += sign * weight
        return _normalize_vector(vector)


class GeminiEmbedder:
    """Gemini-backed embedder with deterministic local fallback."""

    def __init__(
        self,
        dim: int = 256,
        model_name: str | None = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        fallback: TextEmbedder | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dim = dim
        self.model_name = model_name or "text-embedding-004"
        self.task_type = task_type
        self.fallback = fallback or HashingEmbedder(dim=dim)
        self.enabled = gemini_available()

    def embed(self, text: str) -> list[float]:
        normalized_text = text.strip()
        if not normalized_text:
            return [0.0] * self.dim
        if not self.enabled:
            return self.fallback.embed(normalized_text)
        try:
            return embed_text(
                normalized_text,
                model_name=self.model_name,
                dimensions=self.dim,
                task_type=self.task_type,
            )
        except Exception:
            # Avoid repeated retry costs when the remote embedding service is unavailable.
            self.enabled = False
            return self.fallback.embed(normalized_text)


def build_entity_embedder(dim: int = 256, *, prefer_remote: bool = True) -> TextEmbedder:
    """Prefer Gemini embeddings for semantic graph quality, else fall back locally."""

    if not prefer_remote:
        return HashingEmbedder(dim=dim)
    return GeminiEmbedder(dim=dim)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
