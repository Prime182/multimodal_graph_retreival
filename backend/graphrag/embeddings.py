"""Embedding utilities for retrieval and entity similarity."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import blake2b
import math
import os
from pathlib import Path
import re
from typing import Any, Protocol

from .gemini import embed_text, gemini_available


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")
_SENTENCE_TRANSFORMER_CACHE: dict[str, Any] = {}


def _cached_sentence_transformer_path(model_name: str) -> Path | None:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_dir / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


class TextEmbedder(Protocol):
    """Minimal embedder interface used across the pipeline."""

    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for the input text."""


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _resize_vector(vector: list[float], dim: int) -> list[float]:
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    if len(vector) == dim:
        return _normalize_vector(vector)
    if len(vector) > dim:
        return _normalize_vector(vector[:dim])
    return _normalize_vector(vector + [0.0] * (dim - len(vector)))


class HashingEmbedder:
    """Deterministic embedder kept for explicit tests and offline diagnostics."""

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
    """Gemini-backed embedder with local SentenceTransformer fallback."""

    def __init__(
        self,
        dim: int = 256,
        model_name: str | None = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        fallback: TextEmbedder | None = None,
        fallback_factory: Callable[[], TextEmbedder] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dim = dim
        self.model_name = model_name or "text-embedding-004"
        self.task_type = task_type
        self._fallback = fallback
        self._fallback_factory = fallback_factory or (lambda: SentenceTransformerEmbedder(dim=dim))
        self.enabled = gemini_available()

    def _get_fallback(self) -> TextEmbedder:
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def embed(self, text: str) -> list[float]:
        normalized_text = text.strip()
        if not normalized_text:
            return [0.0] * self.dim
        if not self.enabled:
            return self._get_fallback().embed(normalized_text)
        try:
            return _resize_vector(
                embed_text(
                    normalized_text,
                    model_name=self.model_name,
                    dimensions=self.dim,
                    task_type=self.task_type,
                ),
                self.dim,
            )
        except Exception:
            # Avoid repeated retry costs when the remote embedding service is unavailable.
            self.enabled = False
            return self._get_fallback().embed(normalized_text)


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder used as the non-API fallback."""

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, dim: int = 256, model_name: str | None = None) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dim = dim
        self.model_name = model_name or self.MODEL_NAME
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        cached = _SENTENCE_TRANSFORMER_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import path depends on environment
            raise RuntimeError(
                "sentence-transformers is required for the local embedding fallback."
            ) from exc

        allow_model_download = os.getenv("ALLOW_SBERT_MODEL_DOWNLOAD", "0") == "1"
        cached_model_path = _cached_sentence_transformer_path(self.model_name)
        try:
            if cached_model_path is not None:
                self._model = SentenceTransformer(str(cached_model_path), local_files_only=True)
            else:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            if not allow_model_download:
                raise RuntimeError(
                    f"Local SBERT model '{self.model_name}' is not cached. "
                    "Download it once or set ALLOW_SBERT_MODEL_DOWNLOAD=1."
                )
            self._model = SentenceTransformer(self.model_name)
        _SENTENCE_TRANSFORMER_CACHE[self.model_name] = self._model
        return self._model

    def embed(self, text: str) -> list[float]:
        normalized_text = text.strip()
        if not normalized_text:
            return [0.0] * self.dim
        model = self._load_model()
        vector = model.encode(normalized_text, normalize_embeddings=True)
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        return _resize_vector([float(value) for value in values], self.dim)


def build_entity_embedder(dim: int = 256, *, prefer_remote: bool = True) -> TextEmbedder:
    """Prefer Gemini embeddings for semantic graph quality, else use local SBERT."""

    if prefer_remote and gemini_available():
        return GeminiEmbedder(dim=dim)
    return SentenceTransformerEmbedder(dim=dim)


def probe_embedding_backends(dim: int = 256, *, prefer_remote: bool = True) -> dict[str, Any]:
    """Validate that the configured embedding path and local fallback are loadable."""

    status: dict[str, Any] = {
        "requested_dim": dim,
        "remote_available": False,
        "local_available": False,
        "active_backend": None,
    }

    if prefer_remote and gemini_available():
        try:
            remote = GeminiEmbedder(dim=dim)
            remote.embed("embedding health check")
            status["remote_available"] = True
            status["active_backend"] = "gemini"
        except Exception as exc:
            status["remote_error"] = str(exc)

    try:
        local = SentenceTransformerEmbedder(dim=dim)
        local.embed("embedding health check")
        status["local_available"] = True
        if status["active_backend"] is None:
            status["active_backend"] = "sentence-transformers"
    except Exception as exc:
        status["local_error"] = str(exc)

    if not status["local_available"]:
        raise RuntimeError("Local sentence-transformers fallback is not loadable.")
    if not status["remote_available"] and not status["local_available"]:
        raise RuntimeError("No embedding backend is loadable.")
    return status


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
