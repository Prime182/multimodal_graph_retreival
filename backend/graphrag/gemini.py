"""Shared Gemini client helpers with retry and SDK compatibility."""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .circuit_breaker import CircuitBreakerOpenError, get_circuit_breaker

try:  # Prefer the newer SDK when installed.
    import google.genai as modern_genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    modern_genai = None

if modern_genai is None:
    try:
        import google.generativeai as legacy_genai  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        legacy_genai = None
else:  # pragma: no cover - the deprecated SDK is only needed as a fallback
    legacy_genai = None


class GeminiError(RuntimeError):
    """Raised when a Gemini request cannot be completed."""


_JSON_BLOCK_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_model_prefix(model_name: str) -> str:
    return model_name.removeprefix("models/")


def _clean_json_text(text: str) -> str:
    return _JSON_BLOCK_RE.sub("", text.strip()).strip()


def gemini_available() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY")) and (modern_genai is not None or legacy_genai is not None)


@lru_cache(maxsize=1)
def _client_backend() -> tuple[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiError("GOOGLE_API_KEY is not configured.")
    if modern_genai is not None:
        return "modern", modern_genai.Client(api_key=api_key)
    if legacy_genai is not None:
        legacy_genai.configure(api_key=api_key)
        return "legacy", legacy_genai
    raise GeminiError("Gemini SDK is not installed.")


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    if isinstance(response, dict) and "text" in response:
        return str(response["text"])
    raise GeminiError("Gemini response did not contain text.")


def _coerce_embedding_values(payload: Any) -> list[float]:
    if payload is None:
        raise GeminiError("Gemini embedding payload was empty.")
    if isinstance(payload, list):
        if payload and isinstance(payload[0], (int, float)):
            return [float(value) for value in payload]
        if payload:
            return _coerce_embedding_values(payload[0])
    if isinstance(payload, dict):
        for key in ("values", "embedding", "embeddings"):
            if key in payload:
                return _coerce_embedding_values(payload[key])
    values = getattr(payload, "values", None)
    if values is not None:
        return [float(value) for value in values]
    embedding = getattr(payload, "embedding", None)
    if embedding is not None:
        return _coerce_embedding_values(embedding)
    embeddings = getattr(payload, "embeddings", None)
    if embeddings is not None:
        return _coerce_embedding_values(embeddings)
    raise GeminiError("Gemini embedding payload could not be parsed.")


def _normalize_vector(values: list[float]) -> list[float]:
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]


def _resize_embedding(values: list[float], dimensions: int | None) -> list[float]:
    if not dimensions or dimensions <= 0 or len(values) == dimensions:
        return _normalize_vector(values)
    if len(values) < dimensions:
        padded = values + [0.0] * (dimensions - len(values))
        return _normalize_vector(padded)

    buckets = [0.0] * dimensions
    counts = [0] * dimensions
    source_dim = len(values)
    for index, value in enumerate(values):
        bucket = min((index * dimensions) // source_dim, dimensions - 1)
        buckets[bucket] += value
        counts[bucket] += 1
    reduced = [
        buckets[index] / counts[index] if counts[index] else 0.0
        for index in range(dimensions)
    ]
    return _normalize_vector(reduced)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1),
    retry=retry_if_exception_type((GeminiError, json.JSONDecodeError, ValueError)),
    reraise=True,
)
def generate_text(
    prompt: str,
    model_name: str,
    *,
    temperature: float = 0.1,
    response_mime_type: str | None = None,
) -> str:
    backend, client = _client_backend()
    model = _strip_model_prefix(model_name)
    config: dict[str, Any] = {"temperature": temperature}
    if response_mime_type:
        config["response_mime_type"] = response_mime_type
    breaker = get_circuit_breaker()
    try:
        breaker.guard("gemini_text")
    except CircuitBreakerOpenError as exc:
        raise GeminiError(str(exc)) from exc

    try:
        if backend == "modern":
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        else:
            model_client = client.GenerativeModel(model_name)
            response = model_client.generate_content(
                prompt,
                generation_config=config,
            )
        breaker.record_success("gemini_text")
        return _extract_text(response).strip()
    except GeminiError:
        breaker.record_failure("gemini_text")
        raise
    except Exception as exc:
        breaker.record_failure("gemini_text")
        raise GeminiError(str(exc)) from exc


def generate_json(
    prompt: str,
    model_name: str,
    *,
    temperature: float = 0.1,
) -> Any:
    text = generate_text(
        prompt,
        model_name,
        temperature=temperature,
        response_mime_type="application/json",
    )
    return json.loads(_clean_json_text(text))


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1),
    retry=retry_if_exception_type(GeminiError),
    reraise=True,
)
def embed_text(
    text: str,
    *,
    model_name: str = "text-embedding-004",
    dimensions: int | None = None,
    task_type: str = "SEMANTIC_SIMILARITY",
) -> list[float]:
    backend, client = _client_backend()
    model = _strip_model_prefix(model_name)
    breaker = get_circuit_breaker()
    try:
        breaker.guard("gemini_embedding")
    except CircuitBreakerOpenError as exc:
        raise GeminiError(str(exc)) from exc

    try:
        if backend == "modern":
            config: dict[str, Any] = {"task_type": task_type}
            if dimensions:
                config["output_dimensionality"] = dimensions
            response = client.models.embed_content(
                model=model,
                contents=text,
                config=config,
            )
        else:
            response = client.embed_content(
                model=model,
                content=text,
                task_type=task_type,
            )
        values = _coerce_embedding_values(response)
        breaker.record_success("gemini_embedding")
        return _resize_embedding(values, dimensions)
    except GeminiError:
        breaker.record_failure("gemini_embedding")
        raise
    except Exception as exc:
        breaker.record_failure("gemini_embedding")
        raise GeminiError(str(exc)) from exc
