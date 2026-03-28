"""Runtime settings for the Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional fallback
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(slots=True)
class Phase1Settings:
    """Environment-backed settings for chunking and graph persistence."""

    chunk_size_words: int = 180
    chunk_overlap_words: int = 30
    embedding_dim: int = 256
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    # LangFuse tracing settings
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @classmethod
    def from_env(cls) -> "Phase1Settings":
        langfuse_host = os.getenv(
            "LANGFUSE_HOST",
            os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        return cls(
            chunk_size_words=_get_int("CHUNK_SIZE_WORDS", 180),
            chunk_overlap_words=_get_int("CHUNK_OVERLAP_WORDS", 30),
            embedding_dim=_get_int("EMBED_DIM", 256),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            langfuse_enabled=(
                os.getenv("LANGFUSE_PUBLIC_KEY") is not None
                and os.getenv("LANGFUSE_SECRET_KEY") is not None
            ),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=langfuse_host,
        )
