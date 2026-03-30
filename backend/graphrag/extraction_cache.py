"""SQLite-backed cache for chunk-level extraction results."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any


class ExtractionCache:
    """Caches extraction payloads by chunk content and model version."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self.db_path = Path(
            db_path
            or os.getenv("EXTRACTION_CACHE_DB_PATH")
            or ".cache/graphrag_extraction_cache.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_cache (
                chunk_id TEXT NOT NULL,
                chunk_text_hash TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (chunk_id, chunk_text_hash, model_version)
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def chunk_text_hash(chunk_text: str) -> str:
        return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_payload(payload: Any) -> str:
        return json.dumps(payload)

    @staticmethod
    def _deserialize_payload(payload_json: str) -> Any:
        return json.loads(payload_json)

    def get(self, chunk_id: str, chunk_text: str, model_version: str) -> Any | None:
        """Return a cached payload for the exact chunk/model combination."""
        chunk_hash = self.chunk_text_hash(chunk_text)
        row = self._conn.execute(
            """
            SELECT payload_json
            FROM extraction_cache
            WHERE chunk_id = ? AND chunk_text_hash = ? AND model_version = ?
            """,
            (chunk_id, chunk_hash, model_version),
        ).fetchone()
        if row is None:
            return None

        now = time.time()
        self._conn.execute(
            """
            UPDATE extraction_cache
            SET last_accessed = ?, hit_count = hit_count + 1
            WHERE chunk_id = ? AND chunk_text_hash = ? AND model_version = ?
            """,
            (now, chunk_id, chunk_hash, model_version),
        )
        self._conn.commit()
        return self._deserialize_payload(row[0])

    def set(self, chunk_id: str, chunk_text: str, model_version: str, payload: Any) -> None:
        """Store a payload for an exact chunk/model combination."""
        chunk_hash = self.chunk_text_hash(chunk_text)
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO extraction_cache (
                chunk_id, chunk_text_hash, model_version, payload_json,
                created_at, last_accessed, hit_count
            )
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(chunk_id, chunk_text_hash, model_version) DO UPDATE SET
                payload_json = excluded.payload_json,
                last_accessed = excluded.last_accessed
            """,
            (
                chunk_id,
                chunk_hash,
                model_version,
                self._serialize_payload(payload),
                now,
                now,
            ),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM extraction_cache")
        self._conn.commit()


_default_cache: ExtractionCache | None = None


def get_extraction_cache() -> ExtractionCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ExtractionCache()
    return _default_cache
