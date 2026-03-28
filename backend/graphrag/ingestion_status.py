"""SQLite-backed ingestion status tracking for corpus loading."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
from typing import Any


class IngestionStatusStore:
    """Persists per-paper ingestion status and extraction quality."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self.db_path = Path(
            db_path
            or os.getenv("INGESTION_STATUS_DB_PATH")
            or ".cache/graphrag_ingestion.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_status (
                paper_id TEXT PRIMARY KEY,
                paper_title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_attempted REAL NOT NULL,
                extraction_quality REAL NOT NULL DEFAULT 0.0,
                avg_confidence REAL NOT NULL DEFAULT 0.0,
                entity_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def upsert_status(
        self,
        *,
        paper_id: str,
        paper_title: str,
        source_path: str,
        status: str,
        error: str | None = None,
        retry_count: int = 0,
        extraction_quality: float = 0.0,
        avg_confidence: float = 0.0,
        entity_count: int = 0,
        chunk_count: int = 0,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_status (
                paper_id, paper_title, source_path, status, error, retry_count,
                last_attempted, extraction_quality, avg_confidence, entity_count, chunk_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                paper_title = excluded.paper_title,
                source_path = excluded.source_path,
                status = excluded.status,
                error = excluded.error,
                retry_count = excluded.retry_count,
                last_attempted = excluded.last_attempted,
                extraction_quality = excluded.extraction_quality,
                avg_confidence = excluded.avg_confidence,
                entity_count = excluded.entity_count,
                chunk_count = excluded.chunk_count
            """,
            (
                paper_id,
                paper_title,
                source_path,
                status,
                error,
                retry_count,
                time.time(),
                extraction_quality,
                avg_confidence,
                entity_count,
                chunk_count,
            ),
        )
        self._conn.commit()

    def list_statuses(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            """
            SELECT paper_id, paper_title, source_path, status, error, retry_count,
                   last_attempted, extraction_quality, avg_confidence, entity_count, chunk_count
            FROM ingestion_status
            ORDER BY last_attempted DESC
            LIMIT ?
            """,
            (limit,),
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_status(self, paper_id: str) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            """
            SELECT paper_id, paper_title, source_path, status, error, retry_count,
                   last_attempted, extraction_quality, avg_confidence, entity_count, chunk_count
            FROM ingestion_status
            WHERE paper_id = ?
            """,
            (paper_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, row))

    def extraction_quality_report(self, limit: int = 100) -> dict[str, Any]:
        rows = self.list_statuses(limit=limit)
        complete = [row for row in rows if row["status"] == "complete"]
        failed = [row for row in rows if row["status"] == "failed"]
        average_quality = (
            round(sum(float(row["extraction_quality"]) for row in complete) / len(complete), 3)
            if complete
            else 0.0
        )
        return {
            "papers": rows,
            "summary": {
                "paper_count": len(rows),
                "complete_papers": len(complete),
                "failed_papers": len(failed),
                "avg_extraction_quality": average_quality,
            },
        }


_default_store: IngestionStatusStore | None = None


def get_ingestion_status_store() -> IngestionStatusStore:
    global _default_store
    if _default_store is None:
        _default_store = IngestionStatusStore()
    return _default_store
