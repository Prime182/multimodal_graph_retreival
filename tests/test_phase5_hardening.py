from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.graphrag.extraction_cache import ExtractionCache
from backend.graphrag.ingestion_status import IngestionStatusStore


class Phase5HardeningTests(unittest.TestCase):
    def test_ingestion_status_tracks_file_hash_and_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = IngestionStatusStore(db_path=Path(temp_dir) / "ingestion.sqlite3")
            store.upsert_status(
                paper_id="paper-1",
                paper_title="Paper 1",
                source_path="paper1.xml",
                status="complete",
                file_hash="hash-a",
                model_version="model-v1",
                extraction_quality=0.9,
                avg_confidence=0.88,
                entity_count=7,
                chunk_count=3,
            )

            status = store.get_status("paper-1")
            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status["file_hash"], "hash-a")
            self.assertEqual(status["model_version"], "model-v1")
            self.assertTrue(store.is_complete("paper-1", file_hash="hash-a", model_version="model-v1"))
            self.assertFalse(store.is_complete("paper-1", file_hash="hash-b", model_version="model-v1"))
            self.assertFalse(store.is_complete("paper-1", file_hash="hash-a", model_version="model-v2"))

            store.upsert_status(
                paper_id="paper-1",
                paper_title="Paper 1",
                source_path="paper1.xml",
                status="failed",
                file_hash="hash-a",
                model_version="model-v1",
                error="boom",
            )
            self.assertFalse(store.is_complete("paper-1", file_hash="hash-a", model_version="model-v1"))

    def test_extraction_cache_uses_chunk_hash_and_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ExtractionCache(db_path=Path(temp_dir) / "extraction.sqlite3")
            payload = {
                "entities": [
                    {"type": "concept", "name": "WTAP", "confidence": 0.9},
                    {"type": "result", "value": 32.0, "metric": "percentage"},
                ],
                "salience_score": 0.81,
            }

            self.assertIsNone(cache.get("chunk-1", "example text", "model-v1"))
            cache.set("chunk-1", "example text", "model-v1", payload)

            self.assertEqual(cache.get("chunk-1", "example text", "model-v1"), payload)
            self.assertIsNone(cache.get("chunk-1", "example text changed", "model-v1"))
            self.assertIsNone(cache.get("chunk-1", "example text", "model-v2"))

            reopened = ExtractionCache(db_path=Path(temp_dir) / "extraction.sqlite3")
            self.assertEqual(reopened.get("chunk-1", "example text", "model-v1"), payload)


if __name__ == "__main__":
    unittest.main()
