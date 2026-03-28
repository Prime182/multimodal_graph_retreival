from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend.graphrag.corpus import CorpusClient, CorpusMatch
from backend.graphrag.extraction import _GeminiChunkExtractionPayload


class Week1ImprovementTests(unittest.TestCase):
    def test_gemini_payload_validation_rejects_incomplete_result_entities(self) -> None:
        with self.assertRaises(ValidationError):
            _GeminiChunkExtractionPayload.model_validate(
                {
                    "entities": [
                        {
                            "type": "result",
                            "metric": "Accuracy",
                            "confidence": 0.8,
                        }
                    ],
                    "salience_score": 0.5,
                }
            )

    def test_corpus_client_caches_entity_lookups_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = CorpusClient(cache_path=Path(temp_dir) / "corpus.sqlite3")
            match = CorpusMatch(
                found=True,
                label="WTAP",
                aliases=["Wilms tumor 1 associated protein"],
                ontology="NCBI-Gene",
                external_id="9589",
                category="gene",
            )

            with patch("backend.graphrag.corpus.lookup_ncbi_gene", return_value=match) as lookup:
                first = client.enrich_entity("WTAP", "gene")
                second = client.enrich_entity("WTAP", "gene")

            self.assertTrue(first.found)
            self.assertEqual(second.external_id, "9589")
            self.assertEqual(lookup.call_count, 1)


if __name__ == "__main__":
    unittest.main()
