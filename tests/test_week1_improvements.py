from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from backend.graphrag.corpus import CorpusClient, CorpusMatch, lookup_ols
from backend.graphrag.embeddings import GeminiEmbedder
from backend.graphrag.extraction import _GeminiChunkExtractionPayload, _gemini_extract_chunk_entities
from backend.graphrag.gemini import GeminiError
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord
from backend.graphrag.retrieval import LocalVectorIndex


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

    def test_local_vector_index_defaults_to_gemini_embedder(self) -> None:
        index = LocalVectorIndex(documents=[])
        self.assertIsInstance(index.embedder, GeminiEmbedder)

    def test_gemini_extraction_falls_back_cleanly_on_gemini_error(self) -> None:
        paper = PaperRecord(paper_id="paper-1", source_path="paper.xml", title="Paper")
        section = SectionRecord(
            section_id="sec-1",
            paper_id="paper-1",
            title="Results",
            section_type="results",
            level=1,
            ordinal=0,
            text="WTAP increased methylation.",
        )
        chunk = ChunkRecord(
            chunk_id="chunk-1",
            paper_id="paper-1",
            section_id="sec-1",
            ordinal=0,
            text="WTAP increased methylation.",
            chunk_type="results",
            word_count=3,
            token_count=3,
            salience_score=0.1,
        )

        with patch("backend.graphrag.extraction.gemini_available", return_value=True), patch(
            "backend.graphrag.extraction.generate_json",
            side_effect=GeminiError("temporary outage"),
        ):
            self.assertIsNone(_gemini_extract_chunk_entities(paper, section, chunk))

    def test_lookup_ols_parses_real_api_payloads(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "response": {
                "docs": [
                    {
                        "label": "flow cytometry",
                        "ontology_name": "obi",
                        "obo_id": "OBI:0000716",
                        "description": ["A method for counting cells."],
                        "synonym": ["FACS"],
                        "type": "class",
                    }
                ]
            }
        }

        with patch("backend.graphrag.corpus.requests.get", return_value=response):
            match = lookup_ols("flow cytometry", "obi")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertTrue(match.found)
        self.assertEqual(match.label, "flow cytometry")
        self.assertEqual(match.external_id, "OBI:0000716")
        self.assertIn("FACS", match.aliases)


if __name__ == "__main__":
    unittest.main()
