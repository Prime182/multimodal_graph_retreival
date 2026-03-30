from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from backend.graphrag.corpus import CorpusClient, CorpusMatch, lookup_mesh, lookup_ols
from backend.graphrag.embeddings import (
    GeminiEmbedder,
    SentenceTransformerEmbedder,
    build_entity_embedder,
    probe_embedding_backends,
)
from backend.graphrag.extraction import _GeminiChunkExtractionPayload, _gemini_extract_chunk_entities
from backend.graphrag.gemini import GeminiError
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord
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

    def test_build_entity_embedder_prefers_gemini_when_available(self) -> None:
        with patch("backend.graphrag.embeddings.gemini_available", return_value=True):
            embedder = build_entity_embedder(dim=16, prefer_remote=True)

        self.assertIsInstance(embedder, GeminiEmbedder)

    def test_local_fallback_embedder_is_sentence_transformers_not_hashing(self) -> None:
        with patch("backend.graphrag.embeddings.gemini_available", return_value=False):
            embedder = build_entity_embedder(prefer_remote=True)

        self.assertIsInstance(embedder, SentenceTransformerEmbedder)

    def test_embedding_probe_reports_local_fallback_loadable(self) -> None:
        with patch("backend.graphrag.embeddings.gemini_available", return_value=False), patch(
            "backend.graphrag.embeddings.SentenceTransformerEmbedder.embed",
            return_value=[1.0, 0.0, 0.0],
        ):
            status = probe_embedding_backends(dim=3)

        self.assertTrue(status["local_available"])
        self.assertEqual(status["active_backend"], "sentence-transformers")

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

    def test_lookup_mesh_uses_mesh_api_payload(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            {
                "label": "Hypertension",
                "resource": "http://id.nlm.nih.gov/mesh/D006973",
                "synonym": ["High blood pressure"],
            }
        ]

        with patch("backend.graphrag.corpus.requests.get", return_value=response):
            match = lookup_mesh("hypertension")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertTrue(match.found)
        self.assertEqual(match.ontology, "MESH")
        self.assertEqual(match.label, "Hypertension")
        self.assertIn("High blood pressure", match.aliases)


if __name__ == "__main__":
    unittest.main()
