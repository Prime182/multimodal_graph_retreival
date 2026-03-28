from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.graphrag.extraction import extract_layer2
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord


class Week2LangGraphExtractionTests(unittest.TestCase):
    def test_langgraph_self_correction_replaces_invalid_method_payload(self) -> None:
        paper = PaperRecord(
            paper_id="paper-1",
            source_path="paper.xml",
            title="Example paper",
            sections=[
                SectionRecord(
                    section_id="sec-1",
                    paper_id="paper-1",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="MeRIP-seq revealed a 32% increase in m6A methylation.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    section_id="sec-1",
                    ordinal=0,
                    text="MeRIP-seq revealed a 32% increase in m6A methylation.",
                    chunk_type="results",
                    word_count=10,
                    token_count=10,
                    salience_score=0.1,
                )
            ],
        )

        initial_payload = {
            "entities": [
                {
                    "type": "method",
                    "name": "method",
                    "aliases": [],
                    "confidence": 0.92,
                }
            ],
            "salience_score": 0.42,
        }
        corrected_payload = {
            "entities": [
                {
                    "type": "method",
                    "name": "MeRIP-seq",
                    "aliases": [],
                    "confidence": 0.96,
                },
                {
                    "type": "result",
                    "value": 32.0,
                    "unit": "%",
                    "metric": "m6A methylation",
                    "dataset": "",
                    "condition": "",
                    "text": "MeRIP-seq revealed a 32% increase in m6A methylation.",
                    "aliases": [],
                    "confidence": 0.88,
                },
            ],
            "salience_score": 0.83,
        }

        with patch("backend.graphrag.extraction.gemini_available", return_value=True), patch(
            "backend.graphrag.extraction.generate_json",
            side_effect=[initial_payload, corrected_payload],
        ) as generate_json:
            extraction = extract_layer2(paper, use_gemini=True)

        labels = {entity.label for entity in extraction.entities}
        self.assertIn("MeRIP-seq", labels)
        self.assertNotIn("method", labels)
        self.assertEqual(generate_json.call_count, 2)
        self.assertGreater(extraction.chunk_salience_scores["chunk-1"], 0.8)


if __name__ == "__main__":
    unittest.main()
