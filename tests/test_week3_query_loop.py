from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.graphrag.rag import QuerySynthesizer, StateGraph


class Week3QueryLoopTests(unittest.TestCase):
    def test_query_graph_cache_is_per_instance(self) -> None:
        if StateGraph is None:
            self.skipTest("langgraph is not available")

        synthesizer_a = QuerySynthesizer()
        synthesizer_b = QuerySynthesizer()

        graph_a = synthesizer_a._query_graph()
        graph_a_again = synthesizer_a._query_graph()
        graph_b = synthesizer_b._query_graph()

        self.assertIs(graph_a, graph_a_again)
        self.assertIsNot(graph_a, graph_b)

    def test_query_loop_refines_when_verification_fails(self) -> None:
        search_calls: list[str] = []

        def search_fn(*, query: str, top_k: int) -> dict[str, object]:
            search_calls.append(query)
            if query == "m6A methods":
                text = "WTAP is discussed, but the chunk does not describe the assay clearly."
            else:
                text = "MeRIP-seq and qRT-PCR were used to study m6A methylation in WTAP-regulated samples."
            return {
                "query": query,
                "text_hits": [
                    {
                        "paper_title": "Example paper",
                        "doi": "10.1000/example",
                        "section_title": "Results",
                        "text": text,
                    }
                ],
                "table_hits": [],
                "figure_hits": [],
                "papers": [],
                "entities": {},
                "citations": [],
                "stats": {},
            }

        with patch("backend.graphrag.rag.gemini_available", return_value=True):
            synthesizer = QuerySynthesizer()
        synthesizer.enabled = True

        with patch(
            "backend.graphrag.rag.generate_text",
            side_effect=[
                "The paper uses an unspecified method to study m6A.",
                "m6A methylation methods WTAP MeRIP-seq qRT-PCR",
                "The paper studies m6A using MeRIP-seq and qRT-PCR. [Source: Example paper, 2024]",
            ],
        ) as generate_text, patch(
            "backend.graphrag.rag.generate_json",
            side_effect=[
                {
                    "grounded": False,
                    "unsupported_claims": ["unspecified method"],
                    "confidence": 0.2,
                    "reason": "answer is too vague for the retrieved passage",
                },
                {
                    "grounded": True,
                    "unsupported_claims": [],
                    "confidence": 0.93,
                    "reason": "all method claims are supported",
                },
            ],
        ):
            result = synthesizer.answer(
                question="m6A methods",
                search_fn=search_fn,
                top_k=3,
                max_passages=5,
            )

        self.assertEqual(search_calls, ["m6A methods", "m6A methylation methods WTAP MeRIP-seq qRT-PCR"])
        self.assertEqual(result["refined_query"], "m6A methylation methods WTAP MeRIP-seq qRT-PCR")
        self.assertIn("MeRIP-seq", result["answer"])
        self.assertTrue(result["verification_result"]["grounded"])
        self.assertGreater(result["confidence"], 0.0)
        self.assertEqual(generate_text.call_count, 3)

    def test_query_loop_returns_structured_passages_when_synthesis_disabled(self) -> None:
        def search_fn(*, query: str, top_k: int) -> dict[str, object]:
            return {
                "query": query,
                "text_hits": [
                    {
                        "paper_title": "Example paper",
                        "doi": "10.1000/example",
                        "section_title": "Results",
                        "text": "MeRIP-seq was used to study m6A methylation in WTAP-regulated samples.",
                        "score": 0.91,
                    }
                ],
                "table_hits": [],
                "figure_hits": [],
                "papers": [],
                "entities": {},
                "citations": [],
                "stats": {},
            }

        with patch("backend.graphrag.rag.gemini_available", return_value=False):
            synthesizer = QuerySynthesizer()

        result = synthesizer.answer(
            question="m6A methods",
            search_fn=search_fn,
            top_k=3,
            max_passages=5,
        )

        self.assertIsNotNone(result["answer"])
        self.assertIn("Retrieved passages", result["answer"])
        self.assertEqual(len(result["passages"]), 1)
        self.assertEqual(result["passages"][0]["paper_title"], "Example paper")
        self.assertEqual(result["passages"][0]["section_title"], "Results")
        self.assertEqual(result["sources"][0]["paper"], "Example paper")
        self.assertFalse(result["verification_result"]["grounded"])


if __name__ == "__main__":
    unittest.main()
