from __future__ import annotations

from dataclasses import dataclass
import unittest
from typing import Any

from backend.graphrag.bm25_index import BM25Index
from backend.graphrag.context_builder import ContextBuilder
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord
from backend.graphrag.retriever import HybridRetriever, RetrievedPassage


@dataclass(slots=True)
class _Hit:
    score: float
    paper_id: str
    paper_title: str
    section_id: str
    chunk_id: str
    section_title: str
    text: str
    doi: str | None = None
    graph_context: dict[str, Any] | None = None


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        seed = sum(ord(char) for char in text)
        return [float((seed % 11) + 1), float((seed % 7) + 1), 1.0]


class _StubVectorIndex:
    def __init__(self, hits: list[_Hit]) -> None:
        self._hits = hits
        self.embedder = _StubEmbedder()

    def search(self, query: str, top_k: int = 5, section_type: str | None = None) -> list[_Hit]:
        return self._hits[:top_k]


class _StubBM25Index:
    def __init__(self, hits: list[_Hit]) -> None:
        self._hits = hits

    def search(self, query: str, top_k: int = 5, section_type: str | None = None) -> list[_Hit]:
        return self._hits[:top_k]


class _StubGraphBackend:
    def __init__(self, hits: list[_Hit]) -> None:
        self._hits = hits

    def search_chunks(self, embedding: list[float], top_k: int = 5) -> list[_Hit]:
        return self._hits[:top_k]


class Phase4RetrievalPrimitiveTests(unittest.TestCase):
    def test_bm25_prefers_matching_chunk(self) -> None:
        paper = PaperRecord(
            paper_id="paper-1",
            source_path="paper-1.xml",
            title="Battery systems",
            sections=[
                SectionRecord(
                    section_id="sec-1",
                    paper_id="paper-1",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="Battery charging battery charging battery storage.",
                ),
                SectionRecord(
                    section_id="sec-2",
                    paper_id="paper-1",
                    title="Discussion",
                    section_type="discussion",
                    level=1,
                    ordinal=1,
                    text="Solar cells and policy.",
                ),
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    section_id="sec-1",
                    ordinal=0,
                    text="Battery charging battery charging battery storage.",
                    chunk_type="results",
                    word_count=6,
                    token_count=6,
                    salience_score=0.2,
                ),
                ChunkRecord(
                    chunk_id="chunk-2",
                    paper_id="paper-1",
                    section_id="sec-2",
                    ordinal=0,
                    text="Solar cells and policy.",
                    chunk_type="discussion",
                    word_count=4,
                    token_count=4,
                    salience_score=0.2,
                ),
            ],
        )

        index = BM25Index.from_papers([paper])
        hits = index.search("battery charging", top_k=2)

        self.assertEqual(hits[0].chunk_id, "chunk-1")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_hybrid_retriever_uses_rrf_and_keeps_graph_context(self) -> None:
        vector_hits = [
            _Hit(0.95, "paper-a", "Paper A", "sec-a", "chunk-a", "Results", "alpha beta"),
            _Hit(0.85, "paper-b", "Paper B", "sec-b", "chunk-b", "Results", "beta gamma"),
            _Hit(0.75, "paper-c", "Paper C", "sec-c", "chunk-c", "Results", "gamma delta"),
        ]
        bm25_hits = [
            _Hit(0.90, "paper-b", "Paper B", "sec-b", "chunk-b", "Results", "beta gamma"),
            _Hit(0.80, "paper-a", "Paper A", "sec-a", "chunk-a", "Results", "alpha beta"),
            _Hit(0.70, "paper-c", "Paper C", "sec-c", "chunk-c", "Results", "gamma delta"),
        ]
        graph_hits = [
            _Hit(
                0.99,
                "paper-a",
                "Paper A",
                "sec-a",
                "chunk-a",
                "Results",
                "alpha beta",
                graph_context={
                    "nodes": [{"label": "WTAP", "node_type": "Concept"}],
                    "edges": [{"relation_type": "IS_A", "source_label": "CRISPRa", "target_label": "CRISPR"}],
                    "seed_node_ids": ["node-1"],
                },
            ),
            _Hit(0.88, "paper-b", "Paper B", "sec-b", "chunk-b", "Results", "beta gamma"),
            _Hit(0.77, "paper-c", "Paper C", "sec-c", "chunk-c", "Results", "gamma delta"),
        ]

        retriever = HybridRetriever(
            vector_index=_StubVectorIndex(vector_hits),
            bm25_index=_StubBM25Index(bm25_hits),
            graph_backend=_StubGraphBackend(graph_hits),
        )

        passages = retriever.retrieve("alpha beta", top_k=3)

        self.assertEqual(passages[0].chunk_id, "chunk-a")
        self.assertEqual(passages[0].graph_context["seed_node_ids"], ["node-1"])
        self.assertEqual(passages[0].retrieval_method, "fusion")
        self.assertGreater(passages[0].score, passages[1].score)

    def test_context_builder_formats_graph_context_and_neighbor_text(self) -> None:
        passage = RetrievedPassage(
            rank=1,
            score=0.91,
            chunk_id="chunk-1",
            paper_id="paper-1",
            paper_title="Paper One",
            section_id="sec-1",
            section_title="Results",
            text="The method improved accuracy on Dataset X.",
            doi="10.1000/test",
            retrieval_method="fusion",
            graph_context={
                "nodes": [{"label": "WTAP", "node_type": "Concept"}, {"label": "CRISPRa", "node_type": "Method"}],
                "edges": [{"relation_type": "IS_A", "source_label": "CRISPRa", "target_label": "CRISPR"}],
                "seed_node_ids": ["node-1"],
            },
            previous_text="Previous chunk text.",
            next_text="Next chunk text.",
        )

        context = ContextBuilder().build("accuracy", [passage], max_tokens=200)

        self.assertEqual(context.passage_count, 1)
        self.assertIn("Graph context: 2 nodes, 1 edges", context.formatted_context)
        self.assertIn("WTAP (Concept)", context.formatted_context)
        self.assertIn("CRISPRa -[IS_A]-> CRISPR", context.formatted_context)
        self.assertIn("Previous: Previous chunk text.", context.formatted_context)
        self.assertIn("Current: The method improved accuracy on Dataset X.", context.formatted_context)
        self.assertIn("[1] Paper One", context.source_map[1])


if __name__ == "__main__":
    unittest.main()
