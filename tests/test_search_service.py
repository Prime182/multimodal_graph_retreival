from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.graphrag.edges import Layer3CorpusRecord, Layer3EdgeRecord
from backend.graphrag.entities import Layer2DocumentRecord, Layer2EntityRecord
from backend.graphrag.models import ChunkRecord, PaperRecord, SearchHit, SectionRecord, TableRecord
from backend.graphrag.retrieval import FigureHit, TableHit
from backend.graphrag.retriever import RetrievedPassage
from backend.graphrag.search_service import GraphRAGSearchService, build_search_bundle


class _StubRetriever:
    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self._passages = passages

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        return self._passages[:top_k]


class _StubGraphBackend:
    def get_entity_neighborhood(self, entity_ids: list[str], hops: int = 2):
        return {
            "nodes": [{"node_id": "concept-1", "label": "WTAP", "node_type": "Concept"}],
            "edges": [],
            "seed_node_ids": entity_ids,
        }

    def get_chunks_mentioning_entities(self, entity_ids: list[str]) -> list[str]:
        return ["paper-a-results-chunk-0"]


class _StubIndex:
    def __init__(
        self,
        papers: list[PaperRecord] | None = None,
        *,
        text_hits: list[SearchHit] | None = None,
        table_hits: list[TableHit] | None = None,
        figure_hits: list[FigureHit] | None = None,
        paper_hits: list[tuple[float, PaperRecord]] | None = None,
    ) -> None:
        self.papers = papers or []
        if text_hits is not None:
            self._text_hits = text_hits
        elif self.papers and self.papers[0].chunks:
            first_paper = self.papers[0]
            first_chunk = first_paper.chunks[0]
            section_title = first_chunk.section_id
            if first_paper.sections:
                for section in first_paper.sections:
                    if section.section_id == first_chunk.section_id:
                        section_title = section.title
                        break
            self._text_hits = [
                SearchHit(
                    score=0.91,
                    paper_id=first_paper.paper_id,
                    paper_title=first_paper.title,
                    section_id=first_chunk.section_id,
                    chunk_id=first_chunk.chunk_id,
                    section_title=section_title,
                    text=first_chunk.text,
                    doi=first_paper.doi,
                )
            ]
        else:
            self._text_hits = []
        self._table_hits = table_hits or []
        self._figure_hits = figure_hits or []
        self._paper_hits = paper_hits or []

    def search(self, query: str, top_k: int = 5, section_type: str | None = None) -> list[SearchHit]:
        return self._text_hits[:top_k]

    def search_tables(self, query: str, top_k: int = 5) -> list[TableHit]:
        return self._table_hits[:top_k]

    def search_figures(self, query: str, top_k: int = 5) -> list[FigureHit]:
        return self._figure_hits[:top_k]

    def search_papers(self, query: str, top_k: int = 5) -> list[tuple[float, PaperRecord]]:
        return self._paper_hits[:top_k]


class SearchServiceTests(unittest.TestCase):
    def test_build_search_bundle_includes_tables_entities_and_citations(self) -> None:
        paper_a = PaperRecord(
            paper_id="paper-a",
            source_path="paper-a.xml",
            title="Main paper",
            doi="10.1000/a",
            abstract="This paper reports accuracy results and a structured equation.",
            sections=[
                SectionRecord(
                    section_id="paper-a-results",
                    paper_id="paper-a",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="The method improves accuracy on Dataset X.",
                    paragraphs=["The method improves accuracy on Dataset X."],
                    key_sentence="The method improves accuracy on Dataset X.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="paper-a-results-chunk-0",
                    paper_id="paper-a",
                    section_id="paper-a-results",
                    ordinal=0,
                    text="The method improves accuracy on Dataset X.",
                    chunk_type="results",
                    word_count=7,
                    token_count=7,
                    salience_score=0.7,
                )
            ],
            tables=[
                TableRecord(
                    table_id="table-a",
                    paper_id="paper-a",
                    ordinal=1,
                    label="Table 1",
                    caption="Accuracy results",
                    text="Table 1 Accuracy results on Dataset X.",
                    rows=3,
                    columns=2,
                )
            ],
        )
        paper_b = PaperRecord(
            paper_id="paper-b",
            source_path="paper-b.xml",
            title="Cited paper",
            doi="10.1000/b",
            abstract="Foundational work.",
        )
        layer2 = Layer2DocumentRecord(
            paper_id="paper-a",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="claim-a",
                    entity_type="claim",
                    label="The method improves accuracy on Dataset X.",
                    source_chunk_id="paper-a-results-chunk-0",
                    confidence=0.92,
                    extractor_model="heuristic-v2",
                    properties={"claim_type": "finding", "text": "The method improves accuracy on Dataset X."},
                ),
                Layer2EntityRecord(
                    entity_id="equation-a",
                    entity_type="equation",
                    label="x^2 = y",
                    source_chunk_id="paper-a-results-chunk-0",
                    confidence=0.84,
                    extractor_model="heuristic-v2",
                    properties={"latex": "x^2 = y", "plain_desc": "quadratic relationship", "is_loss_fn": False},
                ),
            ],
        )
        index = _StubIndex(
            [paper_a, paper_b],
            text_hits=[
                SearchHit(
                    score=0.93,
                    paper_id="paper-a",
                    paper_title="Main paper",
                    section_id="paper-a-results",
                    chunk_id="paper-a-results-chunk-0",
                    section_title="Results",
                    text="The method improves accuracy on Dataset X.",
                    doi="10.1000/a",
                )
            ],
            table_hits=[
                TableHit(
                    score=0.88,
                    paper_id="paper-a",
                    paper_title="Main paper",
                    table_id="table-a",
                    label="Table 1",
                    caption="Accuracy results",
                    text="Table 1 Accuracy results on Dataset X.",
                    doi="10.1000/a",
                )
            ],
            figure_hits=[],
            paper_hits=[(0.77, paper_a), (0.21, paper_b)],
        )
        layer3 = Layer3CorpusRecord(
            citation_edges=[
                Layer3EdgeRecord(
                    edge_id="cite-1",
                    relation_type="CITES",
                    source_node_id="paper-a",
                    source_node_type="Paper",
                    source_label="Main paper",
                    target_node_id="paper-b",
                    target_node_type="Paper",
                    target_label="Cited paper",
                    confidence=0.99,
                    source_chunk_id="",
                    extractor_model="bibliography-parser",
                    evidence="Main paper cites foundational work.",
                    metadata={},
                )
            ]
        )

        payload = build_search_bundle(
            query="accuracy results",
            index=index,
            papers=[paper_a, paper_b],
            layer2_docs=[layer2],
            layer3=layer3,
            top_k=3,
        )

        self.assertGreaterEqual(len(payload["text_hits"]), 1)
        self.assertGreaterEqual(len(payload["table_hits"]), 1)
        self.assertIn("equation", payload["entities"])
        self.assertEqual(len(payload["citations"]), 1)
        self.assertEqual(payload["citations"][0]["target_paper_title"], "Cited paper")

    def test_build_search_bundle_uses_hybrid_retrieval_and_graph_context(self) -> None:
        paper = PaperRecord(
            paper_id="paper-a",
            source_path="paper-a.xml",
            title="Main paper",
            abstract="WTAP improves accuracy.",
            sections=[
                SectionRecord(
                    section_id="paper-a-results",
                    paper_id="paper-a",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="WTAP improves accuracy on Dataset X.",
                    paragraphs=["WTAP improves accuracy on Dataset X."],
                    key_sentence="WTAP improves accuracy on Dataset X.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="paper-a-results-chunk-0",
                    paper_id="paper-a",
                    section_id="paper-a-results",
                    ordinal=0,
                    text="WTAP improves accuracy on Dataset X.",
                    chunk_type="results",
                    word_count=6,
                    token_count=6,
                    salience_score=0.7,
                )
            ],
        )
        layer2 = Layer2DocumentRecord(
            paper_id="paper-a",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="concept-1",
                    entity_type="concept",
                    label="WTAP",
                    source_chunk_id="paper-a-results-chunk-0",
                    confidence=0.94,
                    extractor_model="heuristic-v2",
                    properties={"name": "WTAP"},
                )
            ],
        )
        passage = RetrievedPassage(
            rank=1,
            score=0.91,
            chunk_id="paper-a-results-chunk-0",
            paper_id="paper-a",
            paper_title="Main paper",
            section_id="paper-a-results",
            section_title="Results",
            text="WTAP improves accuracy on Dataset X.",
            retrieval_method="fusion",
        )

        payload = build_search_bundle(
            query="WTAP accuracy",
            index=_StubIndex(
                [paper],
                text_hits=[
                    SearchHit(
                        score=0.91,
                        paper_id="paper-a",
                        paper_title="Main paper",
                        section_id="paper-a-results",
                        chunk_id="paper-a-results-chunk-0",
                        section_title="Results",
                        text="WTAP improves accuracy on Dataset X.",
                        doi=None,
                    )
                ],
            ),
            papers=[paper],
            layer2_docs=[layer2],
            layer3=Layer3CorpusRecord(),
            top_k=1,
            retriever=_StubRetriever([passage]),
            graph_backend=_StubGraphBackend(),
        )

        self.assertEqual(payload["text_hits"][0]["retrieval_method"], "fusion")
        self.assertEqual(payload["text_hits"][0]["graph_context"]["seed_node_ids"], ["concept-1"])
        self.assertEqual(payload["text_hits"][0]["entities"][0]["entity_id"], "concept-1")

    def test_search_service_can_background_load(self) -> None:
        paper = PaperRecord(
            paper_id="paper-a",
            source_path="paper-a.xml",
            title="Main paper",
            abstract="WTAP improves accuracy.",
            sections=[
                SectionRecord(
                    section_id="paper-a-results",
                    paper_id="paper-a",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="WTAP improves accuracy on Dataset X.",
                    paragraphs=["WTAP improves accuracy on Dataset X."],
                    key_sentence="WTAP improves accuracy on Dataset X.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="paper-a-results-chunk-0",
                    paper_id="paper-a",
                    section_id="paper-a-results",
                    ordinal=0,
                    text="WTAP improves accuracy on Dataset X.",
                    chunk_type="results",
                    word_count=6,
                    token_count=6,
                    salience_score=0.7,
                )
            ],
        )
        layer2 = Layer2DocumentRecord(
            paper_id="paper-a",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="concept-1",
                    entity_type="concept",
                    label="WTAP",
                    source_chunk_id="paper-a-results-chunk-0",
                    confidence=0.94,
                    extractor_model="heuristic-v2",
                    properties={"name": "WTAP"},
                )
            ],
        )

        class _DummyStatusStore:
            pass

        class _DummyTrace:
            def __enter__(self):
                return {}

            def __exit__(self, exc_type, exc, tb):
                return None

        class _DummyTracer:
            enabled = False

            def trace(self, *args, **kwargs):
                return _DummyTrace()

            def log_retrieval(self, *args, **kwargs):
                return None

        with patch("backend.graphrag.search_service._load_corpus", return_value=([paper], [layer2], Layer3CorpusRecord())), patch(
            "backend.graphrag.search_service.get_ingestion_status_store",
            return_value=_DummyStatusStore(),
        ), patch.object(GraphRAGSearchService, "_build_graph_retrieval", return_value=None), patch(
            "backend.graphrag.search_service.LocalVectorIndex",
            _StubIndex,
        ), patch("backend.graphrag.search_service.get_tracing_manager", return_value=_DummyTracer()):
            service = GraphRAGSearchService(input_dir="articles", load_on_init=False)
            self.assertIsNone(service.index)
            thread = service.start_background_load()
            thread.join(timeout=2)

        self.assertFalse(service.loading)
        self.assertIsNotNone(service.index)
        result = service.search("WTAP accuracy", top_k=1)
        self.assertGreaterEqual(len(result["text_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
