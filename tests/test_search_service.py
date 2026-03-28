from __future__ import annotations

import unittest

from backend.graphrag import LocalVectorIndex
from backend.graphrag.edges import Layer3CorpusRecord, Layer3EdgeRecord
from backend.graphrag.entities import Layer2DocumentRecord, Layer2EntityRecord
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord, TableRecord
from backend.graphrag.search_service import build_search_bundle


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
        index = LocalVectorIndex([paper_a, paper_b])
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


if __name__ == "__main__":
    unittest.main()
