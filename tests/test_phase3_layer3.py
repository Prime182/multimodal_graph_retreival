from __future__ import annotations

import unittest
from pathlib import Path

from backend.graphrag import build_layer3, chunk_article, extract_layer2, parse_article
from backend.graphrag.entities import Layer2DocumentRecord, Layer2EntityRecord
from backend.graphrag.edges import build_citation_edges, infer_claim_edges, infer_is_a_edges
from backend.graphrag.models import PaperRecord, ReferenceRecord
from backend.graphrag.embeddings import HashingEmbedder


def _article_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"No test article found in candidates: {candidates}")


class Phase3Layer3Tests(unittest.TestCase):
    def test_parser_extracts_bibliography_references(self) -> None:
        article = parse_article(_article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml"))

        self.assertGreater(len(article.references), 50)
        self.assertTrue(
            any(reference.doi or (reference.source_text and "10." in reference.source_text) for reference in article.references)
        )

    def test_citation_edges_resolve_matching_doi(self) -> None:
        cited = PaperRecord(
            paper_id="cited-paper",
            source_path="cited.xml",
            title="Cited Paper",
            doi="10.9999/cited",
        )
        citing = PaperRecord(
            paper_id="citing-paper",
            source_path="citing.xml",
            title="Citing Paper",
            doi="10.9999/citing",
            references=[
                ReferenceRecord(
                    reference_id="ref-1",
                    title="Cited Paper",
                    doi="10.9999/cited",
                    source_text="Cited Paper. doi: 10.9999/cited",
                )
            ],
        )

        edges = build_citation_edges([citing, cited])

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, "CITES")
        self.assertEqual(edges[0].source_node_id, "citing-paper")
        self.assertEqual(edges[0].target_node_id, "cited-paper")

    def test_citation_edges_create_stub_targets_for_external_references(self) -> None:
        citing = PaperRecord(
            paper_id="citing-paper",
            source_path="citing.xml",
            title="Citing Paper",
            doi="10.9999/citing",
            references=[
                ReferenceRecord(
                    reference_id="ref-1",
                    title="External Paper",
                    source_text="External Paper. doi: 10.9999/external",
                )
            ],
        )

        edges = build_citation_edges([citing])

        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0].target_node_id.startswith("refpaper-"))
        self.assertEqual(edges[0].target_label, "External Paper")
        self.assertTrue(edges[0].metadata.get("is_stub_target"))

    def test_claim_edges_and_is_a_edges_are_inferred(self) -> None:
        embedder = HashingEmbedder(dim=256)
        doc_a = Layer2DocumentRecord(
            paper_id="paper-a",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="claim-a",
                    entity_type="claim",
                    label="The method improves accuracy on Dataset X.",
                    source_chunk_id="chunk-a",
                    confidence=0.9,
                    extractor_model="heuristic-v2",
                    embedding=embedder.embed("The method improves accuracy on Dataset X."),
                    properties={"text": "The method improves accuracy on Dataset X.", "claim_type": "finding"},
                ),
                Layer2EntityRecord(
                    entity_id="concept-a",
                    entity_type="concept",
                    label="CRISPR",
                    source_chunk_id="chunk-a",
                    confidence=0.9,
                    extractor_model="heuristic-v2",
                    embedding=embedder.embed("CRISPR"),
                    properties={"ontology": ""},
                ),
            ],
        )
        doc_b = Layer2DocumentRecord(
            paper_id="paper-b",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="claim-b",
                    entity_type="claim",
                    label="The method reduces accuracy on Dataset X.",
                    source_chunk_id="chunk-b",
                    confidence=0.9,
                    extractor_model="heuristic-v2",
                    embedding=embedder.embed("The method reduces accuracy on Dataset X."),
                    properties={"text": "The method reduces accuracy on Dataset X.", "claim_type": "finding"},
                ),
            ],
        )
        doc_c = Layer2DocumentRecord(
            paper_id="paper-c",
            extractor_model="heuristic-v2",
            entities=[
                Layer2EntityRecord(
                    entity_id="claim-c",
                    entity_type="claim",
                    label="However, the method reduces accuracy on Dataset X.",
                    source_chunk_id="chunk-c",
                    confidence=0.9,
                    extractor_model="heuristic-v2",
                    embedding=embedder.embed("However, the method reduces accuracy on Dataset X."),
                    properties={"text": "However, the method reduces accuracy on Dataset X.", "claim_type": "finding"},
                ),
            ],
        )

        doc_b.entities.append(
            Layer2EntityRecord(
                entity_id="method-b",
                entity_type="method",
                label="CRISPR-CAS9",
                source_chunk_id="chunk-b",
                confidence=0.8,
                extractor_model="heuristic-v2",
                embedding=embedder.embed("CRISPR-CAS9"),
                properties={"method_type": "computational", "first_paper_id": "paper-b"},
            )
        )
        doc_b.entities.append(
            Layer2EntityRecord(
                entity_id="concept-b",
                entity_type="concept",
                label="CRISPR",
                source_chunk_id="chunk-b",
                confidence=0.9,
                extractor_model="heuristic-v2",
                embedding=embedder.embed("CRISPR"),
                properties={"ontology": ""},
            )
        )

        claim_edges = infer_claim_edges([doc_a, doc_b, doc_c], similarity_threshold=0.6)
        is_a_edges = infer_is_a_edges([doc_a, doc_b, doc_c])

        self.assertTrue(any(edge.relation_type == "SUPPORTS" for edge in claim_edges))
        self.assertTrue(any(edge.relation_type == "CONTRADICTS" for edge in claim_edges))
        self.assertTrue(any(edge.relation_type == "IS_A" for edge in is_a_edges))

    def test_build_layer3_combines_claims_and_citations(self) -> None:
        paper = chunk_article(parse_article(_article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml")))
        layer2 = extract_layer2(paper)
        layer3 = build_layer3([paper], [layer2])

        self.assertGreaterEqual(len(layer3.semantic_edges), 0)
        self.assertGreaterEqual(len(layer3.citation_edges), 0)

    def test_biomedical_layer3_infers_method_hierarchies_and_reference_stubs(self) -> None:
        paper = chunk_article(parse_article(_article_path("articles/BJ_100828.xml", "articles/ESR-102001.xml")))
        layer2 = extract_layer2(paper)
        layer3 = build_layer3([paper], [layer2])

        is_a_pairs = {
            (edge.source_label, edge.target_label)
            for edge in layer3.semantic_edges
            if edge.relation_type == "IS_A"
        }
        self.assertIn(("CRISPRa", "CRISPR"), is_a_pairs)
        self.assertIn(("FACS", "Flow cytometry"), is_a_pairs)
        self.assertGreater(len(layer3.citation_edges), 0)


if __name__ == "__main__":
    unittest.main()
