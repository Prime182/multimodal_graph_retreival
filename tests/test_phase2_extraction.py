from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.graphrag import chunk_article, extract_layer2, parse_article
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord


def _article_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"No test article found in candidates: {candidates}")


class Phase2ExtractionTests(unittest.TestCase):
    class _StubEmbedder:
        def embed(self, text: str) -> list[float]:
            seed = sum(ord(char) for char in text)
            return [float((seed % 13) + 1), float((seed % 7) + 1), 1.0]

    def test_extracts_layer2_entities_from_esr_article(self) -> None:
        article_path = _article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml")
        article = chunk_article(parse_article(article_path))
        initial_salience = {chunk.chunk_id: chunk.salience_score for chunk in article.chunks}

        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=self._StubEmbedder()):
            extraction = extract_layer2(article)

        entity_types = {entity.entity_type for entity in extraction.entities}
        self.assertIn("concept", entity_types)
        self.assertIn("claim", entity_types)
        self.assertIn("result", entity_types)
        if article_path.endswith("ESR-102001.xml"):
            self.assertTrue(
                any(
                    entity.entity_type == "result" and entity.properties.get("dataset") == "Scenario 2"
                    for entity in extraction.entities
                )
            )
        else:
            self.assertTrue(
                any(
                    entity.entity_type == "result"
                    and "virulent macrophages (V)" in entity.properties.get("datasets", [])
                    for entity in extraction.entities
                )
            )
        self.assertGreater(max(extraction.chunk_salience_scores.values()), max(initial_salience.values()))

    def test_concepts_include_article_keywords(self) -> None:
        article = chunk_article(parse_article(_article_path("articles/BJ_100828.xml", "articles/ESR-102001.xml")))
        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=self._StubEmbedder()):
            extraction = extract_layer2(article)

        concept_labels = [entity.label.lower() for entity in extraction.entities if entity.entity_type == "concept"]
        self.assertTrue(any("theileria" in label for label in concept_labels))
        self.assertTrue(any("m6a" in label or "hif1" in label for label in concept_labels))
        self.assertTrue(extraction.chunk_entity_ids)

    def test_document_registry_and_suffix_methods_are_detected(self) -> None:
        paper = PaperRecord(
            paper_id="synthetic-registry-paper",
            source_path="synthetic-registry.xml",
            title="Synthetic registry paper",
            abstract=(
                "Control group (C) and Treated group (T) are defined in the abstract. "
                "The study uses these labels consistently."
            ),
            sections=[
                SectionRecord(
                    section_id="synthetic-methods-section",
                    paper_id="synthetic-registry-paper",
                    title="Methods",
                    section_type="methods",
                    level=1,
                    ordinal=0,
                    text="We used Raman spectroscopy and protein chromatography to profile samples.",
                    paragraphs=["We used Raman spectroscopy and protein chromatography to profile samples."],
                    key_sentence="We used Raman spectroscopy and protein chromatography to profile samples.",
                ),
                SectionRecord(
                    section_id="synthetic-results-section",
                    paper_id="synthetic-registry-paper",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=1,
                    text="C achieved 92% accuracy. T reduced error to 0.03.",
                    paragraphs=[
                        "C achieved 92% accuracy.",
                        "T reduced error to 0.03.",
                    ],
                    key_sentence="C achieved 92% accuracy.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="synthetic-methods-section-chunk-0",
                    paper_id="synthetic-registry-paper",
                    section_id="synthetic-methods-section",
                    ordinal=0,
                    text="We used Raman spectroscopy and protein chromatography to profile samples.",
                    chunk_type="methods",
                    word_count=10,
                    token_count=10,
                    salience_score=0.2,
                ),
                ChunkRecord(
                    chunk_id="synthetic-results-section-chunk-0",
                    paper_id="synthetic-registry-paper",
                    section_id="synthetic-results-section",
                    ordinal=0,
                    text="C achieved 92% accuracy. T reduced error to 0.03.",
                    chunk_type="results",
                    word_count=9,
                    token_count=9,
                    salience_score=0.2,
                )
            ],
        )

        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=self._StubEmbedder()):
            extraction = extract_layer2(paper, use_gemini=False)

        dataset_labels = {entity.label for entity in extraction.entities if entity.entity_type == "dataset"}
        self.assertIn("Control group (C)", dataset_labels)
        self.assertIn("Treated group (T)", dataset_labels)

        dataset_entities = {entity.label: entity for entity in extraction.entities if entity.entity_type == "dataset"}
        self.assertIn("C", dataset_entities["Control group (C)"].aliases)
        self.assertIn("T", dataset_entities["Treated group (T)"].aliases)

        method_labels = {entity.label for entity in extraction.entities if entity.entity_type == "method"}
        self.assertIn("Raman spectroscopy", method_labels)
        self.assertIn("protein chromatography", method_labels)

        results = [entity for entity in extraction.entities if entity.entity_type == "result"]
        self.assertTrue(results)
        self.assertTrue(any(entity.properties.get("dataset") == "Control group (C)" for entity in results))
        self.assertTrue(any(entity.properties.get("dataset") == "Treated group (T)" for entity in results))
        self.assertTrue(any(entity.properties.get("value") == 92.0 for entity in results))

    def test_results_are_extracted_without_dataset_labels(self) -> None:
        paper = PaperRecord(
            paper_id="synthetic-paper",
            source_path="synthetic.xml",
            title="Synthetic results paper",
            abstract="Synthetic abstract.",
            sections=[
                SectionRecord(
                    section_id="synthetic-section",
                    paper_id="synthetic-paper",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="The model achieved 92% accuracy and reduced error to 0.18. The p value = 0.03 was significant.",
                    paragraphs=[
                        "The model achieved 92% accuracy and reduced error to 0.18.",
                        "The p value = 0.03 was significant.",
                    ],
                    key_sentence="The model achieved 92% accuracy and reduced error to 0.18.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="synthetic-section-chunk-0",
                    paper_id="synthetic-paper",
                    section_id="synthetic-section",
                    ordinal=0,
                    text="The model achieved 92% accuracy and reduced error to 0.18. The p value = 0.03 was significant.",
                    chunk_type="results",
                    word_count=16,
                    token_count=16,
                    salience_score=0.2,
                )
            ],
        )

        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=self._StubEmbedder()):
            extraction = extract_layer2(paper, use_gemini=False)
        results = [entity for entity in extraction.entities if entity.entity_type == "result"]

        self.assertGreaterEqual(len(results), 2)
        self.assertTrue(any(entity.properties.get("value") == 92.0 for entity in results))
        self.assertTrue(any(not entity.properties.get("datasets") for entity in results))

    def test_generic_concept_patterns_are_detected(self) -> None:
        paper = PaperRecord(
            paper_id="synthetic-concept-paper",
            source_path="synthetic-concept.xml",
            title="Synthetic concept paper",
            abstract="Synthetic abstract.",
            sections=[
                SectionRecord(
                    section_id="synthetic-concept-section",
                    paper_id="synthetic-concept-paper",
                    title="Discussion",
                    section_type="discussion",
                    level=1,
                    ordinal=0,
                    text="The Wnt signaling pathway activates a stress response cascade in these cells.",
                    paragraphs=["The Wnt signaling pathway activates a stress response cascade in these cells."],
                    key_sentence="The Wnt signaling pathway activates a stress response cascade in these cells.",
                )
            ],
            chunks=[
                ChunkRecord(
                    chunk_id="synthetic-concept-section-chunk-0",
                    paper_id="synthetic-concept-paper",
                    section_id="synthetic-concept-section",
                    ordinal=0,
                    text="The Wnt signaling pathway activates a stress response cascade in these cells.",
                    chunk_type="discussion",
                    word_count=12,
                    token_count=12,
                    salience_score=0.2,
                )
            ],
        )

        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=self._StubEmbedder()):
            extraction = extract_layer2(paper, use_gemini=False)

        concept_labels = {entity.label for entity in extraction.entities if entity.entity_type == "concept"}
        self.assertTrue(any("wnt" in label.lower() for label in concept_labels))
        self.assertTrue(any("stress response" in label.lower() for label in concept_labels))


if __name__ == "__main__":
    unittest.main()
