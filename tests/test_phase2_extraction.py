from __future__ import annotations

import unittest
from pathlib import Path

from backend.graphrag import chunk_article, extract_layer2, parse_article


def _article_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"No test article found in candidates: {candidates}")


class Phase2ExtractionTests(unittest.TestCase):
    def test_extracts_layer2_entities_from_esr_article(self) -> None:
        article_path = _article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml")
        article = chunk_article(parse_article(article_path))
        initial_salience = {chunk.chunk_id: chunk.salience_score for chunk in article.chunks}

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
        extraction = extract_layer2(article)

        concept_labels = [entity.label.lower() for entity in extraction.entities if entity.entity_type == "concept"]
        self.assertTrue(any("theileria" in label for label in concept_labels))
        self.assertTrue(any("m6a" in label or "hif1" in label for label in concept_labels))
        self.assertTrue(extraction.chunk_entity_ids)

    def test_biomedical_datasets_results_and_aliases_are_specific(self) -> None:
        article = chunk_article(parse_article(_article_path("articles/BJ_100828.xml", "articles/ESR-102001.xml")))
        extraction = extract_layer2(article)

        dataset_labels = {entity.label for entity in extraction.entities if entity.entity_type == "dataset"}
        self.assertTrue(
            {
                "virulent macrophages (V)",
                "attenuated macrophages (A)",
                "merozoite-producing macrophages (Vm)",
                "infected B cells (TBL20)",
                "uninfected B cells (BL20)",
                "Ode macrophages",
            }.issubset(dataset_labels)
        )

        results = [entity for entity in extraction.entities if entity.entity_type == "result"]
        self.assertTrue(results)
        self.assertFalse(
            any(entity.properties.get("dataset") in {"Experimental dataset", "Cell-based study"} for entity in results)
        )
        self.assertFalse(any(entity.properties.get("metric") == "Dissociation Constant (Ki)" for entity in results))
        self.assertFalse(
            any(entity.properties.get("metric") == "Optical Density" and entity.properties.get("value", 0) > 5 for entity in results)
        )
        self.assertTrue(
            any("virulent macrophages (V)" in entity.properties.get("datasets", []) for entity in results)
        )

        methods = {entity.label: entity for entity in extraction.entities if entity.entity_type == "method"}
        self.assertIn("CRISPR", methods)
        self.assertIn("CRISPRa", methods)
        self.assertIn("DESeq2", methods)
        self.assertEqual(methods["CRISPR"].properties.get("method_type"), "experimental")
        self.assertEqual(methods["DESeq2"].properties.get("method_type"), "statistical")
        self.assertNotIn("4C", methods)

        concepts = {entity.label: entity for entity in extraction.entities if entity.entity_type == "concept"}
        self.assertIn("WTAP", concepts)
        self.assertIn("HIF-1α", concepts)
        self.assertTrue(any("Wilms" in alias for alias in concepts["WTAP"].aliases))


if __name__ == "__main__":
    unittest.main()
