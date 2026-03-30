from __future__ import annotations

import unittest

from backend.graphrag.extraction_schema import ExtractionSchema


class Phase3SchemaTests(unittest.TestCase):
    def test_load_general_schema_exposes_core_entities_and_relations(self) -> None:
        schema = ExtractionSchema.load("general")

        self.assertEqual(schema.domain, "general")
        concept = schema.entity_schema("concept")
        self.assertIsNotNone(concept)
        assert concept is not None
        self.assertIn("name", concept.required_fields)
        self.assertIn("pathway", concept.extraction_hints)
        is_a = schema.relation_schema("is_a")
        self.assertIsNotNone(is_a)
        assert is_a is not None
        self.assertIn("concept", is_a.source_types)
        self.assertIn("method_suffixes", schema.hints)

    def test_biomedical_schema_inherits_general_and_extends_hints(self) -> None:
        schema = ExtractionSchema.load("biomedical")

        concept = schema.entity_schema("concept")
        method = schema.entity_schema("method")
        assert concept is not None
        assert method is not None
        self.assertIn("pathway", concept.extraction_hints)
        self.assertIn("gene", concept.extraction_hints)
        self.assertIn("seq", schema.get_hint_list("method_suffixes"))
        self.assertIn("biomedical_entity_markers", schema.hints)
        result_validation = schema.get_validation_metadata("result")
        self.assertTrue(result_validation["require_metric"])
        self.assertTrue(result_validation["require_unit_if_percentage"])

    def test_physics_schema_adds_domain_specific_quantity_hints(self) -> None:
        schema = ExtractionSchema.load("physics")

        quantity_patterns = schema.get_hints("quantity_patterns")
        self.assertIsInstance(quantity_patterns, dict)
        assert isinstance(quantity_patterns, dict)
        self.assertIn("percentage", quantity_patterns)
        self.assertIn("units", quantity_patterns)
        self.assertIn("eV", quantity_patterns["units"])
        result = schema.entity_schema("result")
        assert result is not None
        self.assertIn("resistivity", result.extraction_hints)
        self.assertTrue(schema.get_validation_metadata("result")["require_unit"])

    def test_missing_schema_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ExtractionSchema.load("astronomy")


if __name__ == "__main__":
    unittest.main()
