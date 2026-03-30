from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from backend.graphrag.corpus import get_hierarchy, lookup_mesh


class Phase1CorpusTests(unittest.TestCase):
    def test_lookup_mesh_uses_mesh_descriptor_lookup_endpoint(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "resultList": {
                "result": [
                    {
                        "label": "Hypertension",
                        "resource": "http://id.nlm.nih.gov/mesh/D006973",
                        "synonym": ["High Blood Pressure"],
                    }
                ]
            }
        }

        with patch("backend.graphrag.corpus.requests.get", return_value=response) as mock_get:
            match = lookup_mesh("hypertension")

        mock_get.assert_called_once()
        self.assertIn("id.nlm.nih.gov/mesh/lookup", mock_get.call_args.args[0])
        self.assertEqual(mock_get.call_args.kwargs["params"]["form"], "descriptor")
        self.assertEqual(mock_get.call_args.kwargs["params"]["label"], "hypertension")
        self.assertIsNotNone(match)
        self.assertTrue(match and match.found)
        self.assertEqual(match.label, "Hypertension")
        self.assertEqual(match.ontology, "MESH")
        self.assertEqual(match.external_id, "http://id.nlm.nih.gov/mesh/D006973")

    def test_get_hierarchy_uses_ols_parent_lookup_before_yaml_fallback(self) -> None:
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "response": {
                "docs": [
                    {
                        "label": "methylation",
                        "iri": "http://purl.obolibrary.org/obo/GO_0006479",
                    }
                ]
            }
        }

        parent_response = Mock()
        parent_response.status_code = 200
        parent_response.json.return_value = {
            "_embedded": {
                "terms": [
                    {"label": "protein modification"},
                    {"label": "post-translational modification"},
                ]
            }
        }

        with patch(
            "backend.graphrag.corpus.requests.get",
            side_effect=[search_response, parent_response],
        ) as mock_get:
            parents = get_hierarchy("methylation", "go")

        self.assertEqual(parents[:2], ["protein modification", "post-translational modification"])
        self.assertEqual(mock_get.call_count, 2)
        self.assertIn("/search", mock_get.call_args_list[0].args[0])
        self.assertIn("/hierarchicalParents", mock_get.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
