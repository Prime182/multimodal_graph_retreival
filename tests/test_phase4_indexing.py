from __future__ import annotations

import unittest

from backend.graphrag.graph_store import _vector_index_statements
from backend.graphrag.indexing import PAGERANK_STEP_FALLBACK_QUERY, PAGERANK_WRITE_QUERY, PROPERTY_INDEX_STATEMENTS


class Phase4IndexingTests(unittest.TestCase):
    def test_property_indexes_cover_graph_metadata(self) -> None:
        joined = "\n".join(PROPERTY_INDEX_STATEMENTS)
        self.assertIn("paper_year", joined)
        self.assertIn("chunk_salience", joined)
        self.assertIn("result_dataset", joined)

    def test_pagerank_query_writes_score(self) -> None:
        self.assertIn("pagerank", PAGERANK_WRITE_QUERY)

    def test_fallback_pagerank_query_writes_score(self) -> None:
        self.assertIn("p.__pagerank_next", PAGERANK_STEP_FALLBACK_QUERY)

    def test_vector_index_statements_close_options_map(self) -> None:
        for statement in _vector_index_statements(256):
            self.assertTrue(statement.endswith("'}}"))


if __name__ == "__main__":
    unittest.main()
