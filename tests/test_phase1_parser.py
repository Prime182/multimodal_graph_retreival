from __future__ import annotations

import unittest

from backend.graphrag import chunk_article, parse_article


class Phase1ParserTests(unittest.TestCase):
    def test_parse_article_extracts_layer1_metadata(self) -> None:
        article = parse_article("articles/ESR-102001.xml")

        self.assertEqual(article.doi, "10.1016/j.esr.2025.102001")
        self.assertEqual(article.metadata["source_format"], "elsevier_xml")
        self.assertEqual(article.metadata["figure_count"], 10)
        self.assertGreaterEqual(len(article.authors), 3)
        self.assertEqual(article.sections[0].section_type, "abstract")
        self.assertTrue(article.highlights)
        self.assertNotIn("Highlights", article.highlights)

    def test_chunking_creates_next_links_within_sections(self) -> None:
        article = chunk_article(parse_article("articles/ESR-102001.xml"))

        abstract_chunks = [chunk for chunk in article.chunks if chunk.section_id.endswith("abstract")]
        self.assertGreaterEqual(len(abstract_chunks), 2)
        self.assertIsNone(abstract_chunks[0].prev_chunk_id)
        self.assertEqual(abstract_chunks[0].next_chunk_id, abstract_chunks[1].chunk_id)

        for chunk in article.chunks:
            self.assertGreater(chunk.word_count, 0)
            self.assertGreaterEqual(chunk.salience_score, 0.0)
            self.assertLessEqual(chunk.salience_score, 1.0)

    def test_section_type_heuristics_handle_methods_and_results(self) -> None:
        article = parse_article("articles/BJ_100828.xml")
        section_types = {section.title: section.section_type for section in article.sections}

        self.assertEqual(section_types["Materials and Methods"], "methods")
        self.assertEqual(section_types["Results:"], "results")


if __name__ == "__main__":
    unittest.main()
