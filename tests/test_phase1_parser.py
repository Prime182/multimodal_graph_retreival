from __future__ import annotations

import unittest
from pathlib import Path

from backend.graphrag import chunk_article, parse_article


def _article_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"No test article found in candidates: {candidates}")


class Phase1ParserTests(unittest.TestCase):
    def test_parse_article_extracts_layer1_metadata(self) -> None:
        article = parse_article(_article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml"))

        self.assertTrue(article.doi)
        self.assertEqual(article.metadata["source_format"], "elsevier_xml")
        self.assertGreater(article.metadata["figure_count"], 0)
        self.assertGreaterEqual(len(article.authors), 3)
        self.assertEqual(article.sections[0].section_type, "abstract")
        self.assertTrue(article.highlights)
        self.assertNotIn("Highlights", article.highlights)

    def test_chunking_creates_next_links_within_sections(self) -> None:
        article = chunk_article(parse_article(_article_path("articles/ESR-102001.xml", "articles/BJ_100828.xml")))

        chunks_by_section: dict[str, list] = {}
        for chunk in article.chunks:
            chunks_by_section.setdefault(chunk.section_id, []).append(chunk)

        linked_chunks = next(
            (chunks for chunks in chunks_by_section.values() if len(chunks) >= 2),
            None,
        )
        self.assertIsNotNone(linked_chunks)
        assert linked_chunks is not None
        self.assertIsNone(linked_chunks[0].prev_chunk_id)
        self.assertEqual(linked_chunks[0].next_chunk_id, linked_chunks[1].chunk_id)

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
