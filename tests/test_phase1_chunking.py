from __future__ import annotations

import unittest

from backend.graphrag.chunking import chunk_article
from backend.graphrag.config import Phase1Settings
from backend.graphrag.models import PaperRecord, SectionRecord


class Phase1ChunkingTests(unittest.TestCase):
    def test_chunk_overlap_carries_tail_words_forward(self) -> None:
        article = PaperRecord(
            paper_id="paper-1",
            source_path="paper-1.xml",
            title="Overlap test paper",
            abstract="Short abstract for chunking.",
            sections=[
                SectionRecord(
                    section_id="paper-1-results",
                    paper_id="paper-1",
                    title="Results",
                    section_type="results",
                    level=1,
                    ordinal=0,
                    text="alpha beta gamma delta. epsilon zeta eta theta. iota kappa lambda mu.",
                    paragraphs=[
                        "alpha beta gamma delta. epsilon zeta eta theta. iota kappa lambda mu.",
                    ],
                )
            ],
        )

        settings = Phase1Settings(chunk_size_words=6, chunk_overlap_words=2)
        chunked = chunk_article(article, settings=settings)

        self.assertGreaterEqual(len(chunked.chunks), 3)
        self.assertTrue(chunked.chunks[1].text.startswith("gamma delta"))


if __name__ == "__main__":
    unittest.main()
