from __future__ import annotations

import unittest

from backend.graphrag.retrieval import LocalVectorIndex


class Phase1RetrievalTests(unittest.TestCase):
    def test_local_index_prefers_the_relevant_article(self) -> None:
        index = LocalVectorIndex.from_xml_paths(
            [
                "articles/ESR-102001.xml",
                "articles/BJ_100828.xml",
            ]
        )

        hits = index.search("district heating neighborhood decision model", top_k=3)

        self.assertEqual(len(hits), 3)
        self.assertIn("heat-transition voting", hits[0].paper_title.lower())
        self.assertTrue(all(hit.score >= hits[-1].score for hit in hits[:1]))

    def test_paper_search_uses_abstract_embeddings(self) -> None:
        index = LocalVectorIndex.from_xml_paths(
            [
                "articles/ESR-102001.xml",
                "articles/BJ_100828.xml",
            ]
        )

        papers = index.search_papers("host parasite leukocytes epigenetic methylation", top_k=2)

        self.assertEqual(len(papers), 2)
        self.assertIn("theileria", papers[0][1].title.lower())
        self.assertGreaterEqual(papers[0][0], papers[1][0])


if __name__ == "__main__":
    unittest.main()
