from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.graphrag.config import Phase1Settings
from backend.graphrag.graph_retrieval import GraphRetrieval


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query: str, **params: object):
        self.calls.append((query, params))
        if "AS node_id" in query and "distance" in query:
            return [
                {
                    "node_id": "concept-1",
                    "node_type": "Concept",
                    "label": "WTAP",
                    "properties": {"name": "WTAP"},
                    "distance": 0,
                },
                {
                    "node_id": "method-1",
                    "node_type": "Method",
                    "label": "MeRIP-seq",
                    "properties": {"method_type": "experimental"},
                    "distance": 1,
                },
            ]
        if "chunk:Chunk" in query and "MENTIONS" in query:
            return [
                {"chunk_id": "chunk-2", "mention_count": 2, "salience": 0.8},
                {"chunk_id": "chunk-1", "mention_count": 1, "salience": 0.2},
            ]
        if "AS edge_id" in query:
            return [
                {
                    "edge_id": "edge-1",
                    "relation_type": "MENTIONS",
                    "source_node_id": "concept-1",
                    "source_node_type": "Concept",
                    "source_label": "WTAP",
                    "target_node_id": "method-1",
                    "target_node_type": "Method",
                    "target_label": "MeRIP-seq",
                    "confidence": 0.92,
                    "properties": {"paper_id": "paper-1"},
                }
            ]
        raise AssertionError(f"Unexpected query: {query}")


class _FakeDriver:
    def __init__(self) -> None:
        self.session_obj = _FakeSession()
        self.closed = False
        self.calls: list[tuple[object, object]] = []

    def session(self, database: str | None = None) -> _FakeSession:
        self.calls.append((database, None))
        return self.session_obj

    def close(self) -> None:
        self.closed = True


class GraphRetrievalTests(unittest.TestCase):
    def test_get_entity_neighborhood_returns_nodes_and_edges(self) -> None:
        fake_driver = _FakeDriver()
        with patch("backend.graphrag.graph_retrieval.GraphDatabase.driver", return_value=fake_driver):
            retriever = GraphRetrieval(
                settings=Phase1Settings(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                )
            )

        neighborhood = retriever.get_entity_neighborhood(["concept-1", "method-1"], hops=2)

        self.assertEqual(neighborhood["seed_node_ids"], ["concept-1", "method-1"])
        self.assertEqual([node["node_id"] for node in neighborhood["nodes"]], ["concept-1", "method-1"])
        self.assertEqual([edge["edge_id"] for edge in neighborhood["edges"]], ["edge-1"])
        self.assertIn("[*0..2]", fake_driver.session_obj.calls[0][0])
        self.assertIn("AS node_id", fake_driver.session_obj.calls[0][0])
        self.assertIn("AS edge_id", fake_driver.session_obj.calls[1][0])

    def test_get_chunks_mentioning_entities_returns_chunk_ids(self) -> None:
        fake_driver = _FakeDriver()
        with patch("backend.graphrag.graph_retrieval.GraphDatabase.driver", return_value=fake_driver):
            retriever = GraphRetrieval(
                settings=Phase1Settings(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                )
            )

        chunk_ids = retriever.get_chunks_mentioning_entities(["concept-1", "method-1"])

        self.assertEqual(chunk_ids, ["chunk-2", "chunk-1"])
        self.assertIn("MATCH (chunk:Chunk)-[:MENTIONS]->(entity)", fake_driver.session_obj.calls[0][0])
        self.assertIn("entity.id IN $entity_ids", fake_driver.session_obj.calls[0][0])

    def test_empty_entity_id_list_short_circuits(self) -> None:
        fake_driver = _FakeDriver()
        with patch("backend.graphrag.graph_retrieval.GraphDatabase.driver", return_value=fake_driver):
            retriever = GraphRetrieval(
                settings=Phase1Settings(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                )
            )

        self.assertEqual(retriever.get_entity_neighborhood([], hops=2), {"nodes": [], "edges": [], "seed_node_ids": []})
        self.assertEqual(retriever.get_chunks_mentioning_entities([]), [])
        self.assertEqual(fake_driver.session_obj.calls, [])


if __name__ == "__main__":
    unittest.main()
