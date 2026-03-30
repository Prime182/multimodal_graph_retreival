from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.graphrag.config import Phase1Settings
from backend.graphrag.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerOpenError
from backend.graphrag.corpus import CorpusClient
from backend.graphrag.embeddings import GeminiEmbedder, HashingEmbedder
from backend.graphrag.edges import Layer3CorpusRecord
from backend.graphrag.extraction import extract_layer2
from backend.graphrag.ingestion_status import IngestionStatusStore
from backend.graphrag.models import ChunkRecord, PaperRecord, SectionRecord
from backend.graphrag.server import create_app


class Week4ObservabilityTests(unittest.TestCase):
    def test_gemini_embedder_disables_remote_after_first_failure(self) -> None:
        with patch("backend.graphrag.embeddings.gemini_available", return_value=True), patch(
            "backend.graphrag.embeddings.embed_text",
            side_effect=RuntimeError("network blocked"),
        ) as embed_text:
            embedder = GeminiEmbedder(dim=16, fallback=HashingEmbedder(dim=16))
            first = embedder.embed("WTAP")
            second = embedder.embed("METTL3")

        self.assertEqual(len(first), 16)
        self.assertEqual(len(second), 16)
        self.assertEqual(embed_text.call_count, 1)

    def test_circuit_breaker_opens_after_threshold(self) -> None:
        breaker = CircuitBreakerRegistry(failure_threshold=2, cooldown_seconds=60)
        breaker.record_failure("ols")
        breaker.record_failure("ols")

        self.assertTrue(breaker.is_open("ols"))
        with self.assertRaises(CircuitBreakerOpenError):
            breaker.guard("ols")

    def test_corpus_client_records_cached_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = CorpusClient(cache_path=Path(temp_dir) / "corpus.sqlite3")
            with patch("backend.graphrag.corpus.lookup_ncbi_gene", return_value=None):
                first = client.enrich_entity("UNKNOWN1", "gene")
                second = client.enrich_entity("UNKNOWN1", "gene")

            self.assertFalse(first.found)
            self.assertFalse(second.found)
            misses = client.list_misses(limit=10)
            self.assertEqual(misses[0]["label"], "UNKNOWN1")
            self.assertEqual(misses[0]["miss_count"], 2)

    def test_extract_layer2_local_mode_avoids_remote_embedder(self) -> None:
        paper = PaperRecord(paper_id="paper-1", source_path="paper.xml", title="Paper")
        section = SectionRecord(
            section_id="sec-1",
            paper_id="paper-1",
            title="Results",
            section_type="results",
            level=1,
            ordinal=0,
            text="WTAP knockdown reduced HIF-1alpha expression by 32%.",
        )
        chunk = ChunkRecord(
            chunk_id="chunk-1",
            paper_id="paper-1",
            section_id="sec-1",
            ordinal=0,
            text="WTAP knockdown reduced HIF-1alpha expression by 32%.",
            chunk_type="results",
            word_count=8,
            token_count=8,
            salience_score=0.0,
        )
        paper.sections.append(section)
        paper.chunks.append(chunk)
        settings = Phase1Settings(embedding_dim=32)

        with patch("backend.graphrag.extraction.build_entity_embedder", return_value=HashingEmbedder(dim=32)) as builder:
            result = extract_layer2(paper, settings=settings, use_gemini=False)

        self.assertGreater(len(result.entities), 0)
        builder.assert_called_once_with(dim=32, prefer_remote=False)

    def test_ingestion_status_store_reports_extraction_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = IngestionStatusStore(db_path=Path(temp_dir) / "ingestion.sqlite3")
            store.upsert_status(
                paper_id="paper-1",
                paper_title="Paper 1",
                source_path="paper1.xml",
                status="complete",
                extraction_quality=0.82,
                avg_confidence=0.9,
                entity_count=12,
                chunk_count=4,
            )
            report = store.extraction_quality_report(limit=10)

            self.assertEqual(report["summary"]["complete_papers"], 1)
            self.assertEqual(report["summary"]["avg_extraction_quality"], 0.82)
            self.assertEqual(report["papers"][0]["paper_id"], "paper-1")

    def test_observability_endpoints_are_registered(self) -> None:
        class DummySearchService:
            def __init__(self, *args, **kwargs) -> None:
                self.papers = []
                self.layer2_docs = []
                self.layer3 = Layer3CorpusRecord()

            def corpus_misses(self, limit: int = 100):
                return [{"label": "UNKNOWN1", "entity_type": "gene", "preferred_corpus": "", "miss_count": 3, "last_seen": 1.0}]

            def extraction_quality_report(self, limit: int = 100):
                return {
                    "papers": [{"paper_id": "paper-1", "status": "complete", "extraction_quality": 0.8}],
                    "summary": {"paper_count": 1, "complete_papers": 1, "failed_papers": 0, "avg_extraction_quality": 0.8},
                }

            def ingestion_status_report(self, limit: int = 100):
                return [{"paper_id": "paper-1", "status": "complete"}]

        class DummyGraphRetrieval:
            def __init__(self, *args, **kwargs) -> None:
                self.closed = False

            def get_entity_neighborhood(self, entity_ids, hops: int = 2):
                return {"nodes": [{"node_id": entity_ids[0], "label": "Entity"}], "edges": [], "seed_node_ids": list(entity_ids)}

            def get_chunks_mentioning_entities(self, entity_ids):
                return [f"chunk-for-{entity_id}" for entity_id in entity_ids]

            def close(self) -> None:
                self.closed = True

        with patch("backend.graphrag.server.GraphRAGSearchService", DummySearchService), patch(
            "backend.graphrag.server.GraphRetrieval",
            DummyGraphRetrieval,
        ), patch(
            "backend.graphrag.server.probe_embedding_backends",
            return_value={"active_backend": "stub"},
        ):
            app = create_app(input_dir="articles", use_gemini=False)

        paths = {route.path for route in app.routes}
        self.assertIn("/api/corpus-misses", paths)
        self.assertIn("/api/extraction-quality", paths)
        self.assertIn("/api/ingestion-status", paths)
        self.assertIn("/api/circuit-breakers", paths)
        self.assertIn("/api/tracing/health", paths)
        self.assertIn("/api/graph/entity/{entity_id}", paths)

    def test_graph_and_tracing_endpoints_return_payloads(self) -> None:
        class DummySearchService:
            def __init__(self, *args, **kwargs) -> None:
                self.papers = []
                self.layer2_docs = []
                self.layer3 = Layer3CorpusRecord()

            def corpus_misses(self, limit: int = 100):
                return []

            def extraction_quality_report(self, limit: int = 100):
                return {"papers": [], "summary": {}}

            def ingestion_status_report(self, limit: int = 100):
                return []

        class DummyGraphRetrieval:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_entity_neighborhood(self, entity_ids, hops: int = 2):
                return {
                    "nodes": [{"node_id": entity_ids[0], "node_type": "Concept", "label": "WTAP", "properties": {}, "distance": 0}],
                    "edges": [],
                    "seed_node_ids": list(entity_ids),
                }

            def get_chunks_mentioning_entities(self, entity_ids):
                return ["chunk-1"]

            def close(self) -> None:
                return None

        class DummyTracingManager:
            enabled = False
            langfuse = None

        with patch("backend.graphrag.server.GraphRAGSearchService", DummySearchService), patch(
            "backend.graphrag.server.GraphRetrieval",
            DummyGraphRetrieval,
        ), patch(
            "backend.graphrag.server.probe_embedding_backends",
            return_value={"active_backend": "stub"},
        ), patch(
            "backend.graphrag.server.get_tracing_manager",
            return_value=DummyTracingManager(),
        ):
            app = create_app(input_dir="articles", use_gemini=False)

        tracing_endpoint = next(route.endpoint for route in app.routes if route.path == "/api/tracing/health")
        graph_endpoint = next(route.endpoint for route in app.routes if route.path == "/api/graph/entity/{entity_id}")

        tracing = asyncio.run(tracing_endpoint())
        graph = asyncio.run(graph_endpoint("entity-1", hops=1))

        self.assertIn(tracing.status, {"ok", "disabled"})
        self.assertEqual(graph.entity_id, "entity-1")
        self.assertEqual(graph.mentioning_chunks, ["chunk-1"])


if __name__ == "__main__":
    unittest.main()
