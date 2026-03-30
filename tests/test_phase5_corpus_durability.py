from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.graphrag.circuit_breaker import CircuitBreakerOpenError, CircuitBreakerRegistry
from backend.graphrag.corpus import CorpusClient, CorpusMatch


class Phase5CorpusDurabilityTests(unittest.TestCase):
    def test_corpus_cache_expires_entries_and_requeries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = CorpusClient(
                cache_path=Path(temp_dir) / "corpus.sqlite3",
                cache_ttl_seconds=0,
            )
            first_match = CorpusMatch(
                found=True,
                label="WTAP",
                aliases=["Wilms tumor 1 associated protein"],
                ontology="NCBI-Gene",
                external_id="9589",
                category="gene",
            )
            second_match = CorpusMatch(
                found=True,
                label="WTAP",
                aliases=["Wilms tumor 1 associated protein"],
                ontology="NCBI-Gene",
                external_id="9999",
                category="gene",
            )

            with patch(
                "backend.graphrag.corpus.lookup_ncbi_gene",
                side_effect=[first_match, second_match],
            ) as lookup:
                first = client.enrich_entity("WTAP", "gene")
                second = client.enrich_entity("WTAP", "gene")

            self.assertEqual(first.external_id, "9589")
            self.assertEqual(second.external_id, "9999")
            self.assertEqual(lookup.call_count, 2)

    def test_circuit_breaker_state_persists_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "breaker.sqlite3"

            registry = CircuitBreakerRegistry(
                failure_threshold=2,
                cooldown_seconds=60,
                state_db_path=db_path,
            )
            registry.record_failure("ols")
            registry.record_failure("ols")
            self.assertTrue(registry.is_open("ols"))

            reloaded = CircuitBreakerRegistry(
                failure_threshold=2,
                cooldown_seconds=60,
                state_db_path=db_path,
            )
            self.assertTrue(reloaded.is_open("ols"))
            with self.assertRaises(CircuitBreakerOpenError):
                reloaded.guard("ols")

            reloaded.record_success("ols")

            restored = CircuitBreakerRegistry(
                failure_threshold=2,
                cooldown_seconds=60,
                state_db_path=db_path,
            )
            self.assertFalse(restored.is_open("ols"))


if __name__ == "__main__":
    unittest.main()
