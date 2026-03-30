"""SQLite-backed circuit breakers for external API integrations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import time


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open for a service."""


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    last_failure: float = 0.0


class CircuitBreakerRegistry:
    """Tracks and persists failure state for external services."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: int = 300,
        state_db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state_db_path = Path(
            state_db_path
            or os.getenv("API_CIRCUIT_BREAKER_DB_PATH")
            or ".cache/graphrag_circuit_breaker.sqlite3"
        )
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.state_db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                service TEXT PRIMARY KEY,
                failures INTEGER NOT NULL,
                last_failure REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._states: dict[str, _CircuitState] = {}
        self._load_persisted_state()

    def _state(self, service: str) -> _CircuitState:
        state = self._states.get(service)
        if state is None:
            state = _CircuitState()
            self._states[service] = state
        return state

    def _load_persisted_state(self) -> None:
        cursor = self._conn.execute(
            "SELECT service, failures, last_failure FROM circuit_breaker_state"
        )
        for service, failures, last_failure in cursor.fetchall():
            self._states[str(service)] = _CircuitState(
                failures=int(failures),
                last_failure=float(last_failure),
            )

    def _persist_state(self, service: str) -> None:
        state = self._state(service)
        self._conn.execute(
            """
            INSERT INTO circuit_breaker_state (service, failures, last_failure, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(service) DO UPDATE SET
                failures = excluded.failures,
                last_failure = excluded.last_failure,
                updated_at = excluded.updated_at
            """,
            (service, state.failures, state.last_failure, time.time()),
        )
        self._conn.commit()

    def is_open(self, service: str) -> bool:
        state = self._state(service)
        if state.failures < self.failure_threshold:
            return False
        if time.time() - state.last_failure > self.cooldown_seconds:
            self.record_success(service)
            return False
        return True

    def guard(self, service: str) -> None:
        if self.is_open(service):
            raise CircuitBreakerOpenError(f"Circuit breaker open for service '{service}'")

    def record_failure(self, service: str) -> None:
        state = self._state(service)
        state.failures += 1
        state.last_failure = time.time()
        self._persist_state(service)

    def record_success(self, service: str) -> None:
        state = self._state(service)
        state.failures = 0
        state.last_failure = 0.0
        self._persist_state(service)

    def snapshot(self) -> list[dict[str, object]]:
        now = time.time()
        snapshot: list[dict[str, object]] = []
        for service, state in sorted(self._states.items()):
            snapshot.append(
                {
                    "service": service,
                    "failures": state.failures,
                    "open": self.is_open(service),
                    "last_failure": state.last_failure or None,
                    "seconds_since_last_failure": round(now - state.last_failure, 3) if state.last_failure else None,
                }
            )
        return snapshot

    def reset(self, service: str | None = None) -> None:
        """Reset failure counts. Pass service=None to reset all."""
        if service is None:
            self._states.clear()
            self._conn.execute("DELETE FROM circuit_breaker_state")
            self._conn.commit()
        elif service in self._states:
            del self._states[service]
            self._conn.execute("DELETE FROM circuit_breaker_state WHERE service = ?", (service,))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_default_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker() -> CircuitBreakerRegistry:
    global _default_registry
    if _default_registry is None:
        threshold = int(os.getenv("API_CIRCUIT_BREAKER_THRESHOLD", "2"))
        cooldown = int(os.getenv("API_CIRCUIT_BREAKER_COOLDOWN_SEC", "300"))
        _default_registry = CircuitBreakerRegistry(
            failure_threshold=threshold,
            cooldown_seconds=cooldown,
        )
    return _default_registry
