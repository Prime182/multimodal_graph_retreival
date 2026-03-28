"""Simple in-memory circuit breakers for external API integrations."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open for a service."""


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    last_failure: float = 0.0


class CircuitBreakerRegistry:
    """Tracks failure state for external services."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 300) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._states: dict[str, _CircuitState] = {}

    def _state(self, service: str) -> _CircuitState:
        return self._states.setdefault(service, _CircuitState())

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

    def record_success(self, service: str) -> None:
        state = self._state(service)
        state.failures = 0
        state.last_failure = 0.0

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
