"""Resilience and fault tolerance for v2.0 (Section 4.7)."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Coroutine

from .utils.logging_config import get_logger

log = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for LLM / external service calls.

    States:
        closed   → normal operation, failures counted
        open     → all calls fail-fast, wait for recovery_timeout
        half_open → one test call allowed; success → closed, failure → open
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                log.info("circuit_half_open", breaker=self._name)
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def call(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute func through the circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit breaker '{self._name}' is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            log.info("circuit_closed", breaker=self._name, reason="half_open_success")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count += 1

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            log.warning(
                "circuit_opened",
                breaker=self._name,
                failures=self._failure_count,
                recovery_s=self._recovery_timeout,
            )
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            log.warning("circuit_reopened", breaker=self._name)

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and the call is rejected."""


class DegradationLevel(str, Enum):
    FULL = "full"
    LLM_PARTIAL = "llm_partial"
    RULE_BASED = "rule_based"
    HUMAN = "human"


class GracefulDegradation:
    """Manages 4 levels of graceful degradation.

    Levels:
        full        → all systems operational
        llm_partial → LLM calls limited (shorter prompts, fewer agents)
        rule_based  → no LLM, use heuristic rules
        human       → escalate to human operator
    """

    _LEVELS = [
        DegradationLevel.FULL,
        DegradationLevel.LLM_PARTIAL,
        DegradationLevel.RULE_BASED,
        DegradationLevel.HUMAN,
    ]

    def __init__(self) -> None:
        self._level = DegradationLevel.FULL
        self._history: list[tuple[float, DegradationLevel]] = []

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def degrade(self) -> DegradationLevel:
        """Move one level down in degradation hierarchy."""
        idx = self._LEVELS.index(self._level)
        if idx < len(self._LEVELS) - 1:
            self._level = self._LEVELS[idx + 1]
            self._history.append((time.monotonic(), self._level))
            log.warning("degradation_level_changed", new_level=self._level.value)
        return self._level

    def recover(self) -> DegradationLevel:
        """Move one level up in degradation hierarchy."""
        idx = self._LEVELS.index(self._level)
        if idx > 0:
            self._level = self._LEVELS[idx - 1]
            self._history.append((time.monotonic(), self._level))
            log.info("degradation_recovered", new_level=self._level.value)
        return self._level

    def set_level(self, level: DegradationLevel) -> None:
        self._level = level
        self._history.append((time.monotonic(), self._level))

    @property
    def is_degraded(self) -> bool:
        return self._level != DegradationLevel.FULL

    @property
    def history(self) -> list[tuple[float, DegradationLevel]]:
        return list(self._history)
