"""Tests for circuit breaker and graceful degradation."""

from __future__ import annotations

import pytest

from chaincommand.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    DegradationLevel,
    GracefulDegradation,
)


class TestCircuitBreakerStates:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    async def test_success_keeps_closed(self):
        cb = CircuitBreaker(failure_threshold=3)

        async def ok():
            return "ok"

        result = await cb.call(ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    async def test_failures_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=2)

        async def fail():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

    async def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker(failure_threshold=1)

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(fail)

        with pytest.raises(CircuitOpenError):
            await cb.call(fail)


class TestCircuitBreakerTransitions:
    async def test_open_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(fail)

        # Recovery timeout = 0, should transition to half_open
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)

        async def fail():
            raise ValueError("boom")

        async def ok():
            return "ok"

        with pytest.raises(ValueError):
            await cb.call(fail)

        # Now half_open (recovery_timeout=0)
        result = await cb.call(ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await cb.call(fail)

        # half_open (recovery_timeout=0), another failure should re-open
        with pytest.raises(ValueError):
            await cb.call(fail)
        # With recovery_timeout=0, internal state is OPEN but property
        # immediately transitions to HALF_OPEN. Check internal state.
        assert cb._state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb._failure_count = 5
        cb._state = CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestGracefulDegradation:
    def test_initial_level_full(self):
        gd = GracefulDegradation()
        assert gd.level == DegradationLevel.FULL

    def test_degrade_sequence(self):
        gd = GracefulDegradation()
        assert gd.degrade() == DegradationLevel.LLM_PARTIAL
        assert gd.degrade() == DegradationLevel.RULE_BASED
        assert gd.degrade() == DegradationLevel.HUMAN
        # Can't degrade further
        assert gd.degrade() == DegradationLevel.HUMAN

    def test_recover_sequence(self):
        gd = GracefulDegradation()
        gd.set_level(DegradationLevel.HUMAN)
        assert gd.recover() == DegradationLevel.RULE_BASED
        assert gd.recover() == DegradationLevel.LLM_PARTIAL
        assert gd.recover() == DegradationLevel.FULL
        # Can't recover further
        assert gd.recover() == DegradationLevel.FULL

    def test_is_degraded(self):
        gd = GracefulDegradation()
        assert gd.is_degraded is False
        gd.degrade()
        assert gd.is_degraded is True
