"""Chaos / fault injection tests — verifying exception survival."""

from __future__ import annotations

import pytest

from chaincommand.resilience import (
    CircuitBreaker,
    CircuitState,
    DegradationLevel,
    GracefulDegradation,
)


class TestExceptionSurvival:
    async def test_circuit_breaker_survives_value_error(self):
        cb = CircuitBreaker(failure_threshold=5)

        async def flaky():
            raise ValueError("transient")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(flaky)
        # Should still be closed (threshold=5, only 3 failures)
        assert cb.state == CircuitState.CLOSED

    async def test_circuit_breaker_survives_type_error(self):
        cb = CircuitBreaker(failure_threshold=2)

        async def bad():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            await cb.call(bad)
        assert cb.failure_count == 1

    async def test_mixed_success_failure(self):
        cb = CircuitBreaker(failure_threshold=3)
        call_count = 0

        async def intermittent():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("intermittent")
            return "ok"

        # Call 1: success
        assert await cb.call(intermittent) == "ok"
        # Call 2: failure
        with pytest.raises(RuntimeError):
            await cb.call(intermittent)
        # Call 3: success (resets failure count)
        assert await cb.call(intermittent) == "ok"
        assert cb.state == CircuitState.CLOSED


class TestDegradationUnderFault:
    def test_degrade_on_circuit_open(self):
        gd = GracefulDegradation()
        # Simulate circuit opening
        gd.degrade()
        assert gd.level == DegradationLevel.LLM_PARTIAL
        gd.degrade()
        assert gd.level == DegradationLevel.RULE_BASED

    def test_recover_after_fix(self):
        gd = GracefulDegradation()
        gd.set_level(DegradationLevel.RULE_BASED)
        gd.recover()
        assert gd.level == DegradationLevel.LLM_PARTIAL
        gd.recover()
        assert gd.level == DegradationLevel.FULL
