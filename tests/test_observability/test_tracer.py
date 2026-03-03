"""Tests for observability: tracer, token budget, cache."""

from __future__ import annotations

from chaincommand.observability import AgentTracer, ResponseCache, TokenBudget


class TestAgentTracer:
    def test_trace_decision(self):
        tracer = AgentTracer()
        entry = tracer.trace_decision("agent_a", "reorder", tokens_used=100)
        assert entry.agent == "agent_a"
        assert entry.tokens_used == 100

    def test_get_decisions_filtered(self):
        tracer = AgentTracer()
        tracer.trace_decision("a", "d1")
        tracer.trace_decision("b", "d2")
        tracer.trace_decision("a", "d3")
        assert len(tracer.get_decisions("a")) == 2
        assert len(tracer.get_decisions("b")) == 1
        assert len(tracer.get_decisions()) == 3

    def test_summary(self):
        tracer = AgentTracer()
        tracer.trace_decision("a", "d1", tokens_used=100)
        tracer.trace_decision("b", "d2", tokens_used=200)
        s = tracer.summary()
        assert s["total_decisions"] == 2
        assert s["total_tokens"] == 300
        assert s["agents"] == 2

    def test_clear(self):
        tracer = AgentTracer()
        tracer.trace_decision("a", "d1")
        tracer.clear()
        assert len(tracer.get_decisions()) == 0


class TestTokenBudget:
    def test_consume_within_budget(self):
        budget = TokenBudget(per_cycle=1000, per_agent=500)
        assert budget.consume("agent_a", 100) is True
        assert budget.remaining("agent_a") == 400
        assert budget.remaining() == 900

    def test_consume_exceeds_cycle(self):
        budget = TokenBudget(per_cycle=200, per_agent=500)
        budget.consume("a", 100)
        result = budget.consume("b", 150)
        assert result is False

    def test_consume_exceeds_agent(self):
        budget = TokenBudget(per_cycle=10000, per_agent=100)
        budget.consume("a", 80)
        result = budget.consume("a", 50)
        assert result is False

    def test_reset_cycle(self):
        budget = TokenBudget(per_cycle=1000, per_agent=500)
        budget.consume("a", 300)
        budget.reset_cycle()
        assert budget.remaining() == 1000
        assert budget.remaining("a") == 500


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache(max_size=10)
        cache.put("hello", "sys", "world")
        assert cache.get("hello", "sys") == "world"

    def test_cache_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_hit_rate(self):
        cache = ResponseCache()
        cache.put("a", "", "result_a")
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate() == 0.5

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("a", "", "1")
        cache.put("b", "", "2")
        cache.put("c", "", "3")  # evicts "a"
        assert cache.get("a", "") is None
        assert cache.get("b", "") == "2"
        assert cache.get("c", "") == "3"

    def test_clear(self):
        cache = ResponseCache()
        cache.put("x", "", "y")
        cache.clear()
        assert cache.size == 0
        assert cache.hit_rate() == 0.0
