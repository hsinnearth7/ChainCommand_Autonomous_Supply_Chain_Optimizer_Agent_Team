"""Observability + cost control for v2.0 (Section 4.5)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .state import DecisionEntry
from .utils.logging_config import get_logger

log = get_logger(__name__)


class AgentTracer:
    """Traces agent decisions for observability and debugging."""

    def __init__(self) -> None:
        self._decisions: List[DecisionEntry] = []

    def trace_decision(
        self,
        agent: str,
        decision: str,
        rationale: str = "",
        confidence: float = 0.0,
        tokens_used: int = 0,
    ) -> DecisionEntry:
        entry = DecisionEntry(
            agent=agent,
            decision=decision,
            rationale=rationale,
            confidence=confidence,
            tokens_used=tokens_used,
        )
        self._decisions.append(entry)
        log.debug("decision_traced", agent=agent, decision=decision[:80])
        return entry

    def get_decisions(self, agent: Optional[str] = None) -> List[DecisionEntry]:
        if agent is None:
            return list(self._decisions)
        return [d for d in self._decisions if d.agent == agent]

    def summary(self) -> Dict[str, Any]:
        total_tokens = sum(d.tokens_used for d in self._decisions)
        agents = set(d.agent for d in self._decisions)
        per_agent = {
            a: sum(d.tokens_used for d in self._decisions if d.agent == a)
            for a in agents
        }
        return {
            "total_decisions": len(self._decisions),
            "total_tokens": total_tokens,
            "agents": len(agents),
            "per_agent_tokens": per_agent,
        }

    def clear(self) -> None:
        self._decisions.clear()


class TokenBudget:
    """Per-cycle and per-agent token budget enforcement."""

    def __init__(self, per_cycle: int = 50_000, per_agent: int = 8_000) -> None:
        self._per_cycle = per_cycle
        self._per_agent = per_agent
        self._cycle_used: int = 0
        self._agent_used: Dict[str, int] = {}

    def consume(self, agent: str, tokens: int) -> bool:
        """Consume tokens. Returns True if within budget, False if exceeded."""
        self._cycle_used += tokens
        self._agent_used[agent] = self._agent_used.get(agent, 0) + tokens

        over_cycle = self._cycle_used > self._per_cycle
        over_agent = self._agent_used[agent] > self._per_agent

        if over_cycle or over_agent:
            log.warning(
                "token_budget_exceeded",
                agent=agent,
                cycle_used=self._cycle_used,
                cycle_limit=self._per_cycle,
                agent_used=self._agent_used[agent],
                agent_limit=self._per_agent,
            )
            return False
        return True

    def remaining(self, agent: Optional[str] = None) -> int:
        if agent:
            used = self._agent_used.get(agent, 0)
            return max(0, self._per_agent - used)
        return max(0, self._per_cycle - self._cycle_used)

    def reset_cycle(self) -> None:
        self._cycle_used = 0
        self._agent_used.clear()

    @property
    def cycle_used(self) -> int:
        return self._cycle_used

    @property
    def per_cycle(self) -> int:
        return self._per_cycle


class ResponseCache:
    """LRU cache for LLM query responses to reduce redundant calls."""

    def __init__(self, max_size: int = 256) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(prompt: str, system: str = "") -> str:
        raw = f"{system}|||{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, system: str = "") -> Optional[str]:
        key = self._key(prompt, system)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, prompt: str, system: str, response: str) -> None:
        key = self._key(prompt, system)
        self._cache[key] = response
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class TrackedLLM:
    """Wrapper around BaseLLM that tracks token usage via AgentTracer + TokenBudget."""

    def __init__(
        self,
        llm: Any,
        tracer: AgentTracer,
        budget: TokenBudget,
        cache: Optional[ResponseCache] = None,
        agent_name: str = "unknown",
    ) -> None:
        self._llm = llm
        self._tracer = tracer
        self._budget = budget
        self._cache = cache
        self._agent_name = agent_name

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> str:
        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(prompt, system)
            if cached is not None:
                log.debug("llm_cache_hit", agent=self._agent_name)
                return cached

        # Check budget
        estimated_tokens = len(prompt.split()) + len(system.split())
        if not self._budget.consume(self._agent_name, estimated_tokens):
            log.warning("llm_budget_exceeded", agent=self._agent_name)
            return "[Budget exceeded — using degraded response]"

        response = await self._llm.generate(prompt, system=system, temperature=temperature)

        # Estimate response tokens and track
        response_tokens = len(response.split())
        self._budget.consume(self._agent_name, response_tokens)

        self._tracer.trace_decision(
            agent=self._agent_name,
            decision="llm_call",
            rationale=prompt[:200],
            tokens_used=estimated_tokens + response_tokens,
        )

        # Cache the response
        if self._cache is not None:
            self._cache.put(prompt, system, response)

        return response

    async def generate_json(self, prompt: str, schema: Any, system: str = "", temperature: float = 0.1) -> Any:
        estimated_tokens = len(prompt.split()) + len(system.split())
        self._budget.consume(self._agent_name, estimated_tokens)
        result = await self._llm.generate_json(prompt, schema, system=system, temperature=temperature)
        self._tracer.trace_decision(
            agent=self._agent_name,
            decision="llm_json_call",
            rationale=prompt[:200],
            tokens_used=estimated_tokens,
        )
        return result
