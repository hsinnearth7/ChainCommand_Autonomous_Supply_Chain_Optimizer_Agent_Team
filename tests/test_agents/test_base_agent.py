"""Tests for BaseAgent."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from chaincommand.agents.base_agent import BaseAgent
from chaincommand.data.schemas import AgentAction, SupplyChainEvent


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing the abstract base class."""

    name = "test_agent"
    role = "Test agent"
    layer = "operational"
    state_key = "test_results"

    async def handle_event(self, event: SupplyChainEvent) -> None:
        pass

    async def run_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent": self.name, "status": "ok"}


@pytest.fixture
def agent(mock_llm):
    return ConcreteAgent(llm=mock_llm)


class TestBaseAgentThink:
    async def test_think_returns_string(self, agent):
        result = await agent.think({"products": [], "cycle": 1})
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_think_with_context_keys(self, agent):
        ctx = {"demand": 100, "stock": 500, "trend": "increasing"}
        result = await agent.think(ctx)
        assert isinstance(result, str)


class TestBaseAgentAct:
    async def test_act_tool_not_found(self, agent):
        action = AgentAction(
            agent_name="test_agent",
            action_type="nonexistent_tool",
            description="test",
        )
        result = await agent.act(action)
        assert "error" in result
        assert action.success is False

    async def test_action_log_grows(self, agent):
        for i in range(3):
            action = AgentAction(
                agent_name="test_agent",
                action_type="missing",
                description=f"test {i}",
            )
            await agent.act(action)
        assert len(agent._action_log) == 3


class TestBaseAgentStatus:
    def test_get_status_structure(self, agent):
        status = agent.get_status()
        assert status["name"] == "test_agent"
        assert status["role"] == "Test agent"
        assert status["layer"] == "operational"
        assert status["active"] is True
        assert status["cycle_count"] == 0

    def test_state_key_attribute(self, agent):
        assert agent.state_key == "test_results"
