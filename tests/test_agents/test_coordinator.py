"""Tests for CoordinatorAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincommand.agents.coordinator import CoordinatorAgent
from chaincommand.data.schemas import AlertSeverity, SupplyChainEvent


@pytest.fixture
def coord_agent(mock_llm):
    from chaincommand.tools.action_tools import EmitEvent, RequestHumanApproval
    from chaincommand.tools.data_tools import (
        QueryDemandHistory,
        QueryInventoryStatus,
        QueryKPIHistory,
        QuerySupplierInfo,
    )

    return CoordinatorAgent(
        llm=mock_llm,
        tools=[
            QueryKPIHistory(),
            QueryInventoryStatus(),
            QuerySupplierInfo(),
            QueryDemandHistory(),
            RequestHumanApproval(),
            EmitEvent(),
        ],
    )


class TestCoordinatorAgent:
    def test_agent_attributes(self, coord_agent):
        assert coord_agent.name == "coordinator"
        assert coord_agent.layer == "orchestration"

    async def test_run_cycle_returns_dict(self, coord_agent, sample_products):
        ctx = {"products": sample_products, "cycle": 1, "agent_results": {}}
        with patch.object(coord_agent, "think", new_callable=AsyncMock, return_value="Summary ok"):
            result = await coord_agent.run_cycle(ctx)
        assert isinstance(result, dict)
        assert result["agent"] == "coordinator"
        assert "conflicts_resolved" in result
        assert "priority_actions" in result
        assert "executive_summary" in result

    async def test_conflict_detection(self, coord_agent):
        agent_results = {
            "inventory_optimizer": {"reorders": [{"product_id": "P1"}]},
            "strategic_planner": {"adjustments": [{"product_id": "P1", "action": "reduce"}]},
        }
        conflicts = coord_agent._detect_conflicts(agent_results)
        assert isinstance(conflicts, list)

    async def test_action_prioritization(self, coord_agent):
        actions = [
            {"source": "a", "type": "alerts", "data": {}},
            {"source": "b", "type": "reorders", "data": {}},
            {"source": "c", "type": "mitigations", "data": {}},
        ]
        prioritized = coord_agent._prioritize_actions(actions)
        # reorders (priority 1) should come first
        assert prioritized[0]["type"] == "reorders"

    async def test_handle_event_logs(self, coord_agent):
        event = SupplyChainEvent(
            event_type="test_event",
            severity=AlertSeverity.LOW,
        )
        await coord_agent.handle_event(event)  # should not raise
