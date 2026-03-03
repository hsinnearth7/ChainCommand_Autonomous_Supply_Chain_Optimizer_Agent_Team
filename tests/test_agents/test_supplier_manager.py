"""Tests for SupplierManagerAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincommand.agents.supplier_manager import SupplierManagerAgent
from chaincommand.data.schemas import AlertSeverity, SupplyChainEvent


@pytest.fixture
def supplier_agent(mock_llm):
    from chaincommand.tools import (
        CreatePurchaseOrder,
        EmitEvent,
        EvaluateSupplier,
        QuerySupplierInfo,
        RequestHumanApproval,
    )

    return SupplierManagerAgent(
        llm=mock_llm,
        tools=[
            QuerySupplierInfo(),
            EvaluateSupplier(),
            CreatePurchaseOrder(),
            RequestHumanApproval(),
            EmitEvent(),
        ],
    )


class TestSupplierManagerAgent:
    def test_agent_attributes(self, supplier_agent):
        assert supplier_agent.name == "supplier_manager"
        assert supplier_agent.layer == "tactical"

    async def test_run_cycle_returns_dict(self, supplier_agent, sample_products):
        ctx = {"products": sample_products, "cycle": 1}
        with patch.object(supplier_agent, "think", new_callable=AsyncMock, return_value="Analysis ok"):
            result = await supplier_agent.run_cycle(ctx)
        assert isinstance(result, dict)
        assert result["agent"] == "supplier_manager"

    async def test_hitl_gate_low_cost_auto_approve(self, supplier_agent, sample_products):
        """Products with low reorder cost should be auto-approved."""
        # Make one product need reorder (stock < reorder_point)
        sample_products[0].current_stock = 50.0  # below reorder_point=100
        sample_products[0].unit_cost = 5.0  # low cost
        ctx = {"products": sample_products, "cycle": 1}
        with patch.object(supplier_agent, "think", new_callable=AsyncMock, return_value="ok"):
            result = await supplier_agent.run_cycle(ctx)
        assert isinstance(result, dict)

    async def test_handle_event_reorder(self, supplier_agent):
        event = SupplyChainEvent(
            event_type="reorder_triggered",
            severity=AlertSeverity.MEDIUM,
            data={"product_id": "PRD-001"},
        )
        await supplier_agent.handle_event(event)

    async def test_handle_event_quality_alert(self, supplier_agent):
        event = SupplyChainEvent(
            event_type="quality_alert",
            severity=AlertSeverity.HIGH,
            data={"supplier_id": "SUP-001"},
        )
        await supplier_agent.handle_event(event)
