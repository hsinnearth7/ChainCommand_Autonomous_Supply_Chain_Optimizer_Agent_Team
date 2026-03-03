"""Tests for DemandForecasterAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincommand.agents.demand_forecaster import DemandForecasterAgent
from chaincommand.data.schemas import AlertSeverity, SupplyChainEvent


@pytest.fixture
def forecaster_agent(mock_llm):
    from chaincommand.tools import (
        EmitEvent,
        GetForecastAccuracy,
        QueryDemandHistory,
        RunDemandForecast,
        ScanMarketIntelligence,
    )

    return DemandForecasterAgent(
        llm=mock_llm,
        tools=[
            QueryDemandHistory(),
            RunDemandForecast(),
            GetForecastAccuracy(),
            ScanMarketIntelligence(),
            EmitEvent(),
        ],
    )


class TestDemandForecasterAgent:
    def test_agent_attributes(self, forecaster_agent):
        assert forecaster_agent.name == "demand_forecaster"
        assert forecaster_agent.layer == "strategic"

    async def test_run_cycle_returns_dict(self, forecaster_agent, sample_products):
        ctx = {"products": sample_products, "cycle": 1}
        with patch.object(forecaster_agent, "think", new_callable=AsyncMock, return_value="Analysis complete"):
            result = await forecaster_agent.run_cycle(ctx)
        assert isinstance(result, dict)
        assert result["agent"] == "demand_forecaster"

    async def test_handle_event_kpi_violation(self, forecaster_agent):
        event = SupplyChainEvent(
            event_type="kpi_threshold_violated",
            severity=AlertSeverity.HIGH,
            source_agent="kpi_engine",
            data={"metric": "mape", "value": 20.0},
        )
        await forecaster_agent.handle_event(event)  # should not raise

    async def test_handle_event_market_intel(self, forecaster_agent):
        event = SupplyChainEvent(
            event_type="new_market_intel",
            severity=AlertSeverity.LOW,
            source_agent="market_intelligence",
            data={"topic": "price_change"},
        )
        await forecaster_agent.handle_event(event)  # should not raise

    async def test_cycle_count_increments(self, forecaster_agent, sample_products):
        assert forecaster_agent._cycle_count == 0
        ctx = {"products": sample_products, "cycle": 1}
        with patch.object(forecaster_agent, "think", new_callable=AsyncMock, return_value="ok"):
            await forecaster_agent.run_cycle(ctx)
        assert forecaster_agent._cycle_count == 1
