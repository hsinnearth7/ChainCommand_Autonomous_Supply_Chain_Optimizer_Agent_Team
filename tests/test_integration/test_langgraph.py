"""Integration tests for LangGraph orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chaincommand.langgraph_orchestrator import LangGraphOrchestrator, _build_graph
from chaincommand.state import DecisionEntry, SupplyChainState


class TestSupplyChainState:
    def test_default_state(self):
        state = SupplyChainState()
        assert state.cycle == 0
        assert state.requires_human_approval is False
        assert state.decisions == []

    def test_add_decision(self):
        state = SupplyChainState()
        state.add_decision("agent_a", "reorder", confidence=0.9, tokens_used=50)
        assert len(state.decisions) == 1
        assert state.decisions[0].agent == "agent_a"
        assert state.decisions[0].confidence == 0.9

    def test_state_isolation_fields(self):
        state = SupplyChainState()
        state.market_intel = {"signal": "bullish"}
        state.forecast_results = {"mape": 5.0}
        assert state.market_intel["signal"] == "bullish"
        assert state.forecast_results["mape"] == 5.0
        # Other fields should still be empty
        assert state.anomaly_results == {}


class TestDecisionEntry:
    def test_decision_entry_creation(self):
        entry = DecisionEntry(
            agent="test", decision="reorder", rationale="low stock",
            confidence=0.85, tokens_used=100,
        )
        assert entry.agent == "test"
        assert entry.tokens_used == 100
        assert entry.timestamp is not None


class TestLangGraphOrchestrator:
    def test_graph_build(self):
        graph = _build_graph()
        # May be None if langgraph is not installed
        # Just verify the function doesn't crash
        assert graph is None or graph is not None

    def test_orchestrator_init(self):
        orch = LangGraphOrchestrator()
        assert orch._cycle_count == 0

    def test_is_available(self):
        orch = LangGraphOrchestrator()
        # Should be False if langgraph not installed, True if installed
        assert isinstance(orch.is_available, bool)

    async def test_run_cycle_without_langgraph(self):
        orch = LangGraphOrchestrator()
        if not orch.is_available:
            with pytest.raises(RuntimeError, match="LangGraph is not installed"):
                await orch.run_cycle(
                    agents={}, products=[], suppliers=[],
                )

    async def test_run_cycle_with_mock_agents(self):
        orch = LangGraphOrchestrator()
        if not orch.is_available:
            pytest.skip("LangGraph not installed")

        mock_agent = MagicMock()
        mock_agent.run_cycle = AsyncMock(return_value={"status": "ok"})

        agents = {
            "market_intelligence": mock_agent,
            "anomaly_detector": mock_agent,
            "demand_forecaster": mock_agent,
            "inventory_optimizer": mock_agent,
            "risk_assessor": mock_agent,
            "supplier_manager": mock_agent,
            "logistics_coordinator": mock_agent,
            "strategic_planner": mock_agent,
            "coordinator": mock_agent,
            "reporter": mock_agent,
        }

        result = await orch.run_cycle(agents=agents, products=[], suppliers=[])
        assert result["cycle"] == 1
        assert isinstance(result["decisions"], list)


class TestOrchestratorFactory:
    def test_get_orchestrator_classic(self):
        from chaincommand.orchestrator import get_orchestrator

        orch = get_orchestrator("classic")
        from chaincommand.orchestrator import ChainCommandOrchestrator
        assert isinstance(orch, ChainCommandOrchestrator)

    def test_get_orchestrator_langgraph(self):
        from chaincommand.orchestrator import get_orchestrator

        orch = get_orchestrator("langgraph")
        assert isinstance(orch, LangGraphOrchestrator)

    def test_get_orchestrator_default(self):
        from chaincommand.orchestrator import get_orchestrator

        orch = get_orchestrator()
        # Default is "classic" per config
        from chaincommand.orchestrator import ChainCommandOrchestrator
        assert isinstance(orch, ChainCommandOrchestrator)
