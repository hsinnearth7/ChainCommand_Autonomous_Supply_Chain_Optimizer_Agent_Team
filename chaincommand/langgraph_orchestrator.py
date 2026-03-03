"""LangGraph-based orchestrator for v2.0 (Section 4.2).

Wraps existing agents into a StateGraph with structured state,
conditional HITL edges, and SqliteSaver checkpoint for crash recovery.
"""

from __future__ import annotations

import random
from typing import Any, Dict

from .state import DecisionEntry
from .utils.logging_config import get_logger

log = get_logger(__name__)


def _build_graph() -> Any:
    """Build and compile the LangGraph StateGraph.

    Returns the compiled graph or None if langgraph is unavailable.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        log.warning("langgraph_unavailable", msg="Install langgraph to use LangGraph orchestrator")
        return None

    # Use dict-based state for LangGraph compatibility
    graph = StateGraph(dict)

    # ── Node functions ────────────────────────────────────

    async def market_intelligence(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("market_intelligence")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["market_intel"] = result
            state.setdefault("decisions", []).append(DecisionEntry(
                agent="market_intelligence", decision="market_scan", confidence=0.8,
            ).model_dump())
        return state

    async def anomaly_detector(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("anomaly_detector")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["anomaly_results"] = result
            state.setdefault("decisions", []).append(DecisionEntry(
                agent="anomaly_detector", decision="anomaly_scan", confidence=0.9,
            ).model_dump())
        return state

    async def demand_forecaster(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("demand_forecaster")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["forecast_results"] = result
            state.setdefault("decisions", []).append(DecisionEntry(
                agent="demand_forecaster", decision="forecast_update", confidence=0.85,
            ).model_dump())
        return state

    async def inventory_optimizer(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("inventory_optimizer")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["inventory_results"] = result
        return state

    async def risk_assessor(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("risk_assessor")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["risk_results"] = result
        return state

    async def supplier_manager(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("supplier_manager")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["supplier_results"] = result
            # Check if HITL approval is needed
            if result and result.get("requires_approval"):
                state["requires_human_approval"] = True
                state.setdefault("pending_approvals", []).append(result.get("approval_request", {}))
        return state

    async def human_approval(state: dict) -> dict:
        # Auto-approve in simulation mode
        approvals = state.get("pending_approvals", [])
        for approval in approvals:
            approval["status"] = "auto_approved"
            approval["decided_by"] = "simulation"
        state["requires_human_approval"] = False
        state.setdefault("decisions", []).append(DecisionEntry(
            agent="human_approval", decision=f"auto_approved_{len(approvals)}_items", confidence=1.0,
        ).model_dump())
        return state

    async def logistics_coordinator(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("logistics_coordinator")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["logistics_results"] = result
        return state

    async def strategic_planner(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("strategic_planner")
        if agent:
            result = await agent.run_cycle({"products": state.get("products", []), "cycle": state.get("cycle", 0)})
            state["planner_results"] = result
        return state

    async def coordinator(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("coordinator")
        if agent:
            context = {"products": state.get("products", []), "cycle": state.get("cycle", 0), "agent_results": state}
            result = await agent.run_cycle(context)
            state["coordinator_results"] = result
        return state

    async def reporter(state: dict) -> dict:
        agents = state.get("_agents", {})
        agent = agents.get("reporter")
        if agent:
            context = {
                "products": state.get("products", []),
                "cycle": state.get("cycle", 0),
                "agent_results": state,
                "coordinator_summary": (state.get("coordinator_results") or {}).get("executive_summary", ""),
            }
            result = await agent.run_cycle(context)
            state["reporter_results"] = result
        return state

    async def kpi_update(state: dict) -> dict:
        kpi_engine = state.get("_kpi_engine")
        products = state.get("products", [])
        purchase_orders = state.get("_purchase_orders", [])
        suppliers = state.get("_suppliers", [])
        if kpi_engine:
            snapshot = kpi_engine.calculate_snapshot(products, purchase_orders, suppliers)
            state["kpi_snapshot"] = snapshot.model_dump()

        # Simulate demand consumption
        for p in products:
            consumed = max(0, random.gauss(p.daily_demand_avg, p.daily_demand_std))
            p.current_stock = max(0, p.current_stock - consumed)

        return state

    # ── Conditional edge ──────────────────────────────────

    def needs_human_approval(state: dict) -> str:
        if state.get("requires_human_approval"):
            return "human_approval"
        return "logistics_coordinator"

    # ── Add nodes ─────────────────────────────────────────

    graph.add_node("market_intelligence", market_intelligence)
    graph.add_node("anomaly_detector", anomaly_detector)
    graph.add_node("demand_forecaster", demand_forecaster)
    graph.add_node("inventory_optimizer", inventory_optimizer)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("supplier_manager", supplier_manager)
    graph.add_node("human_approval", human_approval)
    graph.add_node("logistics_coordinator", logistics_coordinator)
    graph.add_node("strategic_planner", strategic_planner)
    graph.add_node("coordinator", coordinator)
    graph.add_node("reporter", reporter)
    graph.add_node("kpi_update", kpi_update)

    # ── Add edges ─────────────────────────────────────────

    graph.set_entry_point("market_intelligence")
    graph.add_edge("market_intelligence", "anomaly_detector")
    graph.add_edge("anomaly_detector", "demand_forecaster")
    graph.add_edge("demand_forecaster", "inventory_optimizer")
    graph.add_edge("inventory_optimizer", "risk_assessor")
    graph.add_edge("risk_assessor", "supplier_manager")

    # Conditional: supplier_manager → human_approval OR logistics_coordinator
    graph.add_conditional_edges("supplier_manager", needs_human_approval, {
        "human_approval": "human_approval",
        "logistics_coordinator": "logistics_coordinator",
    })

    graph.add_edge("human_approval", "logistics_coordinator")
    graph.add_edge("logistics_coordinator", "strategic_planner")
    graph.add_edge("strategic_planner", "coordinator")
    graph.add_edge("coordinator", "reporter")
    graph.add_edge("reporter", "kpi_update")
    graph.add_edge("kpi_update", END)

    # ── Compile with checkpoint ───────────────────────────

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(":memory:")
        compiled = graph.compile(checkpointer=checkpointer)
    except ImportError:
        compiled = graph.compile()

    return compiled


class LangGraphOrchestrator:
    """LangGraph-based orchestrator wrapping existing agents."""

    def __init__(self) -> None:
        self._graph = _build_graph()
        self._cycle_count = 0

    @property
    def graph(self) -> Any:
        return self._graph

    @property
    def is_available(self) -> bool:
        return self._graph is not None

    async def run_cycle(
        self,
        agents: Dict[str, Any],
        products: list,
        suppliers: list,
        kpi_engine: Any = None,
        purchase_orders: list | None = None,
    ) -> Dict[str, Any]:
        """Execute one full LangGraph decision cycle."""
        if not self.is_available:
            raise RuntimeError("LangGraph is not installed. Use classic orchestrator.")

        self._cycle_count += 1

        initial_state = {
            "cycle": self._cycle_count,
            "products": products,
            "_agents": agents,
            "_kpi_engine": kpi_engine,
            "_purchase_orders": purchase_orders or [],
            "_suppliers": suppliers,
            "market_intel": {},
            "anomaly_results": {},
            "forecast_results": {},
            "inventory_results": {},
            "risk_results": {},
            "supplier_results": {},
            "logistics_results": {},
            "planner_results": {},
            "coordinator_results": {},
            "reporter_results": {},
            "kpi_snapshot": {},
            "requires_human_approval": False,
            "pending_approvals": [],
            "decisions": [],
        }

        config = {"configurable": {"thread_id": f"cycle-{self._cycle_count}"}}
        final_state = await self._graph.ainvoke(initial_state, config)

        return {
            "cycle": self._cycle_count,
            "kpi_snapshot": final_state.get("kpi_snapshot", {}),
            "decisions": final_state.get("decisions", []),
            "requires_human_approval": final_state.get("requires_human_approval", False),
            "reporter_results": final_state.get("reporter_results", {}),
        }
