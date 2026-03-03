"""Coordinator Agent (CSCO) — Orchestration Layer.

Chief Supply Chain Officer. Coordinates all agents, resolves conflicts,
enforces budget/capacity constraints, produces executive summaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ..data.schemas import AgentAction, SupplyChainEvent
from ..utils.logging_config import get_logger
from .base_agent import BaseAgent

log = get_logger(__name__)


class CoordinatorAgent(BaseAgent):
    name = "coordinator"
    role = (
        "Chief coordinator: orchestrate all agents, resolve conflicts, "
        "enforce budget/capacity constraints, produce executive summaries"
    )
    layer = "orchestration"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conflict_log: List[Dict[str, Any]] = []

    async def handle_event(self, event: SupplyChainEvent) -> None:
        # Coordinator listens to all events
        log.debug("coordinator_event", event_type=event.event_type, severity=event.severity.value)

    async def run_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._cycle_count += 1
        self._last_run = datetime.utcnow()

        results = {
            "agent": self.name,
            "cycle": self._cycle_count,
            "conflicts_resolved": [],
            "priority_actions": [],
            "executive_summary": "",
        }

        agent_results = context.get("agent_results", {})

        # Step 1: Conflict detection and resolution
        conflicts = self._detect_conflicts(agent_results)
        for conflict in conflicts:
            resolution = await self._resolve_conflict(conflict)
            results["conflicts_resolved"].append(resolution)

        # Step 2: Priority ranking of recommended actions
        all_actions = self._collect_actions(agent_results)
        prioritized = self._prioritize_actions(all_actions)
        results["priority_actions"] = prioritized[:10]

        # Step 3: HITL gate management
        from ..orchestrator import _runtime
        pending = _runtime.pending_approvals
        results["pending_approvals"] = len(pending)

        # Step 4: KPI review
        kpi_action = AgentAction(
            agent_name=self.name,
            action_type="query_kpi_history",
            description="Review KPIs for executive summary",
            input_data={"periods": 5},
        )
        kpi_data = await self.act(kpi_action)

        # Step 5: Generate executive summary
        summary_context = {
            "cycle": self._cycle_count,
            "agents_reporting": len(agent_results),
            "conflicts": len(conflicts),
            "pending_approvals": len(pending),
            "priority_actions": len(prioritized),
            "kpi_snapshots": kpi_data.get("count", 0),
        }
        executive_summary = await self.think(summary_context)
        results["executive_summary"] = executive_summary

        # Step 6: Emit cycle complete event
        emit_action = AgentAction(
            agent_name=self.name,
            action_type="emit_event",
            description="Signal cycle completion",
            input_data={
                "event_type": "cycle_complete",
                "severity": "low",
                "source_agent": self.name,
                "description": f"Decision cycle {self._cycle_count} complete",
                "data": {
                    "cycle": self._cycle_count,
                    "conflicts": len(conflicts),
                    "actions": len(prioritized),
                },
            },
        )
        await self.act(emit_action)

        log.info(
            "coordinator_cycle_complete",
            cycle=self._cycle_count,
            conflicts=len(conflicts),
            actions=len(prioritized),
        )
        return results

    def _detect_conflicts(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent recommendations."""
        conflicts = []

        # Example: inventory optimizer wants to increase stock, but
        # strategic planner recommends reducing due to overstock pattern
        inv_result = agent_results.get("inventory_optimizer", {})
        planner_result = agent_results.get("strategic_planner", {})

        inv_reorders = inv_result.get("reorders", [])
        planner_recs = planner_result.get("recommendations", [])

        # Check if any product has conflicting recommendations
        for reorder in inv_reorders:
            pid = reorder.get("product_id", "")
            for rec in planner_recs:
                rec_pid = rec.get("product_id", "")
                if pid == rec_pid:
                    # Simplified conflict detection
                    conflicts.append({
                        "type": "reorder_vs_strategy",
                        "product_id": pid,
                        "inventory_says": "reorder",
                        "planner_says": rec,
                    })

        return conflicts

    async def _resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to arbitrate between conflicting recommendations."""
        resolution = await self.think({
            "conflict_type": conflict.get("type"),
            "details": conflict,
            "task": "Resolve this conflict between agent recommendations",
        })
        self._conflict_log.append({
            "conflict": conflict,
            "resolution": resolution,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return {
            "conflict": conflict,
            "resolution": resolution,
            "resolved": True,
        }

    def _collect_actions(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect all recommended actions from all agents."""
        actions = []
        for agent_name, result in agent_results.items():
            if isinstance(result, dict):
                for key in ("reorders", "orders_created", "adjustments", "mitigations", "alerts"):
                    items = result.get(key, [])
                    for item in items:
                        actions.append({
                            "source": agent_name,
                            "type": key,
                            "data": item,
                        })
        return actions

    def _prioritize_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank actions by urgency and impact."""
        priority_order = {
            "reorders": 1,
            "mitigations": 2,
            "orders_created": 3,
            "alerts": 4,
            "adjustments": 5,
        }
        return sorted(actions, key=lambda a: priority_order.get(a.get("type", ""), 99))
