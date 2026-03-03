"""Inventory Optimizer Agent — Tactical Layer.

Monitors inventory levels, calculates reorder points, manages safety stock.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..data.schemas import AgentAction, SupplyChainEvent
from ..utils.logging_config import get_logger
from .base_agent import BaseAgent

log = get_logger(__name__)


class InventoryOptimizerAgent(BaseAgent):
    name = "inventory_optimizer"
    role = "Monitor inventory levels, calculate reorder points, manage safety stock"
    layer = "tactical"

    async def handle_event(self, event: SupplyChainEvent) -> None:
        if event.event_type in ("low_stock_alert", "stockout_alert"):
            product_id = event.data.get("product_id", "")
            log.info("inventory_low_stock_event", product=product_id, severity=event.severity.value)
            # Trigger immediate reorder evaluation
        elif event.event_type == "overstock_alert":
            log.info("inventory_overstock_event", product=event.data.get("product_id"))
        elif event.event_type == "forecast_updated":
            log.info("inventory_forecast_update", product=event.data.get("product_id"))

    async def run_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._cycle_count += 1
        self._last_run = datetime.utcnow()

        results = {"agent": self.name, "actions": [], "reorders": [], "adjustments": []}

        # Step 1: Get full inventory status
        inv_action = AgentAction(
            agent_name=self.name,
            action_type="query_inventory_status",
            description="Check all inventory levels",
            input_data={},
        )
        await self.act(inv_action)
        results["actions"].append(inv_action.model_dump())

        products = context.get("products", [])

        # Step 2: Check each product for reorder needs
        for product in products:
            if product.current_stock < product.reorder_point:
                # Calculate optimal reorder
                rop_action = AgentAction(
                    agent_name=self.name,
                    action_type="calculate_reorder_point",
                    description=f"Calculate ROP for {product.product_id}",
                    input_data={"product_id": product.product_id, "service_level": 0.95},
                )
                rop_result = await self.act(rop_action)

                # Trigger reorder event
                emit_action = AgentAction(
                    agent_name=self.name,
                    action_type="emit_event",
                    description=f"Trigger reorder for {product.product_id}",
                    input_data={
                        "event_type": "reorder_triggered",
                        "severity": "high" if product.current_stock < product.safety_stock else "medium",
                        "source_agent": self.name,
                        "description": (
                            f"Reorder needed: {product.name} "
                            f"(stock={product.current_stock:.0f}, ROP={product.reorder_point:.0f})"
                        ),
                        "data": {
                            "product_id": product.product_id,
                            "current_stock": product.current_stock,
                            "reorder_point": rop_result.get("reorder_point", product.reorder_point),
                        },
                    },
                )
                await self.act(emit_action)
                results["reorders"].append({
                    "product_id": product.product_id,
                    "current_stock": product.current_stock,
                    **rop_result,
                })

        # Step 3: Optimize selected products
        for product in products[:3]:
            opt_action = AgentAction(
                agent_name=self.name,
                action_type="optimize_inventory",
                description=f"Optimize inventory for {product.product_id}",
                input_data={"product_id": product.product_id},
            )
            opt_result = await self.act(opt_action)
            if "error" not in opt_result:
                results["adjustments"].append(opt_result)

        # Step 4: Think and summarize
        think_context = {
            "total_products": len(products),
            "reorders_needed": len(results["reorders"]),
            "adjustments_made": len(results["adjustments"]),
        }
        analysis = await self.think(think_context)
        results["analysis"] = analysis

        log.info(
            "inventory_cycle_complete",
            cycle=self._cycle_count,
            reorders=len(results["reorders"]),
        )
        return results
