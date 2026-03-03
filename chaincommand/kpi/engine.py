"""KPI calculation engine for supply chain metrics."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..config import settings
from ..data.schemas import (
    AlertSeverity,
    KPISnapshot,
    OrderStatus,
    SupplyChainEvent,
)
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class KPIEngine:
    """Calculates KPI snapshots and checks threshold violations."""

    def __init__(self) -> None:
        self._history: List[KPISnapshot] = []

    def calculate_snapshot(
        self,
        products: list,
        purchase_orders: list,
        suppliers: list,
    ) -> KPISnapshot:
        """Compute a full KPI snapshot from current system state."""

        # ── OTIF (On-Time In-Full) ──────────────────────────
        delivered = [po for po in purchase_orders if po.status == OrderStatus.DELIVERED]
        on_time_in_full = 0
        for po in delivered:
            if po.expected_delivery and po.created_at:
                # simplified: assume delivered POs are on-time
                on_time_in_full += 1
        otif = on_time_in_full / max(len(delivered), 1)

        # ── Fill Rate ───────────────────────────────────────
        total_demand = sum(p.daily_demand_avg for p in products)
        fulfilled = sum(
            min(p.current_stock, p.daily_demand_avg) for p in products
        )
        fill_rate = fulfilled / max(total_demand, 1)

        # ── MAPE (from forecaster accuracy cache) ──────────
        # Will be updated by forecaster; use last known value
        mape = self._history[-1].mape if self._history else 12.0

        # ── DSI (Days Sales of Inventory) ──────────────────
        total_stock = sum(p.current_stock for p in products)
        avg_daily = sum(p.daily_demand_avg for p in products)
        dsi = total_stock / max(avg_daily, 1)

        # ── Stockout Count ─────────────────────────────────
        stockout_count = sum(
            1 for p in products if p.current_stock <= 0
        )

        # ── Total Inventory Value ──────────────────────────
        total_value = sum(p.current_stock * p.unit_cost for p in products)

        # ── Carrying Cost (annual 25% of inventory value, daily) ──
        carrying_cost = total_value * 0.25 / 365

        # ── Order Cycle Time ───────────────────────────────
        cycle_times = []
        for po in delivered:
            if po.expected_delivery and po.created_at:
                delta = (po.expected_delivery - po.created_at).total_seconds() / 86400
                cycle_times.append(delta)
        order_cycle_time = float(np.mean(cycle_times)) if cycle_times else 7.0

        # ── Perfect Order Rate ─────────────────────────────
        total_orders = len([po for po in purchase_orders if po.status != OrderStatus.CANCELLED])
        perfect = len([
            po for po in delivered
            if po.quantity > 0  # simplified: delivered = "perfect"
        ])
        perfect_order_rate = perfect / max(total_orders, 1)

        # ── Inventory Turnover ─────────────────────────────
        annual_cogs = sum(p.daily_demand_avg * p.unit_cost * 365 for p in products)
        inventory_turnover = annual_cogs / max(total_value, 1)

        # ── Backorder Rate ─────────────────────────────────
        # Count products at zero stock with active demand as backordered
        backordered = sum(
            1 for p in products
            if p.current_stock <= 0 and p.daily_demand_avg > 0
        )
        backorder_rate = backordered / max(len(products), 1)

        # ── Supplier Defect Rate ───────────────────────────
        active_suppliers = [s for s in suppliers if s.is_active]
        supplier_defect_rate = (
            float(np.mean([s.defect_rate for s in active_suppliers]))
            if active_suppliers else 0.02
        )

        snapshot = KPISnapshot(
            otif=round(otif, 4),
            fill_rate=round(fill_rate, 4),
            mape=round(mape, 2),
            dsi=round(dsi, 1),
            stockout_count=stockout_count,
            total_inventory_value=round(total_value, 2),
            carrying_cost=round(carrying_cost, 2),
            order_cycle_time=round(order_cycle_time, 1),
            perfect_order_rate=round(perfect_order_rate, 4),
            inventory_turnover=round(inventory_turnover, 2),
            backorder_rate=round(backorder_rate, 4),
            supplier_defect_rate=round(supplier_defect_rate, 4),
        )

        self._history.append(snapshot)
        log.info(
            "kpi_snapshot",
            otif=snapshot.otif,
            fill_rate=snapshot.fill_rate,
            dsi=snapshot.dsi,
            stockouts=snapshot.stockout_count,
        )
        return snapshot

    def check_thresholds(self, snapshot: KPISnapshot) -> List[SupplyChainEvent]:
        """Check if any KPI exceeds configured thresholds."""
        events: List[SupplyChainEvent] = []

        if snapshot.otif < settings.otif_target:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.HIGH,
                source_agent="kpi_engine",
                description=f"OTIF {snapshot.otif:.1%} below target {settings.otif_target:.1%}",
                data={"metric": "otif", "value": snapshot.otif, "target": settings.otif_target},
            ))

        if snapshot.fill_rate < settings.fill_rate_target:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.HIGH,
                source_agent="kpi_engine",
                description=f"Fill rate {snapshot.fill_rate:.1%} below target {settings.fill_rate_target:.1%}",
                data={"metric": "fill_rate", "value": snapshot.fill_rate, "target": settings.fill_rate_target},
            ))

        if snapshot.mape > settings.mape_threshold:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.MEDIUM,
                source_agent="kpi_engine",
                description=f"MAPE {snapshot.mape:.1f}% exceeds threshold {settings.mape_threshold:.1f}%",
                data={"metric": "mape", "value": snapshot.mape, "target": settings.mape_threshold},
            ))

        if snapshot.dsi > settings.dsi_max:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.MEDIUM,
                source_agent="kpi_engine",
                description=f"DSI {snapshot.dsi:.1f} exceeds max {settings.dsi_max:.1f}",
                data={"metric": "dsi", "value": snapshot.dsi, "target": settings.dsi_max},
            ))

        if snapshot.dsi < settings.dsi_min:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.HIGH,
                source_agent="kpi_engine",
                description=f"DSI {snapshot.dsi:.1f} below min {settings.dsi_min:.1f}",
                data={"metric": "dsi", "value": snapshot.dsi, "target": settings.dsi_min},
            ))

        if snapshot.stockout_count > settings.stockout_tolerance:
            events.append(SupplyChainEvent(
                event_type="kpi_threshold_violated",
                severity=AlertSeverity.CRITICAL,
                source_agent="kpi_engine",
                description=(
                    f"Stockout count {snapshot.stockout_count} "
                    f"exceeds tolerance {settings.stockout_tolerance}"
                ),
                data={
                    "metric": "stockout_count",
                    "value": snapshot.stockout_count,
                    "target": settings.stockout_tolerance,
                },
            ))

        return events

    def get_trend(self, metric: str, periods: int = 30) -> Dict[str, Any]:
        """Get trend data for a specific KPI metric."""
        recent = self._history[-periods:]
        if not recent:
            return {"metric": metric, "values": [], "trend": "no_data"}

        values = [getattr(s, metric, 0) for s in recent]
        timestamps = [s.timestamp.isoformat() for s in recent]

        # Simple trend detection (account for metrics where lower is better)
        lower_is_better = {
            "mape", "dsi", "stockout_count", "carrying_cost",
            "backorder_rate", "supplier_defect_rate", "order_cycle_time",
        }
        if len(values) >= 3:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if metric in lower_is_better:
                trend = "improving" if slope < -0.001 else "declining" if slope > 0.001 else "stable"
            else:
                trend = "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable"
        else:
            trend = "insufficient_data"

        return {
            "metric": metric,
            "values": values,
            "timestamps": timestamps,
            "current": values[-1] if values else 0,
            "average": round(float(np.mean(values)), 4) if values else 0,
            "trend": trend,
        }

    @property
    def history(self) -> List[KPISnapshot]:
        return self._history
