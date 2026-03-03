"""Dashboard routes — KPI, inventory, agents, events, forecasts, approvals, AWS."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ...auth import require_api_key
from ...config import settings
from ...data.schemas import ApprovalStatus
from ...utils.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["dashboard"], dependencies=[Depends(require_api_key)])


# ── KPI ──────────────────────────────────────────────────────

@router.get("/kpi/current")
async def get_current_kpi():
    """Get the latest KPI snapshot."""
    from ...orchestrator import _runtime

    if not _runtime.kpi_engine or not _runtime.kpi_engine.history:
        return {"error": "No KPI data available yet"}
    snapshot = _runtime.kpi_engine.history[-1]
    return snapshot.model_dump()


@router.get("/kpi/history")
async def get_kpi_history(periods: int = Query(default=30, ge=1, le=365)):
    """Get KPI history for trend analysis."""
    from ...orchestrator import _runtime

    if not _runtime.kpi_engine:
        return {"snapshots": [], "count": 0}
    snapshots = _runtime.kpi_engine.history[-periods:]
    return {
        "snapshots": [s.model_dump() for s in snapshots],
        "count": len(snapshots),
    }


# ── Inventory ────────────────────────────────────────────────

@router.get("/inventory/status")
async def get_inventory_status(product_id: Optional[str] = None):
    """Get inventory status for all or a specific product."""
    from ...orchestrator import _runtime

    products = _runtime.products or []
    if product_id:
        products = [p for p in products if p.product_id == product_id]

    items = []
    for p in products:
        dsi = p.current_stock / p.daily_demand_avg if p.daily_demand_avg > 0 else 999
        items.append({
            "product_id": p.product_id,
            "name": p.name,
            "category": p.category.value,
            "current_stock": p.current_stock,
            "reorder_point": p.reorder_point,
            "safety_stock": p.safety_stock,
            "daily_demand_avg": p.daily_demand_avg,
            "days_of_supply": round(dsi, 1),
            "unit_cost": p.unit_cost,
            "status": (
                "critical" if p.current_stock < p.safety_stock
                else "low" if p.current_stock < p.reorder_point
                else "healthy"
            ),
        })

    return {"products": items, "count": len(items)}


# ── Agents ───────────────────────────────────────────────────

@router.get("/agents/status")
async def get_agents_status():
    """Get status of all agents."""
    from ...orchestrator import _runtime

    agents = _runtime.agents or {}
    return {
        "agents": {name: agent.get_status() for name, agent in agents.items()},
        "count": len(agents),
    }


# ── Events ───────────────────────────────────────────────────

@router.get("/events/recent")
async def get_recent_events(limit: int = Query(default=50, ge=1, le=200)):
    """Get recent supply chain events."""
    from ...orchestrator import _runtime

    if not _runtime.event_bus:
        return {"events": [], "count": 0}
    events = _runtime.event_bus.recent_events[-limit:]
    return {
        "events": [e.model_dump() for e in reversed(events)],
        "count": len(events),
    }


def _get_recent_events(limit: int = 20) -> list[dict]:
    """Helper for WebSocket: return recent events as dicts."""
    from ...orchestrator import _runtime

    if not _runtime.event_bus:
        return []
    events = _runtime.event_bus.recent_events[-limit:]
    return [e.model_dump() for e in events]


# ── Forecast ─────────────────────────────────────────────────

@router.get("/forecast/{product_id}")
async def get_forecast(product_id: str, horizon: int = Query(default=30, ge=1, le=90)):
    """Get demand forecast for a product."""
    from ...orchestrator import _runtime

    if not _runtime.forecaster:
        return {"error": "Forecaster not initialized"}

    results = _runtime.forecaster.predict(product_id, horizon)
    return {
        "product_id": product_id,
        "horizon": horizon,
        "forecasts": [r.model_dump() for r in results],
        "accuracy": _runtime.forecaster.get_accuracy(product_id),
    }


# ── Human Approval ───────────────────────────────────────────

@router.get("/approvals/pending")
async def get_pending_approvals():
    """List all pending approval requests."""
    from ...orchestrator import _runtime

    pending = {
        k: v.model_dump()
        for k, v in _runtime.pending_approvals.items()
        if v.status == ApprovalStatus.PENDING
    }
    return {"approvals": pending, "count": len(pending)}


@router.post("/approval/{request_id}/decide")
async def decide_approval(request_id: str, approved: bool, reason: str = ""):
    """Human decision on an approval request."""
    from ...orchestrator import _runtime

    approval = _runtime.pending_approvals.get(request_id)
    if not approval:
        return {"error": f"Approval request {request_id} not found"}

    approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
    approval.decided_at = datetime.now(timezone.utc)
    approval.decided_by = "human"
    approval.reason = reason

    log.info("approval_decided", request_id=request_id, approved=approved)
    return {"request_id": request_id, "status": approval.status.value, "reason": reason}


# ── AWS Integration ──────────────────────────────────────

@router.get("/aws/status")
async def get_aws_status():
    """Return AWS connection status and configuration."""
    from ...orchestrator import _runtime

    backend_type = type(_runtime.backend).__name__ if _runtime.backend else "None"
    return {
        "enabled": settings.aws_enabled,
        "backend": backend_type,
        "region": settings.aws_region,
        "s3_bucket": settings.aws_s3_bucket,
        "redshift_host": settings.aws_redshift_host or "(not configured)",
        "athena_database": settings.aws_athena_database,
        "quicksight_account": settings.aws_quicksight_account_id or "(not configured)",
    }


@router.get("/aws/kpi-trend/{metric}")
async def get_aws_kpi_trend(
    metric: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Query KPI trend from Redshift via the persistence backend."""
    from ...orchestrator import _runtime

    if not _runtime.backend or not settings.aws_enabled:
        return {"error": "AWS backend not enabled", "data": []}
    data = await _runtime.backend.query_kpi_trend(metric, days)
    return {"metric": metric, "days": days, "data": data}


@router.get("/aws/query")
async def run_aws_query(
    event_type: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Execute an Athena ad-hoc query for events by type."""
    from ...orchestrator import _runtime

    if not _runtime.backend or not settings.aws_enabled:
        return {"error": "AWS backend not enabled", "data": []}
    data = await _runtime.backend.query_events(event_type, limit)
    return {"event_type": event_type, "limit": limit, "data": data}


@router.get("/aws/dashboards")
async def list_aws_dashboards():
    """List QuickSight dashboards (requires AWS backend)."""
    from ...orchestrator import _runtime

    if not _runtime.backend or not settings.aws_enabled:
        return {"error": "AWS backend not enabled", "dashboards": []}
    try:
        from ...aws.quicksight_client import QuickSightClient

        qs = QuickSightClient()
        dashboards = qs.list_dashboards()
        return {"dashboards": dashboards, "count": len(dashboards)}
    except Exception as exc:
        log.error("aws_dashboards_error", error=str(exc))
        return {"error": str(exc), "dashboards": []}
