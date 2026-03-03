"""Control routes — simulation start/stop/speed, agent triggers."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ...auth import require_api_key
from ...config import settings
from ...utils.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["control"], dependencies=[Depends(require_api_key)])


@router.post("/simulation/start")
async def start_simulation():
    """Start the simulation loop."""
    from ...orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    if orchestrator.running:
        return {"status": "already_running"}

    asyncio.create_task(orchestrator.run_loop())
    return {"status": "started", "speed": settings.simulation_speed}


@router.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation loop."""
    from ...orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    await orchestrator.stop_loop()
    return {"status": "stopped"}


@router.post("/simulation/speed")
async def set_speed(speed: float):
    """Adjust simulation speed multiplier."""
    if speed < 0.1 or speed > 100:
        return {"error": "Speed must be between 0.1 and 100"}

    settings.simulation_speed = speed
    log.info("simulation_speed_changed", speed=speed)
    return {"status": "ok", "speed": speed}


@router.post("/agents/{agent_name}/trigger")
async def trigger_agent(agent_name: str):
    """Manually trigger a specific agent's cycle."""
    from ...orchestrator import _runtime

    agents = _runtime.agents or {}
    agent = agents.get(agent_name)
    if agent is None:
        available = list(agents.keys())
        return {"error": f"Agent '{agent_name}' not found", "available": available}

    context = {"products": _runtime.products or [], "manual_trigger": True}
    result = await agent.run_cycle(context)
    return {"agent": agent_name, "result": result}


@router.get("/simulation/status")
async def simulation_status():
    """Get current simulation status."""
    from ...orchestrator import _runtime, get_orchestrator

    orchestrator = get_orchestrator()
    return {
        "running": orchestrator.running,
        "cycle_count": orchestrator.cycle_count,
        "speed": settings.simulation_speed,
        "products": len(_runtime.products) if _runtime.products else 0,
        "suppliers": len(_runtime.suppliers) if _runtime.suppliers else 0,
        "purchase_orders": len(_runtime.purchase_orders),
        "pending_approvals": len(_runtime.pending_approvals),
        "events": _runtime.event_bus.event_count if _runtime.event_bus else 0,
    }
