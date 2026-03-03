"""FastAPI application for ChainCommand dashboard and control."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..auth import require_ws_api_key
from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)

# ── Rate limiting ────────────────────────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(request: Request) -> None:
    """Simple in-memory per-IP rate limiter."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in window if t > now - 60]
    if len(_rate_limit_store[client_ip]) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_limit_store[client_ip].append(now)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    from ..orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    await orchestrator.initialize()
    log.info("app_startup_complete")
    yield
    await orchestrator.shutdown()
    log.info("app_shutdown_complete")


app = FastAPI(
    title="ChainCommand",
    description="Autonomous Supply Chain Optimizer — AI Agent Team",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ── Register routers ────────────────────────────────────────
from .routes.control import router as control_router  # noqa: E402
from .routes.dashboard import router as dashboard_router  # noqa: E402

app.include_router(dashboard_router, prefix="/api")
app.include_router(control_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "ChainCommand",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "name": "ChainCommand", "version": "1.0.0"}


def _json_serial(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Live event stream via WebSocket (top-level route)."""
    # Auth check
    await require_ws_api_key(websocket)

    await websocket.accept()
    log.info("ws_client_connected")

    try:
        seen_ids: set[str] = set()
        while True:
            await asyncio.sleep(1)
            from ..orchestrator import _runtime

            if not _runtime.event_bus:
                continue

            events = _runtime.event_bus.recent_events[-20:]
            for evt in events:
                eid = evt.event_id
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    data = evt.model_dump()
                    text = json.dumps(data, default=_json_serial)
                    await websocket.send_text(text)
            # Cap memory
            if len(seen_ids) > 5000:
                seen_ids = set(list(seen_ids)[-2000:])
    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
    except Exception as exc:
        log.error("ws_error", error=str(exc))
