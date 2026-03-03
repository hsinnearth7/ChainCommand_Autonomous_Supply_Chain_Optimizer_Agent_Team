"""Shared test fixtures for ChainCommand."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from chaincommand.data.schemas import (
    HumanApprovalRequest,
    KPISnapshot,
    Product,
    ProductCategory,
    PurchaseOrder,
    Supplier,
)


@dataclass
class MockRuntime:
    """Lightweight mock of _RuntimeState for testing."""

    products: Optional[List[Product]] = None
    suppliers: Optional[List[Supplier]] = None
    demand_df: Any = None
    forecaster: Any = None
    anomaly_detector: Any = None
    optimizer: Any = None
    kpi_engine: Any = None
    event_bus: Any = None
    monitor: Any = None
    agents: Dict[str, Any] = field(default_factory=dict)
    purchase_orders: List[PurchaseOrder] = field(default_factory=list)
    pending_approvals: Dict[str, HumanApprovalRequest] = field(default_factory=dict)
    kpi_history: List[KPISnapshot] = field(default_factory=list)
    backend: Any = None


@pytest.fixture
def mock_runtime():
    """Provide a mock runtime with sample products and suppliers."""
    rt = MockRuntime()
    rt.products = [
        Product(
            product_id="PRD-test01",
            name="Test Widget",
            category=ProductCategory.ELECTRONICS,
            unit_cost=10.0,
            selling_price=25.0,
            current_stock=500.0,
            reorder_point=100.0,
            safety_stock=50.0,
            daily_demand_avg=20.0,
            daily_demand_std=5.0,
        ),
    ]
    rt.suppliers = [
        Supplier(
            supplier_id="SUP-test01",
            name="Test Supplier",
            lead_time_mean=5.0,
        ),
    ]
    return rt


# ── v2.0 fixtures ─────────────────────────────────────────


@pytest.fixture
def sample_products():
    """Multiple test products across categories."""
    return [
        Product(
            product_id=f"PRD-{i:04d}",
            name=f"Product {i}",
            category=list(ProductCategory)[i % 5],
            unit_cost=10.0 + i * 2,
            selling_price=25.0 + i * 5,
            current_stock=500.0 - i * 50,
            reorder_point=100.0,
            safety_stock=50.0,
            daily_demand_avg=20.0 + i,
            daily_demand_std=5.0,
            lead_time_days=7,
            min_order_qty=100,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_suppliers():
    """Multiple test suppliers."""
    return [
        Supplier(
            supplier_id=f"SUP-{i:04d}",
            name=f"Supplier {i}",
            reliability_score=0.9 - i * 0.05,
            lead_time_mean=5.0 + i,
            cost_multiplier=1.0 + i * 0.1,
            capacity=10000.0 - i * 1000,
            defect_rate=0.01 + i * 0.005,
            on_time_rate=0.95 - i * 0.03,
        )
        for i in range(4)
    ]


@pytest.fixture
def sample_demand_df(sample_products):
    """365-day demand history DataFrame."""
    records = []
    rng = np.random.RandomState(42)
    base_date = datetime(2024, 1, 1)
    for product in sample_products:
        for day in range(365):
            dt = base_date + timedelta(days=day)
            qty = max(0, rng.normal(product.daily_demand_avg, product.daily_demand_std))
            records.append({
                "date": dt,
                "product_id": product.product_id,
                "quantity": round(qty, 1),
                "is_promotion": rng.random() < 0.05,
                "is_holiday": rng.random() < 0.03,
                "temperature": round(rng.normal(20, 5), 1),
                "day_of_week": dt.weekday(),
                "month": dt.month,
            })
    return pd.DataFrame(records)


@pytest.fixture
def trained_forecaster(sample_demand_df, sample_products):
    """Pre-trained EnsembleForecaster."""
    from chaincommand.models.forecaster import EnsembleForecaster

    forecaster = EnsembleForecaster()
    pids = [p.product_id for p in sample_products[:3]]
    forecaster.train_all(sample_demand_df, pids)
    return forecaster


@pytest.fixture
def kpi_engine():
    """Fresh KPI engine instance."""
    from chaincommand.kpi.engine import KPIEngine

    return KPIEngine()


@pytest.fixture
def event_bus():
    """Fresh EventBus instance."""
    from chaincommand.events.bus import EventBus

    return EventBus()


@pytest.fixture
def mock_llm():
    """MockLLM instance for agent testing."""
    from chaincommand.llm.mock_llm import MockLLM

    return MockLLM()


@pytest.fixture
def app_no_lifespan():
    """Create a FastAPI test app without the real lifespan (no orchestrator init)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from chaincommand.config import settings

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = FastAPI(lifespan=noop_lifespan)

    # Mirror CORS config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    # Import and include routers
    from chaincommand.api.routes.control import router as control_router
    from chaincommand.api.routes.dashboard import router as dashboard_router

    app.include_router(dashboard_router, prefix="/api")
    app.include_router(control_router, prefix="/api")

    @app.get("/")
    async def root():
        return {
            "name": "ChainCommand",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
        }

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "name": "ChainCommand", "version": "2.0.0"}

    return app


@pytest.fixture
def client(app_no_lifespan):
    """HTTPX async test client for the FastAPI app."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app_no_lifespan)
    return AsyncClient(transport=transport, base_url="http://test")
