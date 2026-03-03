"""ChainCommand Orchestrator — system coordinator and runtime state."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import settings
from .data.schemas import (
    HumanApprovalRequest,
    KPISnapshot,
    Product,
    PurchaseOrder,
    Supplier,
)
from .utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)


# ── Runtime state (singleton) ───────────────────────────────

@dataclass
class _RuntimeState:
    """Mutable global state shared across agents, tools, and API."""

    products: Optional[List[Product]] = None
    suppliers: Optional[List[Supplier]] = None
    demand_df: Optional[pd.DataFrame] = None

    # ML models
    forecaster: Any = None
    anomaly_detector: Any = None
    optimizer: Any = None

    # Engines
    kpi_engine: Any = None
    event_bus: Any = None
    monitor: Any = None

    # Agent registry
    agents: Dict[str, Any] = field(default_factory=dict)

    # Transaction state
    purchase_orders: List[PurchaseOrder] = field(default_factory=list)
    pending_approvals: Dict[str, HumanApprovalRequest] = field(default_factory=dict)
    kpi_history: List[KPISnapshot] = field(default_factory=list)

    # Persistence backend
    backend: Any = None


_runtime = _RuntimeState()


# ── Orchestrator ────────────────────────────────────────────

class ChainCommandOrchestrator:
    """Main orchestrator that initializes, runs cycles, and shuts down."""

    def __init__(self, ui_callback: Any = None) -> None:
        self._running = False
        self._cycle_count = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._ui_callback = ui_callback

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def _ui(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Safely invoke a UI callback method if present."""
        if self._ui_callback:
            fn = getattr(self._ui_callback, method, None)
            if fn:
                fn(*args, **kwargs)

    async def initialize(self) -> None:
        """Bootstrap the entire system."""
        setup_logging(quiet=self._ui_callback is not None)
        random.seed(settings.random_seed)
        log.info("initializing", llm_mode=settings.llm_mode.value)

        # Phase 0: Generate synthetic data
        self._ui("on_init_phase_start", 0)
        from .data.generator import generate_all

        log.info("generating_data")
        products, suppliers, demand_df = generate_all()
        _runtime.products = products
        _runtime.suppliers = suppliers
        _runtime.demand_df = demand_df
        log.info("data_generated", products=len(products), suppliers=len(suppliers))
        self._ui("on_init_phase_complete", 0)

        # Phase 1: Train forecaster
        self._ui("on_init_phase_start", 1)
        from .models.forecaster import EnsembleForecaster

        log.info("training_models")
        _runtime.forecaster = EnsembleForecaster()
        product_ids = [p.product_id for p in products[:20]]  # Train on first 20 for speed
        _runtime.forecaster.train_all(demand_df, product_ids)
        self._ui("on_init_phase_complete", 1)

        # Phase 2: Train anomaly detector + optimizer
        self._ui("on_init_phase_start", 2)
        from .models.anomaly_detector import AnomalyDetector
        from .models.optimizer import HybridOptimizer

        _runtime.anomaly_detector = AnomalyDetector()
        _runtime.anomaly_detector.train(demand_df)
        _runtime.optimizer = HybridOptimizer()
        log.info("models_trained")
        self._ui("on_init_phase_complete", 2)

        # Phase 3: Initialize engines
        self._ui("on_init_phase_start", 3)
        from .events.bus import EventBus
        from .events.monitor import ProactiveMonitor
        from .kpi.engine import KPIEngine

        _runtime.kpi_engine = KPIEngine()
        _runtime.event_bus = EventBus()
        _runtime.monitor = ProactiveMonitor(
            _runtime.event_bus, _runtime.kpi_engine, _runtime.anomaly_detector
        )

        # Register UI event handler if present
        if self._ui_callback:
            async def _ui_event_handler(event: Any) -> None:
                self._ui_callback.on_event(
                    event_type=event.event_type,
                    severity=event.severity.value,
                    source=event.source_agent,
                    description=event.description,
                )
            _runtime.event_bus.subscribe_all(_ui_event_handler)

        self._ui("on_init_phase_complete", 3)

        # Phase 4: Initialize agents
        self._ui("on_init_phase_start", 4)
        from .agents import (
            AnomalyDetectorAgent,
            CoordinatorAgent,
            DemandForecasterAgent,
            InventoryOptimizerAgent,
            LogisticsCoordinatorAgent,
            MarketIntelligenceAgent,
            ReporterAgent,
            RiskAssessorAgent,
            StrategicPlannerAgent,
            SupplierManagerAgent,
        )
        from .llm.factory import create_llm
        from .tools import (
            AdjustSafetyStock,
            AssessSupplyRisk,
            CalculateReorderPoint,
            CreatePurchaseOrder,
            DetectAnomalies,
            EmitEvent,
            EvaluateSupplier,
            GetForecastAccuracy,
            OptimizeInventory,
            QueryDemandHistory,
            QueryInventoryStatus,
            QueryKPIHistory,
            QuerySupplierInfo,
            RequestHumanApproval,
            RunDemandForecast,
            ScanMarketIntelligence,
        )

        llm = create_llm()
        log.info("llm_created", mode=settings.llm_mode.value)

        _runtime.agents = {
            "demand_forecaster": DemandForecasterAgent(
                llm=llm,
                tools=[
                    QueryDemandHistory(), RunDemandForecast(), GetForecastAccuracy(),
                    ScanMarketIntelligence(), EmitEvent(),
                ],
            ),
            "strategic_planner": StrategicPlannerAgent(
                llm=llm,
                tools=[QueryKPIHistory(), OptimizeInventory(), QueryInventoryStatus(), EmitEvent()],
            ),
            "inventory_optimizer": InventoryOptimizerAgent(
                llm=llm,
                tools=[
                    QueryInventoryStatus(), CalculateReorderPoint(), AdjustSafetyStock(),
                    OptimizeInventory(), EmitEvent(),
                ],
            ),
            "supplier_manager": SupplierManagerAgent(
                llm=llm,
                tools=[
                    QuerySupplierInfo(), EvaluateSupplier(), CreatePurchaseOrder(),
                    RequestHumanApproval(), EmitEvent(),
                ],
            ),
            "logistics_coordinator": LogisticsCoordinatorAgent(
                llm=llm,
                tools=[QueryInventoryStatus(), EmitEvent()],
            ),
            "anomaly_detector": AnomalyDetectorAgent(
                llm=llm,
                tools=[DetectAnomalies(), QueryDemandHistory(), QueryInventoryStatus(), EmitEvent()],
            ),
            "risk_assessor": RiskAssessorAgent(
                llm=llm,
                tools=[AssessSupplyRisk(), ScanMarketIntelligence(), QuerySupplierInfo(), EmitEvent()],
            ),
            "market_intelligence": MarketIntelligenceAgent(
                llm=llm,
                tools=[ScanMarketIntelligence(), EmitEvent()],
            ),
            "coordinator": CoordinatorAgent(
                llm=llm,
                tools=[
                    QueryKPIHistory(), QueryInventoryStatus(), QuerySupplierInfo(),
                    QueryDemandHistory(), RequestHumanApproval(), EmitEvent(),
                ],
            ),
            "reporter": ReporterAgent(
                llm=llm,
                tools=[QueryKPIHistory(), QueryInventoryStatus(), EmitEvent()],
            ),
        }

        # Set up event subscriptions
        bus = _runtime.event_bus
        agents = _runtime.agents

        bus.subscribe("kpi_threshold_violated", agents["demand_forecaster"].handle_event)
        bus.subscribe("new_market_intel", agents["demand_forecaster"].handle_event)
        bus.subscribe("forecast_updated", agents["strategic_planner"].handle_event)
        bus.subscribe("kpi_trend_alert", agents["strategic_planner"].handle_event)
        bus.subscribe("low_stock_alert", agents["inventory_optimizer"].handle_event)
        bus.subscribe("overstock_alert", agents["inventory_optimizer"].handle_event)
        bus.subscribe("stockout_alert", agents["inventory_optimizer"].handle_event)
        bus.subscribe("forecast_updated", agents["inventory_optimizer"].handle_event)
        bus.subscribe("reorder_triggered", agents["supplier_manager"].handle_event)
        bus.subscribe("supplier_issue", agents["supplier_manager"].handle_event)
        bus.subscribe("quality_alert", agents["supplier_manager"].handle_event)
        bus.subscribe("po_created", agents["logistics_coordinator"].handle_event)
        bus.subscribe("delivery_delayed", agents["logistics_coordinator"].handle_event)
        bus.subscribe("anomaly_detected", agents["risk_assessor"].handle_event)
        bus.subscribe("supply_risk_alert", agents["risk_assessor"].handle_event)
        bus.subscribe("cycle_complete", agents["reporter"].handle_event)
        bus.subscribe("kpi_snapshot_created", agents["reporter"].handle_event)

        # Coordinator listens to everything
        bus.subscribe_all(agents["coordinator"].handle_event)

        log.info("agents_initialized", count=len(_runtime.agents))
        self._ui("on_init_phase_complete", 4)

        # Phase 5: Compute initial KPI snapshot
        self._ui("on_init_phase_start", 5)
        _runtime.kpi_engine.calculate_snapshot(
            products, _runtime.purchase_orders, suppliers
        )

        log.info("system_ready")
        self._ui("on_init_phase_complete", 5)

        # Phase 6: Initialize persistence backend
        from .aws import get_backend

        _runtime.backend = get_backend()
        await _runtime.backend.setup()
        if _runtime.demand_df is not None:
            await _runtime.backend.persist_demand_history(_runtime.demand_df)
        log.info("persistence_backend_ready", backend=type(_runtime.backend).__name__)

    async def run_cycle(self) -> Dict[str, Any]:
        """Execute one full decision cycle across all agent layers."""
        self._cycle_count += 1
        log.info("cycle_start", cycle=self._cycle_count)

        products = _runtime.products or []
        context = {"products": products, "cycle": self._cycle_count}
        agent_results: Dict[str, Any] = {}

        agents = _runtime.agents

        # Step 0: Operational layer — Market Intelligence + Anomaly Detection
        self._ui("on_cycle_step_start", 0)
        log.info("cycle_step", step=1, description="Operational layer scan")
        market_result = await agents["market_intelligence"].run_cycle(context)
        anomaly_result = await agents["anomaly_detector"].run_cycle(context)
        agent_results["market_intelligence"] = market_result
        agent_results["anomaly_detector"] = anomaly_result
        self._ui("on_cycle_step_complete", 0)

        # Step 1: Strategic layer — Demand Forecasting
        self._ui("on_cycle_step_start", 1)
        log.info("cycle_step", step=2, description="Strategic forecasting")
        forecast_result = await agents["demand_forecaster"].run_cycle(context)
        agent_results["demand_forecaster"] = forecast_result
        self._ui("on_cycle_step_complete", 1)

        # Step 2: Tactical + Operational — Inventory + Risk
        self._ui("on_cycle_step_start", 2)
        log.info("cycle_step", step=3, description="Inventory check + Risk assessment")
        inv_result = await agents["inventory_optimizer"].run_cycle(context)
        risk_result = await agents["risk_assessor"].run_cycle(context)
        agent_results["inventory_optimizer"] = inv_result
        agent_results["risk_assessor"] = risk_result
        self._ui("on_cycle_step_complete", 2)

        # Step 3: Tactical — Supplier selection + Procurement
        self._ui("on_cycle_step_start", 3)
        log.info("cycle_step", step=4, description="Supplier management")
        supplier_result = await agents["supplier_manager"].run_cycle(context)
        agent_results["supplier_manager"] = supplier_result
        self._ui("on_cycle_step_complete", 3)

        # Step 4: Tactical — Logistics coordination
        self._ui("on_cycle_step_start", 4)
        log.info("cycle_step", step=5, description="Logistics coordination")
        logistics_result = await agents["logistics_coordinator"].run_cycle(context)
        agent_results["logistics_coordinator"] = logistics_result
        self._ui("on_cycle_step_complete", 4)

        # Step 5: Strategic — Strategic planning
        self._ui("on_cycle_step_start", 5)
        log.info("cycle_step", step=6, description="Strategic planning")
        planner_result = await agents["strategic_planner"].run_cycle(context)
        agent_results["strategic_planner"] = planner_result
        self._ui("on_cycle_step_complete", 5)

        # Step 6: Orchestration — Coordinator summarizes + resolves conflicts
        self._ui("on_cycle_step_start", 6)
        log.info("cycle_step", step=7, description="Coordinator arbitration")
        coord_context = {**context, "agent_results": agent_results}
        coord_result = await agents["coordinator"].run_cycle(coord_context)
        agent_results["coordinator"] = coord_result
        self._ui("on_cycle_step_complete", 6)

        # Step 7: Orchestration — Reporter generates summary
        self._ui("on_cycle_step_start", 7)
        log.info("cycle_step", step=8, description="Report generation")
        report_context = {
            **context,
            "agent_results": agent_results,
            "coordinator_summary": coord_result.get("executive_summary", ""),
        }
        report_result = await agents["reporter"].run_cycle(report_context)
        agent_results["reporter"] = report_result
        self._ui("on_cycle_step_complete", 7)

        # Step 9: KPI update (stored in kpi_engine.history automatically)
        snapshot = _runtime.kpi_engine.calculate_snapshot(
            products, _runtime.purchase_orders, _runtime.suppliers or [],
        )
        violations = _runtime.kpi_engine.check_thresholds(snapshot)
        for event in violations:
            if _runtime.event_bus:
                await _runtime.event_bus.publish(event)

        # Persist cycle data to backend
        if _runtime.backend:
            await _runtime.backend.persist_cycle(
                cycle=self._cycle_count,
                kpi=snapshot,
                events=list(_runtime.event_bus.recent_events[-50:]) if _runtime.event_bus else [],
                pos=_runtime.purchase_orders,
                products=products,
                suppliers=_runtime.suppliers or [],
            )

        # Simulate demand consumption
        for p in products:
            consumed = max(0, random.gauss(p.daily_demand_avg, p.daily_demand_std))
            p.current_stock = max(0, p.current_stock - consumed)

        log.info(
            "cycle_complete",
            cycle=self._cycle_count,
            agents_run=len(agent_results),
            kpi_violations=len(violations),
        )

        return {
            "cycle": self._cycle_count,
            "agent_results": {
                k: v.get("analysis", "") if isinstance(v, dict) else ""
                for k, v in agent_results.items()
            },
            "kpi": snapshot.model_dump(),
            "violations": len(violations),
            "report": report_result.get("report", {}).get("report_id"),
        }

    async def run_loop(self) -> None:
        """Run continuous simulation cycles."""
        self._running = True

        # Start proactive monitor
        if _runtime.monitor:
            await _runtime.monitor.start()

        log.info("simulation_loop_started")
        while self._running:
            try:
                await self.run_cycle()
                interval = settings.event_tick_seconds * 2 / max(settings.simulation_speed, 0.1)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                log.error("cycle_error", error=str(exc), exc_type=type(exc).__name__)
                await asyncio.sleep(5)
            except Exception as exc:
                log.error("cycle_unexpected_error", error=str(exc), exc_type=type(exc).__name__)
                await asyncio.sleep(5)

    async def stop_loop(self) -> None:
        """Stop the simulation loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        log.info("simulation_loop_stopped")

    async def run_demo(self) -> Dict[str, Any]:
        """Run a single demo cycle and print results."""
        await self.initialize()
        result = await self.run_cycle()
        await self.shutdown()
        return result

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self._running = False
        if _runtime.backend:
            await _runtime.backend.teardown()
        if _runtime.monitor:
            await _runtime.monitor.stop()
        if _runtime.event_bus:
            await _runtime.event_bus.stop()
        log.info("system_shutdown")


# ── Singleton ───────────────────────────────────────────────

_orchestrator: Optional[ChainCommandOrchestrator] = None


def get_orchestrator(mode: Optional[str] = None) -> Any:
    """Factory: return classic (default) or langgraph orchestrator.

    Args:
        mode: "classic" or "langgraph". Defaults to settings.orchestrator_mode.
    """
    global _orchestrator
    chosen = mode or settings.orchestrator_mode

    if chosen == "langgraph":
        from .langgraph_orchestrator import LangGraphOrchestrator
        return LangGraphOrchestrator()

    if _orchestrator is None:
        _orchestrator = ChainCommandOrchestrator()
    return _orchestrator
