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
    """Mutable global state shared across modules and API."""

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

    # New modules
    bom_manager: Any = None
    rl_policy: Any = None
    risk_scorer: Any = None
    ctb_analyzer: Any = None

    # Transaction state
    purchase_orders: List[PurchaseOrder] = field(default_factory=list)
    pending_approvals: Dict[str, HumanApprovalRequest] = field(default_factory=dict)
    kpi_history: List[KPISnapshot] = field(default_factory=list)

    # Results cache
    last_cycle_results: Dict[str, Any] = field(default_factory=dict)

    # Persistence backend
    backend: Any = None


_runtime = _RuntimeState()


# ── Orchestrator ────────────────────────────────────────────

class ChainCommandOrchestrator:
    """Main orchestrator: initializes modules, runs optimization cycles."""

    STAGES = ["data", "ml", "engines", "bom", "rl", "risk", "cpsat", "ctb", "kpi"]

    def __init__(self, on_progress: Any = None) -> None:
        self._running = False
        self._cycle_count = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._on_progress = on_progress or (lambda *a, **kw: None)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    async def initialize(self) -> None:
        """Bootstrap the entire system."""
        setup_logging()
        random.seed(settings.random_seed)
        log.info("initializing")

        # Phase 0: Generate synthetic data
        self._on_progress("data", "running", {})
        from .data.generator import generate_all

        products, suppliers, demand_df = generate_all()
        _runtime.products = products
        _runtime.suppliers = suppliers
        _runtime.demand_df = demand_df
        log.info("data_generated", products=len(products), suppliers=len(suppliers))
        self._on_progress("data", "completed", {"products": len(products), "suppliers": len(suppliers)})

        # Phase 1: Train ML models
        self._on_progress("ml", "running", {})
        from .models.anomaly_detector import AnomalyDetector
        from .models.forecaster import EnsembleForecaster
        from .models.optimizer import HybridOptimizer

        _runtime.forecaster = EnsembleForecaster()
        product_ids = [p.product_id for p in products[:20]]
        _runtime.forecaster.train_all(demand_df, product_ids)
        _runtime.anomaly_detector = AnomalyDetector()
        _runtime.anomaly_detector.train(demand_df)
        _runtime.optimizer = HybridOptimizer()
        log.info("ml_models_trained")
        self._on_progress("ml", "completed", {})

        # Phase 2: Initialize engines
        self._on_progress("engines", "running", {})
        from .events.bus import EventBus
        from .events.monitor import ProactiveMonitor
        from .kpi.engine import KPIEngine

        _runtime.kpi_engine = KPIEngine()
        _runtime.event_bus = EventBus()
        _runtime.monitor = ProactiveMonitor(
            _runtime.event_bus, _runtime.kpi_engine, _runtime.anomaly_detector
        )
        self._on_progress("engines", "completed", {})

        # Phase 3: BOM Management
        self._on_progress("bom", "running", {})
        from .bom import BOMManager

        _runtime.bom_manager = BOMManager()
        _runtime.bom_manager.generate_synthetic_boms(
            n_assemblies=settings.bom_default_assemblies,
            seed=settings.random_seed,
        )
        bom_summary = _runtime.bom_manager.get_summary()
        log.info("bom_initialized", **bom_summary)
        self._on_progress("bom", "completed", bom_summary)

        # Phase 4: RL Inventory Policy
        self._on_progress("rl", "running", {})
        from .rl import RLInventoryPolicy
        from .rl.environment import InventoryEnvConfig

        avg_demand = float(demand_df["quantity"].mean()) if "quantity" in demand_df.columns else 100.0
        rl_config = InventoryEnvConfig(
            demand_mean=avg_demand,
            demand_std=avg_demand * 0.3,
            episode_length=settings.rl_episode_length,
            holding_cost_per_unit=settings.rl_holding_cost,
            stockout_cost_per_unit=settings.rl_stockout_cost,
            ordering_cost_fixed=settings.rl_ordering_cost_fixed,
        )
        _runtime.rl_policy = RLInventoryPolicy(rl_config)
        rl_result = _runtime.rl_policy.train(
            total_timesteps=settings.rl_total_timesteps,
            seed=settings.random_seed,
        )
        log.info(
            "rl_trained",
            method=rl_result.method,
            improvement_pct=rl_result.improvement_pct,
        )
        self._on_progress("rl", "completed", {
            "method": rl_result.method,
            "mean_reward": rl_result.mean_reward,
            "improvement_pct": rl_result.improvement_pct,
        })

        # Phase 5: Risk Scoring
        self._on_progress("risk", "running", {})
        from .risk import SupplierRiskScorer

        _runtime.risk_scorer = SupplierRiskScorer()
        # Train ML risk model on synthetic history
        history = _runtime.risk_scorer.generate_synthetic_history(n_suppliers=100, seed=settings.random_seed)
        _runtime.risk_scorer.train_ml_model(history, seed=settings.random_seed)
        log.info("risk_scorer_initialized")
        self._on_progress("risk", "completed", {})

        # Phase 6: Initial KPI snapshot
        self._on_progress("kpi", "running", {})
        _runtime.kpi_engine.calculate_snapshot(products, _runtime.purchase_orders, suppliers)
        log.info("initial_kpi_computed")
        self._on_progress("kpi", "completed", {})

        # Phase 7: AWS backend
        from .aws import get_backend

        _runtime.backend = get_backend()
        await _runtime.backend.setup()
        if _runtime.demand_df is not None:
            await _runtime.backend.persist_demand_history(_runtime.demand_df)

        log.info("system_ready")

    async def run_cycle(self) -> Dict[str, Any]:
        """Execute one optimization cycle."""
        self._cycle_count += 1
        log.info("cycle_start", cycle=self._cycle_count)

        products = _runtime.products or []
        suppliers = _runtime.suppliers or []
        results: Dict[str, Any] = {"cycle": self._cycle_count}

        # Step 1: Anomaly detection
        if _runtime.anomaly_detector and _runtime.demand_df is not None:
            anomalies = _runtime.anomaly_detector.detect(products[:10])
            results["anomalies"] = len(anomalies) if anomalies else 0

        # Step 2: Risk scoring for suppliers
        if _runtime.risk_scorer and suppliers:
            from .risk.scorer import SupplierMetrics

            risk_scores = []
            for s in suppliers[:10]:
                metrics = SupplierMetrics(
                    supplier_id=s.supplier_id,
                    on_time_rate=s.on_time_rate,
                    defect_rate=s.defect_rate,
                    lead_time_mean=s.lead_time_mean,
                    lead_time_std=s.lead_time_std,
                )
                score = _runtime.risk_scorer.score_supplier(metrics)
                risk_scores.append(score)

            high_risk = sum(1 for r in risk_scores if r.risk_level in ("high", "critical"))
            results["risk"] = {
                "scored": len(risk_scores),
                "high_risk_count": high_risk,
            }

        # Step 3: CP-SAT supplier allocation for low-stock products
        low_stock = [p for p in products if p.current_stock < p.reorder_point]
        if low_stock:
            from .optimization.cpsat_optimizer import SupplierAllocationOptimizer, SupplierCandidate

            allocator = SupplierAllocationOptimizer()
            for product in low_stock[:5]:
                candidates = []
                for s in suppliers:
                    if product.product_id in s.products:
                        candidates.append(SupplierCandidate(
                            supplier_id=s.supplier_id,
                            unit_cost=product.unit_cost * s.cost_multiplier,
                            risk_score=1.0 - s.reliability_score,
                            capacity=s.capacity,
                            min_order_qty=float(product.min_order_qty),
                            lead_time_days=s.lead_time_mean,
                        ))
                if candidates:
                    alloc = allocator.optimize(candidates, product.daily_demand_avg * 30)
                    results.setdefault("allocations", []).append({
                        "product_id": product.product_id,
                        "status": alloc.solver_status,
                        "total_cost": alloc.total_cost,
                    })

        # Step 4: RL inventory decisions
        if _runtime.rl_policy:
            rl_decisions = []
            for p in products[:10]:
                decision = _runtime.rl_policy.decide(
                    current_stock=p.current_stock,
                    avg_demand=p.daily_demand_avg,
                )
                rl_decisions.append({
                    "product_id": p.product_id,
                    "action": decision.action,
                    "order_qty": decision.order_quantity,
                    "method": decision.method,
                })
            results["rl_decisions"] = len(rl_decisions)

        # Step 5: CTB analysis
        if _runtime.bom_manager and _runtime.ctb_analyzer is None:
            from .ctb import CTBAnalyzer
            _runtime.ctb_analyzer = CTBAnalyzer()

        if _runtime.ctb_analyzer and _runtime.bom_manager:
            ctb_reports = []
            for assembly_id, tree in list(_runtime.bom_manager.assemblies.items())[:3]:
                # Build inventory from current product stock
                inventory = {p.product_id: p.current_stock for p in products}
                roots = tree.root_items
                for root in roots:
                    report = _runtime.ctb_analyzer.analyze(
                        tree, root.part_id, settings.ctb_default_build_qty, inventory,
                    )
                    ctb_reports.append({
                        "assembly_id": assembly_id,
                        "is_clear": report.is_clear,
                        "clear_pct": report.clear_percentage,
                        "shortages": len(report.shortages),
                    })
            results["ctb"] = ctb_reports

        # Step 6: KPI update
        snapshot = _runtime.kpi_engine.calculate_snapshot(
            products, _runtime.purchase_orders, suppliers,
        )
        violations = _runtime.kpi_engine.check_thresholds(snapshot)
        for event in violations:
            if _runtime.event_bus:
                await _runtime.event_bus.publish(event)

        # Persist cycle data
        if _runtime.backend:
            await _runtime.backend.persist_cycle(
                cycle=self._cycle_count,
                kpi=snapshot,
                events=list(_runtime.event_bus.recent_events[-50:]) if _runtime.event_bus else [],
                pos=_runtime.purchase_orders,
                products=products,
                suppliers=suppliers,
            )

        # Simulate demand consumption
        for p in products:
            consumed = max(0, random.gauss(p.daily_demand_avg, p.daily_demand_std))
            p.current_stock = max(0, p.current_stock - consumed)

        results["kpi"] = snapshot.model_dump()
        results["violations"] = len(violations)
        _runtime.last_cycle_results = results

        log.info("cycle_complete", cycle=self._cycle_count, violations=len(violations))
        return results

    async def run_loop(self) -> None:
        """Run continuous optimization cycles."""
        self._running = True
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
            except Exception as exc:
                log.error("cycle_error", error=str(exc), exc_type=type(exc).__name__)
                await asyncio.sleep(5)

    async def stop_loop(self) -> None:
        """Stop the simulation loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        log.info("simulation_loop_stopped")

    async def run_demo(self) -> Dict[str, Any]:
        """Run a single demo cycle and return results."""
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


def get_orchestrator() -> ChainCommandOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChainCommandOrchestrator()
    return _orchestrator
