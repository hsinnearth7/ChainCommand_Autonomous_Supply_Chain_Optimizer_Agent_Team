"""Benchmark CP-SAT vs GA optimizer on same problem instances."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..data.schemas import ForecastResult, Product
from ..models.optimizer import GeneticOptimizer
from ..utils.logging_config import get_logger
from .cpsat_optimizer import SupplierAllocationOptimizer, SupplierCandidate

log = get_logger(__name__)


class OptimizerBenchmark:
    """Run CP-SAT vs GA on the same instances and compare."""

    def __init__(self) -> None:
        self._cpsat = SupplierAllocationOptimizer()
        self._ga = GeneticOptimizer()

    def run(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        product: Product,
        forecast: List[ForecastResult] | None = None,
    ) -> Dict[str, Any]:
        """Compare both optimizers on the same problem."""
        forecast = forecast or []

        # CP-SAT
        t0 = time.monotonic()
        cpsat_result = self._cpsat.optimize(candidates, demand)
        cpsat_ms = (time.monotonic() - t0) * 1000

        # GA
        t0 = time.monotonic()
        ga_result = self._ga.optimize(product, forecast)
        ga_ms = (time.monotonic() - t0) * 1000

        # Compute optimality gap if both have costs
        gap = 0.0
        if cpsat_result.total_cost > 0 and ga_result.expected_cost_saving >= 0:
            gap = abs(cpsat_result.total_cost - ga_result.expected_cost_saving) / max(cpsat_result.total_cost, 1)

        report = {
            "cpsat": {
                "cost": cpsat_result.total_cost,
                "risk": cpsat_result.total_risk,
                "status": cpsat_result.solver_status,
                "time_ms": round(cpsat_ms, 1),
                "suppliers_used": len(cpsat_result.allocations),
            },
            "ga": {
                "reorder_point": ga_result.recommended_reorder_point,
                "safety_stock": ga_result.recommended_safety_stock,
                "order_qty": ga_result.recommended_order_qty,
                "saving": ga_result.expected_cost_saving,
                "time_ms": round(ga_ms, 1),
            },
            "optimality_gap": round(gap, 4),
        }

        log.info("benchmark_complete", cpsat_ms=cpsat_ms, ga_ms=ga_ms, gap=gap)
        return report
