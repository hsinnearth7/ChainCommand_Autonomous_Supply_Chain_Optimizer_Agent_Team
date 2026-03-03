"""Tool wrapper for the CP-SAT supplier allocation optimizer."""

from __future__ import annotations

from typing import Any, Dict

from ..optimization.cpsat_optimizer import SupplierAllocationOptimizer, SupplierCandidate
from .base_tool import BaseTool


class OptimizeSupplierAllocation(BaseTool):
    """Optimize supplier allocation using CP-SAT MILP solver."""

    name = "OptimizeSupplierAllocation"
    description = (
        "Run CP-SAT MILP optimization for supplier allocation. "
        "Minimizes cost + risk subject to demand, capacity, MOQ, and lead-time constraints."
    )

    def __init__(self) -> None:
        self._optimizer = SupplierAllocationOptimizer()

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        candidates_raw = kwargs.get("candidates", [])
        demand = float(kwargs.get("demand", 1000))
        risk_lambda = kwargs.get("risk_lambda")
        max_suppliers = kwargs.get("max_suppliers")

        candidates = [
            SupplierCandidate(**c) if isinstance(c, dict) else c
            for c in candidates_raw
        ]

        if not candidates:
            return {"error": "No supplier candidates provided"}

        result = self._optimizer.optimize(
            candidates=candidates,
            demand=demand,
            risk_lambda=float(risk_lambda) if risk_lambda is not None else None,
            max_suppliers=int(max_suppliers) if max_suppliers is not None else None,
        )
        return result.model_dump()
