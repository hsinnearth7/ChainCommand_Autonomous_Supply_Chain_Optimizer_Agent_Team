"""Property-based tests using Hypothesis."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


if HAS_HYPOTHESIS:

    class TestKPIRanges:
        @given(
            otif=st.floats(min_value=0, max_value=1),
            fill_rate=st.floats(min_value=0, max_value=1),
            mape=st.floats(min_value=0, max_value=100),
            dsi=st.floats(min_value=0, max_value=365),
        )
        @settings(max_examples=50)
        def test_kpi_snapshot_valid_ranges(self, otif, fill_rate, mape, dsi):
            from chaincommand.data.schemas import KPISnapshot

            snap = KPISnapshot(otif=otif, fill_rate=fill_rate, mape=mape, dsi=dsi)
            assert 0 <= snap.otif <= 1
            assert 0 <= snap.fill_rate <= 1
            assert 0 <= snap.mape <= 100
            assert 0 <= snap.dsi <= 365

        @given(
            unit_cost=st.floats(min_value=0.01, max_value=1000),
            risk_score=st.floats(min_value=0, max_value=1),
            capacity=st.floats(min_value=1, max_value=100000),
        )
        @settings(max_examples=30)
        def test_supplier_candidate_valid(self, unit_cost, risk_score, capacity):
            from chaincommand.optimization.cpsat_optimizer import SupplierCandidate

            sc = SupplierCandidate(
                supplier_id="TEST",
                unit_cost=unit_cost,
                risk_score=risk_score,
                capacity=capacity,
            )
            assert sc.unit_cost > 0
            assert 0 <= sc.risk_score <= 1

        @given(demand=st.floats(min_value=100, max_value=10000))
        @settings(max_examples=20)
        def test_optimizer_satisfies_demand(self, demand):
            from chaincommand.optimization.cpsat_optimizer import (
                SupplierAllocationOptimizer,
                SupplierCandidate,
            )

            candidates = [
                SupplierCandidate(
                    supplier_id="S1", unit_cost=10.0, risk_score=0.1, capacity=50000,
                ),
                SupplierCandidate(
                    supplier_id="S2", unit_cost=12.0, risk_score=0.05, capacity=50000,
                ),
            ]
            opt = SupplierAllocationOptimizer()
            result = opt.optimize(candidates, demand=demand)
            total = sum(result.allocations.values())
            assert total >= demand * 0.99  # allow tiny rounding

        @given(tokens=st.integers(min_value=0, max_value=10000))
        @settings(max_examples=20)
        def test_token_budget_non_negative(self, tokens):
            from chaincommand.observability import TokenBudget

            budget = TokenBudget(per_cycle=50000, per_agent=10000)
            budget.consume("agent", tokens)
            assert budget.remaining("agent") >= 0
            assert budget.remaining() >= 0

        @given(n=st.integers(min_value=10, max_value=500))
        @settings(max_examples=10)
        def test_causal_data_generation(self, n):
            from chaincommand.causal.data_generator import generate_supplier_switch_history

            df = generate_supplier_switch_history(n_samples=n)
            assert len(df) == n
            assert set(df["switched_supplier"].unique()).issubset({0, 1})
