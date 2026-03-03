"""Tests for DoWhy causal inference — supplier switching."""

from __future__ import annotations

import pytest

from chaincommand.causal.data_generator import generate_supplier_switch_history
from chaincommand.causal.supplier_switch import CausalResult, SupplierSwitchCausalAnalysis


@pytest.fixture
def switch_data():
    return generate_supplier_switch_history(n_samples=500, true_ate=-5000, seed=42)


@pytest.fixture
def analysis():
    return SupplierSwitchCausalAnalysis()


class TestSyntheticData:
    def test_columns_present(self, switch_data):
        expected = {
            "initial_quality_score",
            "disruption_severity",
            "alternative_count",
            "product_criticality",
            "switched_supplier",
            "total_cost_delta",
        }
        assert expected.issubset(set(switch_data.columns))

    def test_sample_count(self, switch_data):
        assert len(switch_data) == 500

    def test_treatment_binary(self, switch_data):
        assert set(switch_data["switched_supplier"].unique()).issubset({0, 1})

    def test_both_treatment_groups(self, switch_data):
        counts = switch_data["switched_supplier"].value_counts()
        assert counts[0] > 50  # enough controls
        assert counts[1] > 50  # enough treated


class TestCausalAnalysis:
    def test_analyze_returns_result(self, analysis, switch_data):
        result = analysis.analyze(switch_data)
        assert isinstance(result, CausalResult)
        assert result.method in ("dowhy_ipw", "manual_ipw")

    def test_ate_direction(self, analysis, switch_data):
        """ATE should be negative (switching reduces cost)."""
        result = analysis.analyze(switch_data)
        # True ATE is -5000; allow generous tolerance for small sample
        assert result.ate < 0, f"Expected negative ATE, got {result.ate}"

    def test_confidence_interval(self, analysis, switch_data):
        result = analysis.analyze(switch_data)
        assert result.ci_lower <= result.ate <= result.ci_upper

    def test_refutations_present(self, analysis, switch_data):
        result = analysis.analyze(switch_data)
        assert isinstance(result.refutations, dict)
        assert len(result.refutations) > 0

    def test_significance(self, analysis, switch_data):
        result = analysis.analyze(switch_data)
        # With enough data and true ATE=-5000, should be significant
        assert result.is_significant is True

    def test_small_sample_still_works(self, analysis):
        data = generate_supplier_switch_history(n_samples=50, seed=99)
        result = analysis.analyze(data)
        assert isinstance(result, CausalResult)
