"""DoWhy causal analysis for supplier switching decisions (Section 4.4)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..utils.logging_config import get_logger

log = get_logger(__name__)


class CausalResult(BaseModel):
    """Result of causal analysis."""

    ate: float = 0.0  # Average Treatment Effect
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0
    method: str = "ipw"
    refutations: Dict[str, Any] = Field(default_factory=dict)
    is_significant: bool = False


class SupplierSwitchCausalAnalysis:
    """DoWhy 4-step causal analysis for supplier switching.

    DAG:
        initial_quality_score ──┐
        disruption_severity ────┤
        alternative_count ──────┼──→ switched_supplier ──→ total_cost_delta
        product_criticality ────┘

    Falls back to manual IPW if dowhy is not installed.
    """

    def __init__(self) -> None:
        self._has_dowhy = False
        try:
            import dowhy  # noqa: F401
            self._has_dowhy = True
        except ImportError:
            log.info("dowhy_unavailable", fallback="manual_ipw")

    def analyze(
        self,
        data: pd.DataFrame,
        treatment: str = "switched_supplier",
        outcome: str = "total_cost_delta",
        confounders: Optional[List[str]] = None,
    ) -> CausalResult:
        """Run 4-step causal analysis: Model → Identify → Estimate → Refute."""
        if confounders is None:
            confounders = [
                "initial_quality_score",
                "disruption_severity",
                "alternative_count",
                "product_criticality",
            ]

        if self._has_dowhy:
            return self._analyze_dowhy(data, treatment, outcome, confounders)
        return self._analyze_ipw_fallback(data, treatment, outcome, confounders)

    def _analyze_dowhy(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> CausalResult:
        """Full DoWhy pipeline."""
        import dowhy

        # Step 1: Model — define causal DAG
        model = dowhy.CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
        )

        # Step 2: Identify — find estimand
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Step 3: Estimate — compute ATE
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting",
        )

        ate = float(estimate.value)
        ci = getattr(estimate, "get_confidence_intervals", lambda: None)()
        ci_lower = float(ci[0]) if ci is not None else ate - abs(ate) * 0.5
        ci_upper = float(ci[1]) if ci is not None else ate + abs(ate) * 0.5

        # Step 4: Refute — 3 robustness checks
        refutations = {}

        try:
            ref1 = model.refute_estimate(
                identified_estimand, estimate,
                method_name="random_common_cause",
            )
            refutations["random_common_cause"] = {
                "new_effect": float(ref1.new_effect),
                "p_value": float(getattr(ref1, "refutation_result", {}).get("p_value", 1.0))
                if isinstance(getattr(ref1, "refutation_result", None), dict) else 1.0,
            }
        except Exception as exc:
            refutations["random_common_cause"] = {"error": str(exc)}

        try:
            ref2 = model.refute_estimate(
                identified_estimand, estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
            )
            refutations["placebo_treatment"] = {
                "new_effect": float(ref2.new_effect),
            }
        except Exception as exc:
            refutations["placebo_treatment"] = {"error": str(exc)}

        try:
            ref3 = model.refute_estimate(
                identified_estimand, estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
            )
            refutations["data_subset"] = {
                "new_effect": float(ref3.new_effect),
            }
        except Exception as exc:
            refutations["data_subset"] = {"error": str(exc)}

        return CausalResult(
            ate=round(ate, 2),
            ci_lower=round(ci_lower, 2),
            ci_upper=round(ci_upper, 2),
            method="dowhy_ipw",
            refutations=refutations,
            is_significant=(ci_lower < 0 < ci_upper) is False and ate != 0,
        )

    def _analyze_ipw_fallback(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> CausalResult:
        """Manual IPW (Inverse Propensity Weighting) fallback."""
        from sklearn.linear_model import LogisticRegression

        X = data[confounders].values
        T = data[treatment].values.astype(int)
        Y = data[outcome].values

        # Fit propensity score model
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        propensity = ps_model.predict_proba(X)[:, 1]

        # Clip propensity to avoid extreme weights
        propensity = np.clip(propensity, 0.05, 0.95)

        # IPW estimator
        weights_treated = T / propensity
        weights_control = (1 - T) / (1 - propensity)

        n = len(Y)
        ate_treated = np.sum(weights_treated * Y) / np.sum(weights_treated)
        ate_control = np.sum(weights_control * Y) / np.sum(weights_control)
        ate = float(ate_treated - ate_control)

        # Bootstrap CI
        bootstrap_ates = []
        rng = np.random.RandomState(42)
        for _ in range(200):
            idx = rng.choice(n, size=n, replace=True)
            b_T, b_Y, b_ps = T[idx], Y[idx], propensity[idx]
            b_wt = b_T / b_ps
            b_wc = (1 - b_T) / (1 - b_ps)
            b_ate = np.sum(b_wt * b_Y) / np.sum(b_wt) - np.sum(b_wc * b_Y) / np.sum(b_wc)
            bootstrap_ates.append(b_ate)

        ci_lower = float(np.percentile(bootstrap_ates, 2.5))
        ci_upper = float(np.percentile(bootstrap_ates, 97.5))

        # Refutation: placebo test (permute treatment)
        rng2 = np.random.RandomState(123)
        permuted_T = rng2.permutation(T)
        pw = permuted_T / propensity
        pc = (1 - permuted_T) / (1 - propensity)
        placebo_ate = float(
            np.sum(pw * Y) / max(np.sum(pw), 1e-10) - np.sum(pc * Y) / max(np.sum(pc), 1e-10)
        )

        refutations = {
            "placebo_treatment": {"new_effect": round(placebo_ate, 2)},
            "method": "manual_ipw_bootstrap",
        }

        return CausalResult(
            ate=round(ate, 2),
            ci_lower=round(ci_lower, 2),
            ci_upper=round(ci_upper, 2),
            method="manual_ipw",
            refutations=refutations,
            is_significant=(ci_lower > 0 or ci_upper < 0),
        )
