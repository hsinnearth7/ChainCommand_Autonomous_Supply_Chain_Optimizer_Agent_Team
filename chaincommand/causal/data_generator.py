"""Synthetic observational data for causal analysis experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_supplier_switch_history(
    n_samples: int = 1000,
    true_ate: float = -5000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic supplier-switch observational data.

    Confounders affect both the treatment (switched_supplier) and
    the outcome (total_cost_delta). The true ATE is injected as a
    known constant so downstream tests can validate the estimator.

    Args:
        n_samples: Number of observations.
        true_ate: Known causal effect of switching supplier on cost.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: initial_quality_score, disruption_severity,
        alternative_count, product_criticality, switched_supplier,
        total_cost_delta.
    """
    rng = np.random.RandomState(seed)

    # Confounders
    initial_quality_score = rng.uniform(0.3, 1.0, n_samples)
    disruption_severity = rng.uniform(0.0, 1.0, n_samples)
    alternative_count = rng.poisson(3, n_samples).astype(float)
    product_criticality = rng.uniform(0.0, 1.0, n_samples)

    # Treatment propensity (affected by confounders)
    logit = (
        -1.0
        - 2.0 * initial_quality_score
        + 1.5 * disruption_severity
        + 0.3 * alternative_count
        + 0.5 * product_criticality
    )
    propensity = 1 / (1 + np.exp(-logit))
    switched_supplier = (rng.uniform(0, 1, n_samples) < propensity).astype(int)

    # Outcome: affected by confounders + treatment
    noise = rng.normal(0, 2000, n_samples)
    total_cost_delta = (
        3000 * disruption_severity
        - 2000 * initial_quality_score
        + 500 * product_criticality
        + true_ate * switched_supplier
        + noise
    )

    return pd.DataFrame({
        "initial_quality_score": initial_quality_score,
        "disruption_severity": disruption_severity,
        "alternative_count": alternative_count,
        "product_criticality": product_criticality,
        "switched_supplier": switched_supplier,
        "total_cost_delta": total_cost_delta,
    })
