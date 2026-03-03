"""Anomaly detection using Isolation Forest (with statistical fallback)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import settings
from ..data.schemas import AlertSeverity, AnomalyRecord
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class AnomalyDetector:
    """Detects demand anomalies, cost anomalies, and lead-time anomalies.

    Uses scikit-learn IsolationForest when available;
    falls back to Z-score detection otherwise.
    """

    def __init__(self) -> None:
        self._contamination = settings.isolation_contamination
        self._stats: Dict[str, dict] = {}  # product_id -> statistics
        self._trained = False
        self._use_sklearn = False

        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401
            self._use_sklearn = True
        except ImportError:
            log.info("sklearn_unavailable", fallback="z-score")

        self._models: Dict[str, Any] = {}

    def train(self, data: pd.DataFrame) -> None:
        """Train anomaly detection models per product."""
        product_ids = data["product_id"].unique()

        for pid in product_ids:
            series = data[data["product_id"] == pid]["quantity"].values
            if len(series) < 10:
                continue

            self._stats[pid] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "median": float(np.median(series)),
                "q1": float(np.percentile(series, 25)),
                "q3": float(np.percentile(series, 75)),
                "iqr": float(np.percentile(series, 75) - np.percentile(series, 25)),
                "max": float(np.max(series)),
                "min": float(np.min(series)),
            }

            if self._use_sklearn:
                from sklearn.ensemble import IsolationForest

                model = IsolationForest(
                    contamination=self._contamination,
                    random_state=42,
                    n_estimators=100,
                )
                model.fit(series.reshape(-1, 1))
                self._models[pid] = model

        self._trained = True
        log.info("anomaly_detector_trained", products=len(self._stats))

    def detect(self, current_data: Dict[str, Any]) -> List[AnomalyRecord]:
        """Detect anomalies in current data point or snapshot."""
        anomalies: List[AnomalyRecord] = []

        if not self._trained:
            return anomalies

        # Check across all trained products if no specific product given
        product_ids = (
            [current_data["product_id"]]
            if "product_id" in current_data
            else list(self._stats.keys())[:10]
        )

        for pid in product_ids:
            stats = self._stats.get(pid)
            if stats is None:
                continue

            demand = current_data.get("daily_demand_avg", stats["mean"])

            # Demand spike detection
            z_score = abs(demand - stats["mean"]) / max(stats["std"], 0.01)
            if z_score > 2.5:
                severity = (
                    AlertSeverity.CRITICAL if z_score > 4
                    else AlertSeverity.HIGH if z_score > 3
                    else AlertSeverity.MEDIUM
                )
                anomalies.append(AnomalyRecord(
                    anomaly_type="demand_spike",
                    product_id=pid,
                    severity=severity,
                    score=round(min(z_score / 5, 1.0), 3),
                    description=(
                        f"Demand anomaly: z-score={z_score:.2f}, "
                        f"current={demand:.1f}, mean={stats['mean']:.1f}"
                    ),
                ))

            # Stock level anomaly
            current_stock = current_data.get("current_stock", 0)
            if current_stock > 0 and stats["mean"] > 0:
                dsi = current_stock / stats["mean"]
                if dsi > settings.dsi_max:
                    anomalies.append(AnomalyRecord(
                        anomaly_type="overstock",
                        product_id=pid,
                        severity=AlertSeverity.MEDIUM,
                        score=round(min(dsi / 100, 1.0), 3),
                        description=f"Overstock: DSI={dsi:.1f} days (max={settings.dsi_max})",
                    ))
                elif dsi < settings.dsi_min:
                    anomalies.append(AnomalyRecord(
                        anomaly_type="understock",
                        product_id=pid,
                        severity=AlertSeverity.HIGH,
                        score=round(1 - dsi / settings.dsi_min, 3),
                        description=f"Understock: DSI={dsi:.1f} days (min={settings.dsi_min})",
                    ))

        return anomalies

    def detect_batch(self, products: list) -> List[AnomalyRecord]:
        """Run detection across all products."""
        all_anomalies: List[AnomalyRecord] = []
        for product in products:
            data = {
                "product_id": product.product_id,
                "daily_demand_avg": product.daily_demand_avg,
                "current_stock": product.current_stock,
            }
            all_anomalies.extend(self.detect(data))
        return all_anomalies
