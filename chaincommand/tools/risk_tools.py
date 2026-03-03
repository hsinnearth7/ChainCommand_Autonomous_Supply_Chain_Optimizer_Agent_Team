"""Risk assessment tools — anomaly detection, supply risk, market intelligence."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base_tool import BaseTool


class DetectAnomalies(BaseTool):
    """Run anomaly detection on current data."""

    name = "detect_anomalies"
    description = "Use Isolation Forest to detect demand, cost, or lead-time anomalies."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..orchestrator import _runtime

        product_id: str = kwargs.get("product_id", "")

        if _runtime.anomaly_detector is None:
            return {"error": "Anomaly detector not initialized"}

        product = next(
            (p for p in (_runtime.products or []) if p.product_id == product_id),
            None,
        )
        if product is None and product_id:
            return {"error": f"Product {product_id} not found"}

        current_data = {}
        if product:
            current_data = {
                "current_stock": product.current_stock,
                "daily_demand_avg": product.daily_demand_avg,
                "daily_demand_std": product.daily_demand_std,
                "reorder_point": product.reorder_point,
            }

        anomalies = _runtime.anomaly_detector.detect(current_data)
        return {
            "product_id": product_id,
            "anomalies": [a.model_dump() for a in anomalies],
            "count": len(anomalies),
        }


class AssessSupplyRisk(BaseTool):
    """Assess supply risk across depth, breadth, and criticality dimensions."""

    name = "assess_supply_risk"
    description = "Quantify supply risk for a product or supplier using multi-dimensional scoring."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..orchestrator import _runtime

        product_id: str = kwargs.get("product_id", "")
        supplier_id: str = kwargs.get("supplier_id", "")

        suppliers = _runtime.suppliers or []

        # Find relevant suppliers
        if product_id:
            relevant_suppliers = [s for s in suppliers if product_id in s.products]
        elif supplier_id:
            relevant_suppliers = [s for s in suppliers if s.supplier_id == supplier_id]
        else:
            relevant_suppliers = suppliers[:5]

        risk_scores = []
        for s in relevant_suppliers:
            # Depth: how reliable is each supplier?
            depth = 1 - s.reliability_score
            # Breadth: concentration risk (fewer suppliers = higher risk)
            breadth = 1.0 / max(len(relevant_suppliers), 1)
            # Criticality: based on defect and on-time rates
            criticality = (s.defect_rate * 0.5) + ((1 - s.on_time_rate) * 0.5)

            overall = depth * 0.4 + breadth * 0.3 + criticality * 0.3
            risk_scores.append({
                "supplier_id": s.supplier_id,
                "name": s.name,
                "depth_risk": round(depth, 3),
                "breadth_risk": round(breadth, 3),
                "criticality_risk": round(criticality, 3),
                "overall_risk": round(overall, 3),
                "level": (
                    "critical" if overall > 0.6
                    else "high" if overall > 0.4
                    else "medium" if overall > 0.2
                    else "low"
                ),
            })

        risk_scores.sort(key=lambda x: x["overall_risk"], reverse=True)
        return {
            "product_id": product_id,
            "supplier_risks": risk_scores,
            "highest_risk": risk_scores[0] if risk_scores else None,
        }


class ScanMarketIntelligence(BaseTool):
    """Scan for market intelligence and emerging trends."""

    name = "scan_market_intelligence"
    description = "Gather market signals — price changes, competitor moves, regulatory updates."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..data.schemas import MarketIntelligence

        # In mock mode, generate simulated market intel
        topics = [
            ("Raw material price increase", "commodity_price", -0.3),
            ("New trade tariff announced", "regulation", -0.5),
            ("Competitor supply shortage", "competitor", 0.4),
            ("Seasonal demand uptick expected", "seasonality", 0.2),
            ("Port congestion easing", "logistics", 0.3),
            ("Currency fluctuation risk", "forex", -0.2),
            ("New supplier market entry", "supplier_market", 0.3),
            ("Weather disruption forecast", "weather", -0.4),
        ]

        selected = random.sample(topics, k=min(3, len(topics)))
        intel_items = []
        for topic, source, base_impact in selected:
            impact = base_impact + random.uniform(-0.1, 0.1)
            intel = MarketIntelligence(
                source=source,
                topic=topic,
                summary=f"Simulated: {topic}. Estimated impact on supply chain operations.",
                impact_score=round(max(-1, min(1, impact)), 2),
            )
            intel_items.append(intel.model_dump())

        return {
            "intel_count": len(intel_items),
            "items": intel_items,
        }
