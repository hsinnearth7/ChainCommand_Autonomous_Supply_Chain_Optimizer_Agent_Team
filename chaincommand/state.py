"""Structured shared state for LangGraph orchestration (v2.0)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DecisionEntry(BaseModel):
    """Audit-trail record for every agent decision."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    decision: str
    rationale: str = ""
    confidence: float = 0.0  # 0-1
    tokens_used: int = 0


class SupplyChainState(BaseModel):
    """Structured shared state passed through the LangGraph pipeline.

    Each agent writes ONLY its own field → state isolation.
    """

    # ── Cycle metadata ───────────────────────────────────
    cycle: int = 0
    products: List[Any] = Field(default_factory=list)
    suppliers: List[Any] = Field(default_factory=list)

    # ── Per-agent output fields (state isolation) ────────
    market_intel: Dict[str, Any] = Field(default_factory=dict)
    anomaly_results: Dict[str, Any] = Field(default_factory=dict)
    forecast_results: Dict[str, Any] = Field(default_factory=dict)
    inventory_results: Dict[str, Any] = Field(default_factory=dict)
    risk_results: Dict[str, Any] = Field(default_factory=dict)
    supplier_results: Dict[str, Any] = Field(default_factory=dict)
    logistics_results: Dict[str, Any] = Field(default_factory=dict)
    planner_results: Dict[str, Any] = Field(default_factory=dict)
    coordinator_results: Dict[str, Any] = Field(default_factory=dict)
    reporter_results: Dict[str, Any] = Field(default_factory=dict)

    # ── KPI ──────────────────────────────────────────────
    kpi_snapshot: Dict[str, Any] = Field(default_factory=dict)

    # ── HITL ─────────────────────────────────────────────
    requires_human_approval: bool = False
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Audit trail ──────────────────────────────────────
    decisions: List[DecisionEntry] = Field(default_factory=list)

    def add_decision(
        self,
        agent: str,
        decision: str,
        rationale: str = "",
        confidence: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        """Append a decision entry to the audit trail."""
        self.decisions.append(DecisionEntry(
            agent=agent,
            decision=decision,
            rationale=rationale,
            confidence=confidence,
            tokens_used=tokens_used,
        ))
