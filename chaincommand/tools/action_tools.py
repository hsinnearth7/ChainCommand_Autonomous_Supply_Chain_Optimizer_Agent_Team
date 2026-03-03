"""Action tools — purchase orders, approvals, safety stock adjustments, events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from ..config import settings
from ..data.schemas import (
    AlertSeverity,
    ApprovalStatus,
    HumanApprovalRequest,
    OrderStatus,
    PurchaseOrder,
    SupplyChainEvent,
)
from .base_tool import BaseTool


def _create_approval(
    request_type: str,
    description: str,
    estimated_cost: float,
    risk_level: AlertSeverity,
    data: dict,
) -> HumanApprovalRequest:
    """Create an approval request and register it in the runtime."""
    from ..orchestrator import _runtime

    approval = HumanApprovalRequest(
        request_type=request_type,
        description=description,
        estimated_cost=estimated_cost,
        risk_level=risk_level,
        data=data,
    )
    _runtime.pending_approvals[approval.request_id] = approval
    return approval


class CreatePurchaseOrder(BaseTool):
    """Create a new purchase order."""

    name = "create_purchase_order"
    description = "Generate a purchase order for a supplier and product."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..orchestrator import _runtime

        supplier_id: str = kwargs["supplier_id"]
        product_id: str = kwargs["product_id"]
        quantity: float = kwargs["quantity"]
        unit_cost: float = kwargs.get("unit_cost", 0.0)

        # Input validation
        if quantity <= 0:
            return {"error": "quantity must be positive"}
        if unit_cost < 0:
            return {"error": "unit_cost must be non-negative"}

        # Look up actual unit cost from product if not given
        if unit_cost == 0.0:
            product = next(
                (p for p in (_runtime.products or []) if p.product_id == product_id),
                None,
            )
            if product:
                unit_cost = product.unit_cost

        # Look up supplier lead time
        supplier = next(
            (s for s in (_runtime.suppliers or []) if s.supplier_id == supplier_id),
            None,
        )
        lead_days = supplier.lead_time_mean if supplier else 7

        total_cost = quantity * unit_cost
        po = PurchaseOrder(
            supplier_id=supplier_id,
            product_id=product_id,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            expected_delivery=datetime.now(timezone.utc) + timedelta(days=int(lead_days)),
        )

        # HITL gate: auto-approve below threshold, escalate above
        if total_cost < settings.auto_approve_below:
            po.approval_status = ApprovalStatus.AUTO_APPROVED
            po.status = OrderStatus.APPROVED
            po.approved_by = "system"
        elif total_cost >= settings.cost_escalation_threshold:
            po.approval_status = ApprovalStatus.PENDING
            _create_approval(
                request_type="purchase_order",
                description=f"PO {po.po_id}: {quantity} units of {product_id} from {supplier_id}",
                estimated_cost=total_cost,
                risk_level=AlertSeverity.HIGH,
                data={"po_id": po.po_id, "po": po.model_dump()},
            )
        else:
            # Middle range: requires review
            po.approval_status = ApprovalStatus.PENDING
            _create_approval(
                request_type="purchase_order",
                description=f"PO {po.po_id}: {quantity} units of {product_id} from {supplier_id}",
                estimated_cost=total_cost,
                risk_level=AlertSeverity.MEDIUM,
                data={"po_id": po.po_id, "po": po.model_dump()},
            )

        _runtime.purchase_orders.append(po)

        return {
            "po_id": po.po_id,
            "total_cost": total_cost,
            "approval_status": po.approval_status.value,
            "expected_delivery": po.expected_delivery.isoformat() if po.expected_delivery else None,
        }


class RequestHumanApproval(BaseTool):
    """Request human approval for a high-cost or high-risk action."""

    name = "request_human_approval"
    description = "Escalate an action to human review."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        request_type: str = kwargs.get("request_type", "general")
        description: str = kwargs.get("description", "")
        estimated_cost: float = kwargs.get("estimated_cost", 0.0)
        risk_level: str = kwargs.get("risk_level", "medium")
        data: dict = kwargs.get("data", {})

        try:
            severity = AlertSeverity(risk_level)
        except ValueError:
            severity = AlertSeverity.MEDIUM

        approval = _create_approval(
            request_type=request_type,
            description=description,
            estimated_cost=estimated_cost,
            risk_level=severity,
            data=data,
        )

        return {
            "request_id": approval.request_id,
            "status": "pending",
            "message": f"Approval request created: {description}",
        }


class AdjustSafetyStock(BaseTool):
    """Adjust safety stock level for a product."""

    name = "adjust_safety_stock"
    description = "Modify the safety stock level for a product."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..orchestrator import _runtime

        product_id: str = kwargs["product_id"]
        new_safety_stock: float = kwargs["new_safety_stock"]

        # Input validation
        if new_safety_stock < 0:
            return {"error": "new_safety_stock must be non-negative"}

        product = next(
            (p for p in (_runtime.products or []) if p.product_id == product_id),
            None,
        )
        if product is None:
            return {"error": f"Product {product_id} not found"}

        old_value = product.safety_stock
        change_pct = (
            abs(new_safety_stock - old_value) / max(old_value, 1) * 100
        )

        # Check if change requires approval
        if change_pct > settings.inventory_change_pct_threshold:
            approval = _create_approval(
                request_type="inventory_adjustment",
                description=(
                    f"Safety stock change for {product_id}: "
                    f"{old_value:.0f} → {new_safety_stock:.0f} ({change_pct:.1f}% change)"
                ),
                estimated_cost=abs(new_safety_stock - old_value) * product.unit_cost,
                risk_level=AlertSeverity.MEDIUM,
                data={
                    "product_id": product_id,
                    "old_value": old_value,
                    "new_value": new_safety_stock,
                },
            )
            return {
                "product_id": product_id,
                "status": "pending_approval",
                "request_id": approval.request_id,
                "change_pct": round(change_pct, 1),
            }

        product.safety_stock = new_safety_stock
        return {
            "product_id": product_id,
            "old_safety_stock": old_value,
            "new_safety_stock": new_safety_stock,
            "status": "applied",
        }


class EmitEvent(BaseTool):
    """Publish an event to the EventBus."""

    name = "emit_event"
    description = "Publish a supply chain event so other agents can react."

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        from ..orchestrator import _runtime

        try:
            severity = AlertSeverity(kwargs.get("severity", "medium"))
        except ValueError:
            severity = AlertSeverity.MEDIUM

        event = SupplyChainEvent(
            event_type=kwargs.get("event_type", "general"),
            severity=severity,
            source_agent=kwargs.get("source_agent", "unknown"),
            description=kwargs.get("description", ""),
            data=kwargs.get("data", {}),
        )

        published = False
        if _runtime.event_bus is not None:
            await _runtime.event_bus.publish(event)
            published = True

        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "published": published,
        }
