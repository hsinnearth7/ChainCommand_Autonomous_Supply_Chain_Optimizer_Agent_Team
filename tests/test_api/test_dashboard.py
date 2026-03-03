"""Dashboard endpoint tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def api_key():
    from chaincommand.config import settings
    return settings.api_key


@pytest.fixture
def auth_headers(api_key):
    return {"X-API-Key": api_key}


class TestKPIEndpoints:
    @pytest.mark.asyncio
    async def test_kpi_current_no_data(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/kpi/current", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "error" in data  # No KPI engine set up

    @pytest.mark.asyncio
    async def test_kpi_history_no_data(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/kpi/history", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 0


class TestInventoryEndpoints:
    @pytest.mark.asyncio
    async def test_inventory_status(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/inventory/status", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1
            assert data["products"][0]["product_id"] == "PRD-test01"

    @pytest.mark.asyncio
    async def test_inventory_status_filter(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/inventory/status?product_id=PRD-test01",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_inventory_status_missing_product(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/inventory/status?product_id=NONEXISTENT",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 0


class TestApprovalEndpoints:
    @pytest.mark.asyncio
    async def test_pending_approvals_empty(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/approvals/pending", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_decide_approval_not_found(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.post(
                "/api/approval/FAKE-ID/decide?approved=true",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            assert "error" in resp.json()


class TestAWSEndpoints:
    @pytest.mark.asyncio
    async def test_aws_status(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get("/api/aws/status", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aws_query_disabled(self, client, auth_headers, mock_runtime):
        with patch("chaincommand.orchestrator._runtime", mock_runtime):
            resp = await client.get(
                "/api/aws/query?event_type=test",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "error" in data
