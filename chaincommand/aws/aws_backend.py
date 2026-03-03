"""AWSBackend — assembles S3, Redshift, Athena, and QuickSight clients."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from ..config import settings
from ..data.schemas import KPISnapshot
from ..utils.logging_config import get_logger
from .athena_client import AthenaClient
from .backend import PersistenceBackend
from .quicksight_client import QuickSightClient
from .redshift_client import RedshiftClient
from .s3_client import S3Client

log = get_logger(__name__)


class AWSBackend(PersistenceBackend):
    """Full AWS persistence backend using S3, Redshift, Athena, and QuickSight."""

    def __init__(self) -> None:
        self._s3: S3Client | None = None
        self._redshift: RedshiftClient | None = None
        self._athena: AthenaClient | None = None
        self._quicksight: QuickSightClient | None = None

    async def setup(self) -> None:
        """Initialize all AWS clients and create required tables."""
        log.info("aws_backend_setup_start")

        self._s3 = S3Client()
        self._redshift = RedshiftClient()
        self._athena = AthenaClient()
        self._quicksight = QuickSightClient()

        # Redshift: create tables
        self._redshift.create_tables()

        # Athena: create database + external tables
        self._athena.create_database()
        self._athena.create_external_tables()

        log.info("aws_backend_setup_complete")

    async def teardown(self) -> None:
        """Close Redshift connection."""
        if self._redshift:
            self._redshift.close()
        log.info("aws_backend_teardown")

    async def persist_cycle(
        self,
        cycle: int,
        kpi: KPISnapshot,
        events: list,
        pos: list,
        products: list,
        suppliers: list,
    ) -> None:
        """Persist one cycle's data to S3 and Redshift."""
        now = datetime.now(timezone.utc)
        date_path = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        prefix = settings.aws_s3_prefix.rstrip("/")

        # ── S3: upload JSONL ──
        # KPI snapshot
        kpi_key = f"{prefix}/kpi_snapshots/{date_path}/cycle_{cycle}.json"
        kpi_data = kpi.model_dump()
        kpi_data["cycle"] = cycle
        self._s3.upload_json(kpi_data, kpi_key)

        # Events
        if events:
            events_key = f"{prefix}/events/{date_path}/cycle_{cycle}.jsonl"
            event_records = [
                e.model_dump() if hasattr(e, "model_dump") else e for e in events
            ]
            self._s3.upload_jsonl(event_records, events_key)

        # Purchase orders
        if pos:
            pos_key = f"{prefix}/purchase_orders/{date_path}/cycle_{cycle}.jsonl"
            po_records = [
                p.model_dump() if hasattr(p, "model_dump") else p for p in pos
            ]
            self._s3.upload_jsonl(po_records, pos_key)

        # ── Redshift: direct INSERT for KPI (fast, single row) ──
        self._redshift.insert_kpi_snapshot(cycle, kpi)

        log.info("aws_persist_cycle", cycle=cycle)

    async def persist_demand_history(self, df: pd.DataFrame) -> None:
        """Upload demand history DataFrame to S3 as Parquet."""
        prefix = settings.aws_s3_prefix.rstrip("/")
        key = f"{prefix}/demand_history/full_history.parquet"
        self._s3.upload_dataframe(df, key)
        log.info("aws_persist_demand_history", rows=len(df))

    async def query_kpi_trend(self, metric: str, days: int) -> list:
        """Query KPI trend from Redshift."""
        allowed = {
            "otif", "fill_rate", "mape", "dsi", "stockout_count",
            "total_inventory_value", "carrying_cost", "order_cycle_time",
            "perfect_order_rate", "inventory_turnover", "backorder_rate",
            "supplier_defect_rate",
        }
        if metric not in allowed:
            return []

        safe_days = max(1, min(int(days), 365))
        sql = (
            f"SELECT cycle, timestamp, {metric} "
            f"FROM kpi_snapshots "
            f"WHERE timestamp >= DATEADD(day, -{safe_days}, GETDATE()) "
            f"ORDER BY cycle"
        )
        return self._redshift.query(sql)

    async def query_events(self, event_type: str, limit: int) -> list:
        """Query events from Athena (ad-hoc on S3)."""
        # Strict validation: only alphanumeric, underscores, dots
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', event_type):
            return []
        safe_limit = max(1, min(int(limit), 500))
        sql = (
            f"SELECT event_id, timestamp, event_type, severity, source_agent, description "
            f"FROM events "
            f"WHERE event_type = '{event_type}' "
            f"ORDER BY timestamp DESC "
            f"LIMIT {safe_limit}"
        )
        return self._athena.run_query(sql)
