"""Athena operations for ChainCommand ad-hoc analytics."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)

# ── External table DDL templates ────────────────────────

CREATE_DATABASE_DDL = "CREATE DATABASE IF NOT EXISTS {database}"

EXTERNAL_TABLE_DEMAND_HISTORY = """
CREATE EXTERNAL TABLE IF NOT EXISTS {database}.demand_history (
    date            STRING,
    product_id      STRING,
    quantity        DOUBLE,
    is_promotion    BOOLEAN,
    is_holiday      BOOLEAN,
    temperature     DOUBLE,
    day_of_week     INT,
    month           INT
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/demand_history/'
TBLPROPERTIES ('parquet.compression'='SNAPPY');
"""

EXTERNAL_TABLE_KPI_SNAPSHOTS = """
CREATE EXTERNAL TABLE IF NOT EXISTS {database}.kpi_snapshots (
    cycle           INT,
    timestamp       STRING,
    otif            DOUBLE,
    fill_rate       DOUBLE,
    mape            DOUBLE,
    dsi             DOUBLE,
    stockout_count  INT,
    total_inventory_value DOUBLE,
    carrying_cost   DOUBLE,
    order_cycle_time DOUBLE,
    perfect_order_rate DOUBLE,
    inventory_turnover DOUBLE,
    backorder_rate  DOUBLE,
    supplier_defect_rate DOUBLE
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/kpi_snapshots/'
TBLPROPERTIES ('parquet.compression'='SNAPPY');
"""

EXTERNAL_TABLE_EVENTS = """
CREATE EXTERNAL TABLE IF NOT EXISTS {database}.events (
    event_id        STRING,
    timestamp       STRING,
    event_type      STRING,
    severity        STRING,
    source_agent    STRING,
    description     STRING,
    resolved        BOOLEAN,
    resolution      STRING
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/events/'
TBLPROPERTIES ('parquet.compression'='SNAPPY');
"""

ALL_EXTERNAL_TABLES = [
    EXTERNAL_TABLE_DEMAND_HISTORY,
    EXTERNAL_TABLE_KPI_SNAPSHOTS,
    EXTERNAL_TABLE_EVENTS,
]


class AthenaClient:
    """Encapsulates AWS Athena query operations."""

    def __init__(
        self,
        database: Optional[str] = None,
        output_location: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        import boto3

        self._database = self._validate_identifier(
            database or settings.aws_athena_database, "database"
        )
        self._output = output_location or settings.aws_athena_output
        self._client = boto3.client("athena", region_name=region or settings.aws_region)
        self._bucket = self._validate_identifier(
            settings.aws_s3_bucket, "bucket"
        )
        self._prefix = settings.aws_s3_prefix.rstrip("/")

    @staticmethod
    def _validate_identifier(value: str, name: str) -> str:
        """Validate that a value is a safe SQL/AWS identifier."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-]*$', value):
            raise ValueError(f"Invalid {name}: {value}")
        return value

    def create_database(self) -> str:
        """Create the Athena database if it doesn't exist."""
        sql = CREATE_DATABASE_DDL.format(database=self._database)
        execution_id = self._start_query(sql)
        self._wait_for_query(execution_id)
        log.info("athena_database_created", database=self._database)
        return execution_id

    def create_external_tables(self) -> List[str]:
        """Create all external tables pointing at S3 data."""
        execution_ids = []
        for template in ALL_EXTERNAL_TABLES:
            sql = template.format(
                database=self._database,
                bucket=self._bucket,
                prefix=self._prefix,
            )
            eid = self._start_query(sql)
            self._wait_for_query(eid)
            execution_ids.append(eid)
        log.info("athena_external_tables_created", count=len(execution_ids))
        return execution_ids

    def run_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a query, wait for completion, and return results."""
        execution_id = self._start_query(sql)
        self._wait_for_query(execution_id)
        return self.get_query_results(execution_id)

    def get_query_results(self, execution_id: str) -> List[Dict[str, Any]]:
        """Retrieve results from a completed query execution."""
        resp = self._client.get_query_results(QueryExecutionId=execution_id)
        rows = resp["ResultSet"]["Rows"]
        if not rows:
            return []

        # First row is the header
        headers = [col["VarCharValue"] for col in rows[0]["Data"]]
        results = []
        for row in rows[1:]:
            values = [col.get("VarCharValue", "") for col in row["Data"]]
            results.append(dict(zip(headers, values, strict=False)))
        return results

    def _start_query(self, sql: str) -> str:
        """Start a query execution and return the execution ID."""
        resp = self._client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self._database},
            ResultConfiguration={"OutputLocation": self._output},
        )
        return resp["QueryExecutionId"]

    def _wait_for_query(
        self, execution_id: str, max_wait: int = 300, interval: float = 1.0
    ) -> str:
        """Poll until a query completes. Returns final state."""
        elapsed = 0.0
        while elapsed < max_wait:
            resp = self._client.get_query_execution(QueryExecutionId=execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                if state != "SUCCEEDED":
                    reason = resp["QueryExecution"]["Status"].get(
                        "StateChangeReason", "unknown"
                    )
                    log.error("athena_query_failed", state=state, reason=reason)
                    raise RuntimeError(
                        f"Athena query {execution_id} {state}: {reason}"
                    )
                return state
            time.sleep(interval)
            elapsed += interval
        raise TimeoutError(
            f"Athena query {execution_id} did not complete within {max_wait}s"
        )
