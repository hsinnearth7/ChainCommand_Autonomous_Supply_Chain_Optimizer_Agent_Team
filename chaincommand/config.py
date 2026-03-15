"""Central configuration using Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from env / .env file."""

    model_config = {"env_file": ".env", "env_prefix": "CC_"}

    # ── Simulation ───────────────────────────────────────
    num_products: int = 50
    num_suppliers: int = 20
    history_days: int = 365
    simulation_speed: float = 1.0

    # ── Event engine ─────────────────────────────────────
    event_tick_seconds: float = 5.0
    enable_proactive_monitoring: bool = True

    # ── KPI thresholds ───────────────────────────────────
    otif_target: float = 0.95
    fill_rate_target: float = 0.97
    mape_threshold: float = 15.0
    dsi_max: float = 60.0
    dsi_min: float = 10.0
    stockout_tolerance: int = 3

    # ── Escalation / HITL ────────────────────────────────
    cost_escalation_threshold: float = 50_000.0
    inventory_change_pct_threshold: float = 25.0
    auto_approve_below: float = 10_000.0

    # ── Security ─────────────────────────────────────────
    api_key: str = "dev-key-change-me"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    rate_limit_per_minute: int = 60

    # ── Reproducibility ──────────────────────────────────
    random_seed: int = 42

    # ── Server ───────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ── AWS ──────────────────────────────────────────────
    aws_enabled: bool = False
    aws_region: str = "ap-northeast-1"
    aws_s3_bucket: str = "chaincommand-data"
    aws_s3_prefix: str = "supply-chain/"
    aws_redshift_host: str = ""
    aws_redshift_port: int = 5439
    aws_redshift_db: str = "chaincommand"
    aws_redshift_user: str = ""
    aws_redshift_password: str = ""
    aws_redshift_iam_role: str = ""
    aws_athena_database: str = "chaincommand"
    aws_athena_output: str = "s3://chaincommand-data/athena-results/"
    aws_quicksight_account_id: str = ""

    # ── ML model params ─────────────────────────────────
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_seq_length: int = 30
    lstm_epochs: int = 50
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    isolation_contamination: float = 0.05
    ga_population_size: int = 50
    ga_generations: int = 100
    dqn_hidden_size: int = 128
    dqn_episodes: int = 200
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: float = 0.995

    # ── CP-SAT Optimization ─────────────────────────────
    ortools_time_limit_ms: int = 10_000
    ortools_risk_lambda: float = 0.3
    ortools_max_suppliers: int = 5

    # ── RL Inventory Policy ──────────────────────────────
    rl_total_timesteps: int = 50_000
    rl_episode_length: int = 90
    rl_holding_cost: float = 0.5
    rl_stockout_cost: float = 10.0
    rl_ordering_cost_fixed: float = 50.0

    # ── BOM Management ───────────────────────────────────
    bom_default_assemblies: int = 5
    bom_long_lead_threshold_days: int = 14

    # ── CTB (Clear-to-Build) ─────────────────────────────
    ctb_default_build_qty: float = 100.0


settings = Settings()
