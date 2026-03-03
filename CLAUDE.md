# ChainCommand — Claude Code Guide

## Project Overview
ChainCommand is a supply chain AI agent system with 10 autonomous agents, ML forecasting, optimization, and event-driven architecture.

## Quick Start
```bash
pip install -e ".[dev]"           # Install with dev deps
python -m chaincommand --demo     # Run demo cycle
pytest tests/ -v                  # Run tests
ruff check chaincommand/ tests/   # Lint
```

## Architecture (v2.0)
- **10 agents** across 4 layers (strategic/tactical/operational/orchestration)
- **Orchestration**: Classic sequential OR LangGraph state machine (`CC_ORCHESTRATOR_MODE`)
- **Forecasting**: LSTM + XGBoost ensemble + optional Chronos-2
- **Optimization**: GA + DQN hybrid + OR-Tools CP-SAT (supplier allocation)
- **Causal**: DoWhy causal inference for supplier switching decisions
- **Observability**: AgentTracer, TokenBudget, ResponseCache
- **Resilience**: CircuitBreaker, GracefulDegradation (4 levels)

## Key Directories
- `chaincommand/agents/` — 10 agent implementations
- `chaincommand/models/` — ML models (forecaster, anomaly_detector, optimizer, chronos)
- `chaincommand/optimization/` — CP-SAT MILP optimizer + benchmark
- `chaincommand/causal/` — DoWhy causal inference
- `chaincommand/kpi/` — 12-metric KPI engine
- `chaincommand/events/` — Async pub/sub event bus
- `chaincommand/api/` — FastAPI REST + WebSocket
- `chaincommand/aws/` — AWS persistence (S3/Redshift/Athena/QuickSight)
- `tests/` — 120+ tests across 10 test modules

## Configuration
All settings via `CC_` env prefix or `.env` file. Key v2.0 settings:
- `CC_ORCHESTRATOR_MODE`: classic | langgraph
- `CC_ENABLE_CAUSAL_ANALYSIS`: true/false
- `CC_ENABLE_CHRONOS`: true/false
- `CC_TOKEN_BUDGET_PER_CYCLE`: token limit per decision cycle
- `CC_CIRCUIT_BREAKER_FAILURE_THRESHOLD`: failures before circuit opens

## Testing
```bash
pytest tests/ -v --tb=short       # All tests
pytest tests/test_optimization/   # CP-SAT tests only
pytest tests/test_causal/         # Causal inference tests
pytest tests/test_resilience/     # Circuit breaker tests
```

## Dependencies
Core deps always installed. Optional groups:
- `pip install -e ".[langgraph]"` — LangGraph orchestrator
- `pip install -e ".[ortools]"` — OR-Tools CP-SAT
- `pip install -e ".[causal]"` — DoWhy
- `pip install -e ".[chronos]"` — Chronos-2
- `pip install -e ".[all]"` — Everything
