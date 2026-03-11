<div align="center">

# ChainCommand — Autonomous Supply Chain Optimizer Agent Team

**10 AI Agents | 4 Collaborative Layers | Event-Driven Architecture | From Data to Autonomous Decisions**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-E92063.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LLM Ready](https://img.shields.io/badge/LLM-Mock%20%7C%20OpenAI%20%7C%20Ollama-blueviolet.svg)](#llm-backends)

<br>

<img src="https://img.shields.io/badge/Agents-10%20Autonomous-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Architecture-Event%20Driven-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-LSTM%20%2B%20XGBoost%20%2B%20Chronos--2%20%2B%20GA%20%2B%20DQN-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Optimization-CP--SAT%20MILP%20%2B%20DoWhy-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/AWS-S3%20%7C%20Redshift%20%7C%20Athena%20%7C%20QuickSight-FF9900?style=for-the-badge" />
<img src="https://img.shields.io/badge/Tests-120%2B%20Passed-brightgreen?style=for-the-badge" />

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Agent Team](#agent-team)
- [Getting Started](#getting-started)
- [Pipeline Details](#pipeline-details)
- [API Reference](#api-reference)
- [Decision Cycle Walkthrough](#decision-cycle-walkthrough)
- [Research Foundations](#research-foundations)
- [AWS Integration (Optional)](#aws-integration-optional)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**ChainCommand** is a multi-agent AI system that autonomously optimizes end-to-end supply chain operations. Ten specialized agents — organized across four collaborative layers (Strategic, Tactical, Operational, and Orchestration) — communicate through an async event-driven pub/sub architecture to forecast demand, optimize inventory, assess supply risk, coordinate logistics, and generate executive reports.

The system runs from a single command (`python -m chaincommand --demo`) with zero API keys: it generates a realistic supply chain scenario (50 products, 20 suppliers, 365-day demand history), trains ML models, wires 10 agents with 16+ tools across an EventBus, and executes a complete decision cycle. Orchestration supports both a classic sequential 8-step pipeline and a LangGraph state machine with conditional routing and HITL gates.

### Why This Project?

| Challenge | Approach |
|-----------|----------|
| Supply chain decisions are siloed | 10 specialized agents with cross-layer event communication |
| Forecasting relies on single models | LSTM + XGBoost ensemble with dynamic MAPE-based weighting + Chronos-2 zero-shot |
| Inventory optimization is static | GA global search + DQN reinforcement learning + CP-SAT MILP for supplier allocation |
| Supplier switching lacks causal rigor | DoWhy 4-step causal inference (Model → Identify → Estimate → Refute) |
| High-cost decisions lack oversight | HITL gates with configurable thresholds |
| Bullwhip effect amplifies volatility | Beer game consensus mechanism across agent layers |
| Systems fail silently | CircuitBreaker + 4-level GracefulDegradation + AgentTracer observability |
| Production data lacks persistence | AWS Strategy Pattern backend (S3 / Redshift / Athena / QuickSight) |

---

## Key Features

- **10-Agent Autonomous Team** — Demand Forecaster, Strategic Planner, Inventory Optimizer, Supplier Manager, Logistics Coordinator, Anomaly Detector, Risk Assessor, Market Intelligence, Coordinator (CSCO), and Reporter
- **4-Layer Architecture** — Strategic (weekly/monthly), Tactical (daily), Operational (real-time), and Orchestration (cross-layer coordination) — inspired by JD.com's two-tier architecture
- **Dual Orchestration** — Classic sequential 8-step pipeline OR LangGraph state machine with conditional routing (`CC_ORCHESTRATOR_MODE`)
- **Ensemble Forecasting** — LSTM + XGBoost with dynamic inverse-MAPE weighting + optional Chronos-2 zero-shot for cold-start products
- **CP-SAT MILP Optimization** — OR-Tools constraint programming for optimal supplier allocation with demand satisfaction, capacity, MOQ, and lead-time constraints
- **Causal Inference** — DoWhy causal analysis for supplier switching decisions with IPW estimation and 3 refutation tests
- **Hybrid Inventory Optimization** — Genetic Algorithm for global parameter search + DQN reinforcement learning for dynamic inventory decisions
- **Anomaly Detection** — Isolation Forest + Z-score detection for demand spikes, cost anomalies, and lead-time deviations
- **Resilience** — CircuitBreaker (closed/open/half-open) + 4-level GracefulDegradation (full → llm_partial → rule_based → human)
- **Observability** — AgentTracer (span-based per-agent tracing), TokenBudget (per-cycle token limits), ResponseCache (LRU caching)
- **Pydantic State Management** — `SupplyChainState` with per-agent isolation and immutable snapshots
- **Event-Driven Communication** — Async pub/sub EventBus decouples agent interactions
- **HITL Approval Gates** — Orders ≥$50K require human approval; $10K–$50K pending review; <$10K auto-approved
- **12 KPI Metrics** — OTIF, fill rate, MAPE, DSI, stockout count, inventory turnover, carrying cost, and more
- **REST API + WebSocket** — FastAPI dashboard with live event streaming, agent triggers, and simulation control
- **Rich Terminal UI** — Demo mode with animated progress bars, color-coded KPIs, and agent-layer results
- **Mock-First Design** — Complete system runs without any API keys using rule-based mock LLM
- **AWS Persistence (Optional)** — Strategy Pattern backend with S3, Redshift, Athena, and QuickSight integration
- **120+ Tests** — Unit, integration, property-based (Hypothesis), and chaos testing across 12 test modules
- **Docker Deployment** — Dockerfile + docker-compose.yml for containerized deployment

---

## System Architecture

### Overall Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     LangGraph StateGraph                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  Market Intel ──→ Anomaly ──→ Forecast ──→ Inventory        │  │
│  │  ──→ Risk ──→ Supplier ──┬──→ Logistics ──→ Planner         │  │
│  │                    [HITL?]│                                  │  │
│  │                    ├──────┘                                  │  │
│  │  ──→ Coordinator ──→ Reporter ──→ KPI Update                │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────┬────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  CP-SAT  │ │  DoWhy   │ │ Chronos  │
    │   MILP   │ │  Causal  │ │    -2    │
    │Optimizer │ │Inference │ │Forecaster│
    └──────────┘ └──────────┘ └──────────┘
```

### MILP Formulation (CP-SAT)

```
min  Σ(cᵢ · xᵢ) + λ · Σ(rᵢ · xᵢ)

s.t. Σ xᵢ ≥ D                    (demand satisfaction)
     xᵢ ≥ MOQᵢ · yᵢ              (minimum order qty)
     xᵢ ≤ Capᵢ · yᵢ              (capacity limit)
     Σ yᵢ ≤ K                     (max suppliers)
     LTᵢ · yᵢ ≤ LT_max           (lead-time)
```

### DoWhy Causal DAG

```
initial_quality_score ──┐
disruption_severity ────┤
alternative_count ──────┼──→ switched_supplier ──→ total_cost_delta
product_criticality ────┘
```

### Resilience Levels

| Level | Behavior |
|-------|----------|
| `full` | All systems operational |
| `llm_partial` | LLM calls limited, shorter prompts |
| `rule_based` | No LLM, heuristic rules only |
| `human` | Escalate to human operator |

---

## Project Structure

```
chaincommand/
│
├── __init__.py                          # Package metadata (v2.0.0)
├── __main__.py                          # CLI entry point (--demo / server)
├── config.py                            # Pydantic Settings (CC_ prefix, env-driven)
├── state.py                             # Pydantic SupplyChainState (per-agent isolation)
├── orchestrator.py                      # Classic 8-step sequential orchestrator
├── langgraph_orchestrator.py            # LangGraph StateGraph orchestrator
├── resilience.py                        # CircuitBreaker + 4-level GracefulDegradation
├── observability.py                     # AgentTracer, TokenBudget, ResponseCache
├── auth.py                              # API key authentication
│
├── llm/                                 # LLM Abstraction Layer
│   ├── base.py                          # Abstract base class (generate / generate_json)
│   ├── mock_llm.py                      # Rule-based mock (regex intent matching)
│   ├── openai_llm.py                    # OpenAI async client (JSON mode)
│   ├── ollama_llm.py                    # Ollama local model (httpx)
│   └── factory.py                       # Factory: create_llm() based on CC_LLM_MODE
│
├── data/                                # Domain Data
│   ├── schemas.py                       # 13+ Pydantic models + enums
│   └── generator.py                     # Synthetic data: 50 products, 20 suppliers
│
├── tools/                               # Agent Tools (16+ tools)
│   ├── base_tool.py                     # Abstract BaseTool
│   ├── data_tools.py                    # QueryDemandHistory, QueryInventoryStatus, etc.
│   ├── forecast_tools.py               # RunDemandForecast, GetForecastAccuracy
│   ├── optimization_tools.py           # CalculateReorderPoint, OptimizeInventory
│   ├── risk_tools.py                    # DetectAnomalies, AssessSupplyRisk
│   ├── action_tools.py                 # CreatePurchaseOrder, RequestHumanApproval
│   └── allocation_tools.py             # CP-SAT supplier allocation tools
│
├── models/                              # ML Models
│   ├── forecaster.py                    # LSTM + XGBoost ensemble forecaster
│   ├── chronos_forecaster.py            # Chronos-2 zero-shot (optional)
│   ├── anomaly_detector.py             # Isolation Forest + Z-score
│   └── optimizer.py                     # GA + DQN hybrid optimizer
│
├── optimization/                        # Mathematical Optimization
│   ├── cpsat_optimizer.py               # OR-Tools CP-SAT MILP solver
│   └── benchmark.py                     # GA vs CP-SAT benchmark comparison
│
├── causal/                              # Causal Inference
│   ├── supplier_switch.py               # DoWhy causal analysis (IPW + refutation)
│   └── data_generator.py               # Synthetic observational data
│
├── kpi/                                 # KPI Engine
│   └── engine.py                        # 12 metrics, threshold checks, trends
│
├── events/                              # Event Engine
│   ├── bus.py                           # EventBus (async pub/sub)
│   └── monitor.py                       # ProactiveMonitor (tick-based health checks)
│
├── agents/                              # Agent Team (10 agents)
│   ├── base_agent.py                    # BaseAgent (think / act / handle_event)
│   ├── demand_forecaster.py             # Strategic: demand analysis & forecasting
│   ├── strategic_planner.py             # Strategic: inventory policy & consensus
│   ├── inventory_optimizer.py           # Tactical: reorder points & safety stock
│   ├── supplier_manager.py              # Tactical: supplier evaluation & procurement
│   ├── logistics_coordinator.py         # Tactical: order tracking & delivery
│   ├── anomaly_detector_agent.py        # Operational: real-time anomaly detection
│   ├── risk_assessor.py                 # Operational: supply risk quantification
│   ├── market_intelligence.py           # Operational: market signal scanning
│   ├── coordinator.py                   # Orchestration: CSCO conflict resolution
│   └── reporter.py                      # Orchestration: structured report generation
│
├── api/                                 # FastAPI Application
│   ├── app.py                           # FastAPI app with CORS & lifespan
│   └── routes/
│       ├── dashboard.py                 # KPI, inventory, agents, events, AWS, WebSocket
│       └── control.py                   # Simulation start/stop/speed, agent triggers
│
├── aws/                                 # AWS Persistence Backend (optional)
│   ├── backend.py                       # PersistenceBackend ABC, NullBackend, factory
│   ├── aws_backend.py                   # AWSBackend — assembles all clients
│   ├── s3_client.py                     # S3 upload/download (Parquet, JSONL, JSON)
│   ├── redshift_client.py              # Redshift DDL, COPY, queries
│   ├── athena_client.py                # Athena external tables, ad-hoc queries
│   └── quicksight_client.py            # QuickSight datasets + dashboards
│
├── ui/                                  # Rich Terminal UI (demo mode)
│   ├── theme.py                        # Visual constants, colors, layer badges
│   └── console.py                      # Progress bars, KPI dashboard, trees
│
└── utils/
    └── logging_config.py               # structlog configuration

tests/                                   # 120+ tests across 12 modules
├── conftest.py                          # Shared fixtures
├── test_agents/                         # Agent lifecycle tests (21 tests)
├── test_models/                         # ML model tests (18 tests)
├── test_kpi/                            # KPI engine tests (8 tests)
├── test_optimization/                   # CP-SAT optimizer tests (10 tests)
├── test_causal/                         # DoWhy causal inference tests (10 tests)
├── test_resilience/                     # Circuit breaker tests (12 tests)
├── test_observability/                  # AgentTracer, TokenBudget tests (14 tests)
├── test_integration/                    # LangGraph integration tests (8 tests)
├── test_chaos/                          # Fault injection tests (5 tests)
├── test_property_based.py               # Hypothesis property tests (5 tests)
├── test_api/                            # API endpoint tests
├── test_aws/                            # AWS backend tests (47 tests, all mocked)
└── test_tools/                          # Tool execution tests
```

---

## Agent Team

### 10 Agents Across 4 Layers

| Layer | Agent | Role | Tools | Event Subscriptions |
|-------|-------|------|-------|-------------------|
| **Strategic** | Demand Forecaster | Analyze sales patterns, produce demand forecasts | QueryDemandHistory, RunDemandForecast, GetForecastAccuracy, ScanMarketIntelligence | `kpi_threshold_violated`, `new_market_intel` |
| **Strategic** | Strategic Planner | Develop inventory policies, reduce bullwhip effect | QueryKPIHistory, OptimizeInventory, QueryInventoryStatus | `forecast_updated`, `kpi_trend_alert` |
| **Tactical** | Inventory Optimizer | Monitor stock levels, manage reorder points | QueryInventoryStatus, CalculateReorderPoint, AdjustSafetyStock, OptimizeInventory | `low_stock_alert`, `overstock_alert`, `stockout_alert`, `forecast_updated` |
| **Tactical** | Supplier Manager | Evaluate & select suppliers, manage procurement | QuerySupplierInfo, EvaluateSupplier, CreatePurchaseOrder, RequestHumanApproval | `reorder_triggered`, `supplier_issue`, `quality_alert` |
| **Tactical** | Logistics Coordinator | Track shipments, manage delivery timelines | QueryInventoryStatus, EmitEvent | `po_created`, `delivery_delayed` |
| **Operational** | Anomaly Detector | Real-time anomaly detection (demand/cost/quality) | DetectAnomalies, QueryDemandHistory, QueryInventoryStatus | `new_data_point`, `tick` |
| **Operational** | Risk Assessor | Quantify supply risk (depth/breadth/criticality) | AssessSupplyRisk, ScanMarketIntelligence, QuerySupplierInfo | `anomaly_detected`, `supply_risk_alert` |
| **Operational** | Market Intelligence | Monitor market dynamics, scan for trends | ScanMarketIntelligence, EmitEvent | `tick` |
| **Orchestration** | Coordinator (CSCO) | Resolve conflicts, enforce constraints, executive summary | All query tools, RequestHumanApproval, EmitEvent | **All events** |
| **Orchestration** | Reporter | Aggregate outputs into structured reports | QueryKPIHistory, QueryInventoryStatus | `cycle_complete`, `kpi_snapshot_created` |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/hsinnearth7/ChainCommand.git
cd ChainCommand

# Install core + dev dependencies (PEP 621)
pip install -e ".[dev]"

# Optional: install specific feature groups
pip install -e ".[langgraph]"     # LangGraph orchestrator
pip install -e ".[ortools]"      # OR-Tools CP-SAT MILP
pip install -e ".[causal]"       # DoWhy causal inference
pip install -e ".[chronos]"      # Chronos-2 zero-shot forecasting
pip install -e ".[all]"          # Everything
```

### Quick Start

```bash
# Run a single decision cycle (no API keys needed)
python -m chaincommand --demo

# Start the FastAPI server
python -m chaincommand --host 0.0.0.0 --port 8000

# Docker
docker compose up --build
```

### Environment Variables

All settings are configurable via `CC_` prefixed environment variables or a `.env` file:

```bash
# Orchestrator mode
CC_ORCHESTRATOR_MODE=classic           # classic | langgraph

# LLM Backend
CC_LLM_MODE=mock                       # mock | openai | ollama
CC_OPENAI_API_KEY=sk-...               # when CC_LLM_MODE=openai
CC_OLLAMA_BASE_URL=http://localhost:11434  # when CC_LLM_MODE=ollama

# Optional modules
CC_ENABLE_CAUSAL_ANALYSIS=false        # DoWhy causal inference
CC_ENABLE_CHRONOS=false                # Chronos-2 zero-shot forecaster

# Resilience
CC_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5 # failures before circuit opens
CC_TOKEN_BUDGET_PER_CYCLE=100000       # token limit per decision cycle

# Simulation
CC_NUM_PRODUCTS=50
CC_NUM_SUPPLIERS=20
CC_SIMULATION_SPEED=1.0

# KPI Thresholds
CC_OTIF_TARGET=0.95
CC_FILL_RATE_TARGET=0.97
CC_MAPE_THRESHOLD=15.0

# HITL Escalation
CC_COST_ESCALATION_THRESHOLD=50000
CC_AUTO_APPROVE_BELOW=10000
```

---

## Pipeline Details

### LLM Abstraction Layer

> `chaincommand/llm/` — Unified LLM interface for all agents

```python
async def generate(prompt, system, temperature) -> str          # Free-form text
async def generate_json(prompt, schema, system, temperature) -> BaseModel  # Structured output
```

| Backend | Class | Description |
|---------|-------|-------------|
| **Mock** | `MockLLM` | Regex-based intent matching with pre-defined responses (no API key) |
| **OpenAI** | `OpenAILLM` | Async OpenAI client with JSON mode support |
| **Ollama** | `OllamaLLM` | httpx async client connecting to local Ollama instance |

### ML Models

> `chaincommand/models/` — Forecasting, anomaly detection, and optimization

**Ensemble Forecaster** — Dynamic-weighted LSTM + XGBoost:
```
Weight_LSTM = (1/MAPE_LSTM) / ((1/MAPE_LSTM) + (1/MAPE_XGB))
Weight_XGB  = (1/MAPE_XGB)  / ((1/MAPE_LSTM) + (1/MAPE_XGB))
```

**Chronos-2 Zero-Shot** — Foundation model for cold-start products with no training history (optional, requires `pip install -e ".[chronos]"`).

**Anomaly Detector** — Isolation Forest with Z-score fallback:
- Demand spike detection (|z| > 2.5)
- Overstock detection (DSI > 60 days)
- Understock detection (DSI < 10 days)

**Hybrid Optimizer** — GA + DQN:
```
GA  → reorder_point, safety_stock    (global search, 50 pop × 100 gen)
DQN → order_quantity                  (dynamic policy, 200 episodes, ε-greedy)
Blend: 60% GA + 40% DQN for order quantity
```

### CP-SAT MILP Optimization

> `chaincommand/optimization/` — OR-Tools constraint programming for supplier allocation

Solves the multi-supplier allocation problem with demand, capacity, MOQ, lead-time, and max-supplier constraints. Includes a benchmark module comparing CP-SAT vs GA solutions on cost and optimality gap.

### Causal Inference

> `chaincommand/causal/` — DoWhy causal analysis for supplier switching

4-step pipeline: (1) Build causal DAG, (2) Identify causal effect via backdoor criterion, (3) Estimate ATE with IPW, (4) Validate with 3 refutation tests (placebo treatment, random common cause, data subset).

### Observability & Resilience

> `chaincommand/observability.py` + `chaincommand/resilience.py`

| Component | Description |
|-----------|-------------|
| **AgentTracer** | Span-based per-agent tracing with timing and metadata |
| **TokenBudget** | Per-cycle token consumption tracking and enforcement |
| **ResponseCache** | LRU cache for repeated LLM queries |
| **CircuitBreaker** | Closed → Open → Half-Open state machine with failure threshold |
| **GracefulDegradation** | 4 levels: full → llm_partial → rule_based → human escalation |

### KPI Engine

> `chaincommand/kpi/engine.py` — 12 real-time supply chain metrics

| KPI | Formula | Threshold |
|-----|---------|-----------|
| **OTIF** | On-Time In-Full deliveries / Total deliveries | ≥ 95% |
| **Fill Rate** | Fulfilled demand / Total demand | ≥ 97% |
| **MAPE** | Mean Absolute Percentage Error of forecasts | ≤ 15% |
| **DSI** | Total Stock / Average Daily Demand | 10–60 days |
| **Stockout Count** | Products with zero stock | ≤ 3 |
| **Inventory Value** | Σ (stock × unit cost) | — |
| **Carrying Cost** | 25% of inventory value / 365 (daily) | — |
| **Order Cycle Time** | Average days from PO creation to delivery | — |
| **Perfect Order Rate** | Perfect deliveries / Total orders | — |
| **Inventory Turnover** | Annual COGS / Average inventory value | — |
| **Backorder Rate** | Backordered products / Total products | — |
| **Supplier Defect Rate** | Average defect rate across active suppliers | — |

### Event Engine

> `chaincommand/events/` — Async pub/sub and proactive monitoring

**EventBus** — Asynchronous publish/subscribe with error isolation:
```
publish(event) → dispatches to type-specific + wildcard subscribers
subscribe(event_type, handler) → register for specific events
subscribe_all(handler) → register for ALL events (Coordinator)
```

**ProactiveMonitor** — Tick-based health scanning:
1. Inventory water level checks → `stockout_alert`, `low_stock_alert`, `overstock_alert`
2. KPI threshold violations → `kpi_threshold_violated`
3. Delivery delay detection → `delivery_delayed`
4. Anomaly detection batch → `anomaly_detected`
5. Tick heartbeat → `tick` (for agents that act every cycle)

---

## API Reference

### Dashboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/kpi/current` | Latest KPI snapshot (12 metrics) |
| `GET` | `/api/kpi/history?periods=30` | KPI trend data |
| `GET` | `/api/inventory/status` | All products with stock status |
| `GET` | `/api/inventory/status?product_id=PRD-0001` | Single product detail |
| `GET` | `/api/agents/status` | All 10 agent statuses |
| `GET` | `/api/events/recent?limit=50` | Recent supply chain events |
| `GET` | `/api/forecast/{product_id}` | 30-day demand forecast |
| `GET` | `/api/approvals/pending` | Pending HITL approval requests |
| `POST` | `/api/approval/{id}/decide` | Approve or reject a request |
| `GET` | `/api/aws/status` | AWS backend status and configuration |
| `GET` | `/api/aws/kpi-trend/{metric}` | KPI trend from Redshift |
| `GET` | `/api/aws/query` | Ad-hoc event query via Athena |
| `GET` | `/api/aws/dashboards` | List QuickSight dashboards |
| `WS` | `/ws/live` | Real-time event stream |

### Control Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/simulation/start` | Start continuous simulation loop |
| `POST` | `/api/simulation/stop` | Stop simulation |
| `POST` | `/api/simulation/speed?speed=5.0` | Adjust simulation speed (0.1–100x) |
| `POST` | `/api/agents/{name}/trigger` | Manually trigger one agent's cycle |
| `GET` | `/api/simulation/status` | Running state, cycle count, stats |

---

## Decision Cycle Walkthrough

Each cycle follows an 8-step sequence (classic mode) or a conditional state graph (LangGraph mode):

```
Step 1: OPERATIONAL SCAN
  ├── Market Intelligence → scans market signals (price, regulatory, competitor)
  └── Anomaly Detector   → scans all products for demand/cost/stock anomalies

Step 2: STRATEGIC FORECASTING
  └── Demand Forecaster  → LSTM+XGB ensemble (+ Chronos-2 if enabled)
                            → publishes forecast_updated events

Step 3: INVENTORY + RISK
  ├── Inventory Optimizer → identifies products below reorder point
  │                         → triggers reorder_triggered events
  └── Risk Assessor       → evaluates supply risk (depth/breadth/criticality)

Step 4: SUPPLIER MANAGEMENT
  └── Supplier Manager    → evaluates suppliers, creates purchase orders
                            → CP-SAT optimal allocation (if ortools enabled)
                            → HITL gate: ≥$50K requires approval

Step 5: LOGISTICS
  └── Logistics Coordinator → tracks active shipments, simulates progression

Step 6: STRATEGIC PLANNING
  └── Strategic Planner   → reviews KPIs, runs optimization, applies consensus

Step 7: COORDINATOR ARBITRATION
  └── Coordinator (CSCO)  → collects all actions, resolves conflicts, prioritizes

Step 8: REPORT GENERATION
  └── Reporter            → produces structured report with KPI snapshot
```

---

## Research Foundations

| Research | Source | Applied Concept |
|----------|--------|----------------|
| JD.com Two-Layer Architecture | ArXiv 2509.03811 | Strategic + Tactical agent separation |
| Seven-Agent Disruption Monitoring | ArXiv 2601.09680 | Proactive monitoring with specialized detection agents |
| MARL Inventory Replenishment | ArXiv 2511.23366 | DQN-based reinforcement learning for inventory decisions |
| Temporal Hierarchical MAS | ArXiv 2508.12683 | Three temporal layers (strategic/tactical/operational) |
| Beer Game Consensus Mechanism | ArXiv 2411.10184 | Cross-agent consensus to reduce bullwhip effect |

---

## AWS Integration (Optional)

ChainCommand supports an optional AWS persistence backend using the **Strategy Pattern**. When enabled, cycle data is persisted to S3/Redshift for durable storage and analytics via Athena and QuickSight — defaulting to a zero-overhead `NullBackend` when disabled.

### Architecture

```
PersistenceBackend (ABC)
  ├── NullBackend        # Default — no-op, zero overhead
  └── AWSBackend         # S3 + Redshift + Athena + QuickSight
        ├── S3Client         # Upload Parquet/JSONL/JSON
        ├── RedshiftClient   # COPY from S3, SQL queries
        ├── AthenaClient     # External tables on S3, ad-hoc queries
        └── QuickSightClient # Datasets + dashboards
```

### Environment Variables

```bash
CC_AWS_ENABLED=true
CC_AWS_REGION=ap-northeast-1
CC_AWS_S3_BUCKET=chaincommand-data
CC_AWS_S3_PREFIX=supply-chain/
CC_AWS_REDSHIFT_HOST=my-cluster.abc123.redshift.amazonaws.com
CC_AWS_REDSHIFT_PORT=5439
CC_AWS_REDSHIFT_DB=chaincommand
CC_AWS_REDSHIFT_USER=admin
CC_AWS_REDSHIFT_PASSWORD=secret
CC_AWS_REDSHIFT_IAM_ROLE=arn:aws:iam::123456789012:role/RedshiftS3Access
CC_AWS_ATHENA_DATABASE=chaincommand
CC_AWS_ATHENA_OUTPUT=s3://chaincommand-data/athena-results/
CC_AWS_QUICKSIGHT_ACCOUNT_ID=123456789012
```

---

## Testing

```bash
# Run all tests (120+)
pytest tests/ -v

# Run by module
pytest tests/test_agents/         # Agent tests (21 tests)
pytest tests/test_models/         # ML model tests (18 tests)
pytest tests/test_kpi/            # KPI engine tests (8 tests)
pytest tests/test_optimization/   # CP-SAT optimizer tests (10 tests)
pytest tests/test_causal/         # Causal inference tests (10 tests)
pytest tests/test_resilience/     # Circuit breaker tests (12 tests)
pytest tests/test_observability/  # Observability tests (14 tests)
pytest tests/test_integration/    # LangGraph integration tests (8 tests)
pytest tests/test_chaos/          # Fault injection tests (5 tests)
pytest tests/test_property_based.py  # Hypothesis property tests (5 tests)
pytest tests/test_aws/ -v         # AWS backend tests (47 tests)
```

| Test Module | Tests | Coverage |
|-------------|-------|----------|
| `test_agents/` | 21 | BaseAgent, DemandForecaster, SupplierManager, Coordinator |
| `test_models/` | 18 | LSTM, XGBoost, Ensemble, AnomalyDetector, GA, DQN, Hybrid |
| `test_kpi/` | 8 | KPI calculations, thresholds, violations |
| `test_optimization/` | 10 | CP-SAT allocation, constraints, sensitivity analysis |
| `test_causal/` | 10 | Synthetic data, ATE, refutation, IPW fallback |
| `test_resilience/` | 12 | Circuit breaker states, transitions, degradation |
| `test_observability/` | 14 | AgentTracer, TokenBudget, ResponseCache |
| `test_integration/` | 8 | LangGraph graph build, cycle, HITL, factory |
| `test_chaos/` | 5 | Exception survival, fault injection |
| `test_property_based.py` | 5 | Hypothesis: KPI ranges, optimizer constraints |
| `test_api/` | 3 | API endpoints, security |
| `test_aws/` | 47 | All AWS clients fully mocked |
| **Total** | **120+** | Full coverage across all modules |

All tests use mocked dependencies — no real AWS, LLM API keys, or GPU required.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.11+ |
| **Data Models** | Pydantic 2.0+, Pydantic Settings |
| **Data Processing** | pandas, numpy |
| **ML/Statistics** | scikit-learn (Isolation Forest), custom LSTM/XGB/GA/DQN |
| **Optimization** | OR-Tools CP-SAT (MILP), Genetic Algorithm, DQN |
| **Causal Inference** | DoWhy (IPW, refutation), networkx (DAG) |
| **Forecasting** | Custom LSTM/XGB + Chronos-2 (optional zero-shot) |
| **Orchestration** | LangGraph StateGraph (optional) + classic sequential |
| **Observability** | AgentTracer, TokenBudget, ResponseCache |
| **Resilience** | CircuitBreaker, 4-level GracefulDegradation |
| **API Server** | FastAPI, uvicorn |
| **Async Runtime** | asyncio (native Python) |
| **Terminal UI** | rich (progress bars, tables, trees) |
| **Logging** | structlog (structured, ISO 8601) |
| **LLM Clients** | openai (optional), httpx (Ollama, optional) |
| **AWS (optional)** | boto3, redshift-connector, pyarrow (S3, Redshift, Athena, QuickSight) |
| **Testing** | pytest, hypothesis, pytest-asyncio, pytest-cov |
| **Deployment** | Docker, docker-compose |
| **Configuration** | Environment variables (CC_ prefix), .env file |

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add new agent or tool'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Architecture inspired by JD.com's autonomous supply chain research and multi-agent systems literature
- Beer game consensus mechanism adapted from ArXiv 2411.10184
- CP-SAT formulation based on OR-Tools constraint programming best practices
- DoWhy causal inference framework by Microsoft Research
- Built upon [ChainInsight](https://github.com/hsinnearth7/ChainInsight) — our supply chain analytics predecessor project

---

## Enterprise Deployment Infrastructure

ChainCommand includes production-grade deployment infrastructure spanning three maturity phases.

### Project Structure (Infrastructure)

```
├── k8s/                        # Kubernetes manifests
│   ├── namespace.yaml          # Namespace isolation
│   ├── configmap.yaml          # Application configuration
│   ├── secret.yaml             # Sensitive credentials
│   ├── deployment.yaml         # 2-replica deployment with health probes
│   ├── service.yaml            # ClusterIP service
│   ├── hpa.yaml                # Horizontal Pod Autoscaler (2-10 pods)
│   ├── ingress.yaml            # Nginx ingress controller
│   ├── postgres.yaml           # PostgreSQL 16 StatefulSet
│   ├── redis.yaml              # Redis 7 deployment
│   └── canary/                 # Istio + Flagger canary deployment
│       ├── virtualservice.yaml
│       ├── destinationrule.yaml
│       └── canary.yaml
├── helm/chaincommand/          # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/              # 7 templated manifests
├── serving/                    # BentoML model serving
│   ├── bentofile.yaml
│   └── service.py              # forecast / detect_anomalies / optimize
├── monitoring/                 # Observability stack
│   ├── prometheus.yml
│   ├── docker-compose.monitoring.yaml
│   └── grafana/                # 9-panel dashboard
├── pipelines/                  # Airflow orchestration
│   ├── dags/
│   │   ├── chaincommand_training.py      # Weekly ML training pipeline
│   │   └── chaincommand_monitoring.py    # 6-hourly drift detection
│   └── docker-compose.airflow.yaml
├── mlflow/                     # Model registry
│   └── docker-compose.mlflow.yaml
├── terraform/                  # AWS infrastructure as code
│   ├── main.tf                 # VPC + EKS + RDS + ElastiCache + S3
│   ├── variables.tf
│   ├── outputs.tf
│   ├── modules/                # eks / rds / redis / s3
│   └── environments/           # dev / prod configs
├── loadtests/                  # Performance testing
│   ├── k6_health.js
│   ├── k6_api.js               # Ramp to 100 VUs, P95 < 500ms
│   └── slo.yaml                # SLO definitions
└── data_quality/               # Great Expectations
    ├── great_expectations.yml
    ├── expectations/            # demand_history + inventory_status
    ├── checkpoints/
    └── validate.py
```

### Phase 1 — Minimum Viable Deployment

| Component | Technology | Details |
|-----------|-----------|---------|
| **Container Orchestration** | Kubernetes | 2-replica deployment, liveness/readiness probes, HPA (2-10 pods, CPU 70%) |
| **Helm Chart** | Helm v3 | Parameterized values for all environments |
| **Model Serving** | BentoML | Dedicated inference service: `forecast`, `detect_anomalies`, `optimize` |
| **Database** | PostgreSQL 16 | StatefulSet with 1Gi persistent volume |
| **Cache** | Redis 7 | In-memory cache for feature store & sessions |
| **Secrets** | K8s Secrets | Base64-encoded, env-injected credentials |

### Phase 2 — Production Ready

| Component | Technology | Details |
|-----------|-----------|---------|
| **Model Registry** | MLflow | Version management, stage transitions (staging → production → archived) |
| **Metrics** | Prometheus | 13 custom metrics (`chaincommand_*`): request rate, latency, KPIs, token budget, circuit breaker, app info |
| **Dashboards** | Grafana | 9-panel dashboard: request rate, latency P50/P95/P99, agents, errors, KPIs, simulation, tokens |
| **Canary Deployment** | Istio + Flagger | 10% step weight, max 50%, success rate > 99%, P99 < 500ms |
| **Pipeline Orchestration** | Apache Airflow | Training DAG (weekly, 8 tasks) + Monitoring DAG (6-hourly drift detection) |

### Phase 3 — Enterprise Grade

| Component | Technology | Details |
|-----------|-----------|---------|
| **Infrastructure as Code** | Terraform | AWS: VPC, EKS, RDS PostgreSQL, ElastiCache Redis, S3 (dev + prod environments) |
| **Access Control** | RBAC Middleware | 3 roles (Viewer/Operator/Admin), 11 permissions, FastAPI middleware |
| **Audit Trail** | Audit Logger | Structured logging: user, action, resource, result, IP, user-agent |
| **Load Testing** | k6 | Ramp to 100 VUs, thresholds: P95 < 500ms, error rate < 1% |
| **SLO** | YAML definitions | API availability 99.9%, latency P95, forecast accuracy |
| **Data Quality** | Great Expectations | Demand history (13 rules) + inventory status (13 rules) validation |

### Quick Start — Local Infrastructure

```bash
# Core services (app + PostgreSQL + Redis)
docker compose up -d

# Monitoring (Prometheus + Grafana)
docker compose -f monitoring/docker-compose.monitoring.yaml up -d
# → Grafana: http://localhost:3000 (admin/changeme)

# MLflow Model Registry
docker compose -f mlflow/docker-compose.mlflow.yaml up -d
# → MLflow: http://localhost:5000

# Airflow Pipeline Orchestration
docker compose -f pipelines/docker-compose.airflow.yaml up -d
# → Airflow: http://localhost:8080 (admin/admin)

# Kubernetes (local)
minikube start
kubectl apply -f k8s/
# Or with Helm:
helm install chaincommand helm/chaincommand/

# Load Testing
k6 run loadtests/k6_api.js

# Data Quality Validation
python data_quality/validate.py
```

### Cloud Deployment (AWS)

```bash
cd terraform/environments/dev
terraform init
terraform plan
terraform apply
```

---

<div align="center">

**Built with agents, driven by autonomy.**

</div>
