---
title: IncidentOps OpenEnv
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI incident management benchmark with 5 tasks
tags:
  - incident-management
  - reinforcement-learning
  - benchmark
  - fintech
  - llm-agents
---

<div align="center">

# 🚨 IncidentOps OpenEnv

**A simulation environment where AI agents learn to handle real production outages — fast.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-90%20passed-brightgreen.svg)](#)
[![GitHub](https://img.shields.io/badge/GitHub-IncidentOps-black?logo=github)](https://github.com/Chetanareddy18/IncidentOps-OpenEnv)

</div>

---

## What is this?

Think of it like a **flight simulator — but for AI engineers**.

When a fintech app crashes at 3 AM, every minute costs thousands. An on-call engineer has to diagnose which service broke, figure out why, fix it, and keep customers informed — all under pressure.

**IncidentOps puts an AI in that seat.**

It simulates a realistic fintech production system that can fail in different ways. An AI agent gets paged, sees real metrics (latency, error rates, CPU, memory), and has to take the right actions — rollback a bad deploy, fix the root cause, send status updates — before the SLA breach hits.

> **5 scenarios. 13 actions. Dense rewards. A live war-room dashboard. 90 tests, all passing.**

---

## Why we built this

Incident response is one of the hardest workflows to automate. It's not just about fixing the broken service — it requires:

- 🔍 **Diagnosing** root causes across interconnected services
- ⚡ **Prioritising** under time pressure with revenue on the line
- 📣 **Communicating** with customers and stakeholders at the right time
- ⚖️ **Balancing** technical recovery against business impact

No existing benchmark tested all of this together. So we built one.

---

## Environment at a Glance

| Property | Value |
|---|---|
| **Observation** | Structured JSON — service health (CPU/memory/p99), customer impact, business impact, dependency graph |
| **Action space** | 13 typed actions (`rollback_version`, `escalation_team`, `replica_delta`, and more) |
| **Reward** | Dense per-step + final normalised score ∈ [0.0, 1.0] |
| **Tasks** | 5 scenarios (easy → hard) |
| **Episode length** | 8–12 steps depending on difficulty |
| **Deterministic** | Yes — same actions always produce the same outcome |

---

## Observation Space

Each step returns a structured observation:

```json
{
  "task_name": "single_service_outage",
  "step_count": 2,
  "max_steps": 8,
  "incident_summary": "[SEV2] Payment gateway outage after deploy\nPhase: diagnose | Elapsed: 10min",
  "service_health": {
    "payment_api": {
      "status": "down", "latency_ms": 2400, "error_rate": 0.71,
      "replicas": 2, "cpu_usage_pct": 85.0, "memory_usage_pct": 60.0,
      "response_time_p99": 4800.0
    },
    "ledger": {"status": "healthy", "latency_ms": 120, "error_rate": 0.02, "replicas": 2, "cpu_usage_pct": 25.0, "memory_usage_pct": 35.0, "response_time_p99": 240.0},
    "database": {"status": "healthy", "connections_pct": 0.45, "replication_lag_ms": 10}
  },
  "customer_impact": {"complaint_count": 120, "vip_users_affected": 3, "sentiment_score": -0.33, "social_noise_score": 0.20},
  "business_impact": {"revenue_loss_per_min": 800, "total_revenue_lost": 5600, "sla_breach_risk": 0.39, "compliance_risk": 0.05},
  "operations_status": {"status_page_updated": false, "vip_outreach_sent": false, "human_escalated": false},
  "dependency_graph": {"payment_api": ["ledger", "db"], "ledger": ["db"], "notification_service": []},
  "available_actions": ["inspect_service", "rollback_service", "send_status_update", "..."],
  "recent_events": ["Step 1: inspect_service(payment_api)", "Step 2: inspect_logs(payment_api)"],
  "recent_action_details": [{"action_type": "inspect_service", "target_service": "payment_api", "reward": 0.03}]
}
```

---

## Action Space

| Action | Parameters | Effect |
|---|---|---|
| `inspect_service` | `target_service` | Reveals detailed health metrics (CPU, memory, p99) |
| `inspect_logs` | `target_service` | Reveals root cause hint (if correct service) |
| `restart_service` | `target_service` | Restarts — effective if root cause matches |
| `rollback_service` | `target_service`, `rollback_version` | Rolls back to specified version — strong fix for regressions |
| `scale_service` | `target_service`, `replica_delta` | Adds replicas — may worsen DB pressure |
| `enable_autoscaling` | `target_service` | Enables auto-scaling |
| `failover_database` | — | Resets DB connections — cascading improvement |
| `send_status_update` | — | Updates status page, reduces complaint growth |
| `send_vip_update` | — | Notifies VIP customers specifically |
| `prioritize_customers` | — | Reduces complaint backlog |
| `escalate_to_human` | `escalation_team` | Escalates — valuable in hard tasks only |
| `resolve_incident` | — | Closes incident — penalised if premature |
| `wait` | — | Does nothing — situation degrades |

**Dynamic actions**: The `available_actions` field in observations is dynamically filtered — one-time actions (e.g., `send_status_update`) are removed after completion.

---

## How Scoring Works

### Per-step reward

```
reward = (
    diagnosis_reward          # +0.05–0.15 for inspecting affected services/logs
  + technical_improvement     # +0.08–0.20 for fixing the root cause
  + communication_reward      # +0.06–0.10 for status/VIP updates
  + customer_impact_reward    # +0.04 for reducing complaints
  - base_time_penalty         # −0.02 per step (time is money)
  - progressive_delay_penalty # −0.01 to −0.03 past 70% of step budget
  - communication_neglect     # −0.03 if complaints > 200 and no status update
  - sla_breach_penalty        # −0.02 if SLA breach risk > 80%
  - invalid_action_penalty    # −0.05 for invalid actions
  - harmful_action_penalty    # −0.08 for repeated no-ops, −0.20 for false resolve
)
```

### Final score ∈ [0.0, 1.0]

| Component | Weight | What it measures |
|---|---|---|
| Technical recovery | 30% | Service health, CPU/memory/latency restored |
| Customer impact | 20% | Complaints reduced, VIP handled, sentiment |
| Communication | 15% | Status page updated, VIP outreach sent |
| Action efficiency | 15% | Steps used vs steps available |
| Resolution correctness | 10% | Properly resolved, no false close |
| Business impact | 10% | SLA breach risk, revenue loss, compliance |

### Anti-cheat protections
- Spamming restarts → penalised after 2nd attempt (−0.08)
- Closing incident prematurely → heavy penalty (−0.20, +50 complaints)
- Communicating without fixing → score is capped
- Fixing without communicating → can't reach max score on medium/hard

---

## The 5 Scenarios

| # | Task | Difficulty | Services affected | Root cause |
|---|---|---|---|---|
| 1 | `single_service_outage` | 🟢 Easy | 1 | Bad deploy → rollback fixes it |
| 2 | `dependency_degradation` | 🟡 Medium | 3 | Upstream ledger overload, competing priorities |
| 3 | `memory_leak_degradation` | 🟡 Medium | 1 (+DB pressure) | Memory leak — restart, not rollback |
| 4 | `multi_service_incident` | 🔴 Hard | 3+ | DB saturation, must sequence correctly |
| 5 | `cascading_timeout_storm` | 🔴 Hard | 3+ (reverse) | Notification service → payment → DB cascade |

Each scenario tests a fundamentally different reasoning pattern. An agent that memorises one fix will fail on the others.

### Scenario details

**🟢 Easy — Single Service Outage**
Payment API fails after a deploy. Agent should: inspect → rollback → communicate → resolve. Straightforward, but a good baseline.

**🟡 Medium — Dependency Degradation**
Payment API is degraded, but the real problem is upstream ledger pressure. Two services need attention. Agent must prioritise, not just fix the first thing it sees.

**🟡 Medium — Memory Leak**
Payment API is slowly dying from a memory leak (memory at 92%). Naïve agents try a rollback — it won't help. The fix is a restart to clear the leaked memory.

**🔴 Hard — Multi-Service Incident**
Payment down, ledger degraded, DB near saturation. Agent must triage, sequence correctly (DB first), communicate without false reassurance, and possibly escalate.

**🔴 Hard — Cascading Timeout Storm**
The root cause is `notification_service` — an "independent" service nobody suspects. Its failures cascade backwards through payment → ledger → DB. Agent must trace the cascade in reverse.

---

## Service Dependency Graph

```
payment_api ──→ ledger ──→ database
     │                        ↑
     └────────────────────────┘

notification_service  (independent — but can cascade)
```

Key behaviours built in:
- Restarting `payment_api` won't help if `ledger` is the root cause
- Scaling `payment_api` **increases** DB connection pressure
- `notification_service` works even during payment failures — so communication is always possible
- The full dependency graph is included in every observation

---

## Project Structure

```
Incident_environment/
├── models.py              # Typed Pydantic models (Action, State, Observation)
├── inference.py           # LLM-powered baseline agent
├── client.py              # Python client for the REST API
├── requirements.txt       # Dependencies
├── Dockerfile             # Lean deployment with healthcheck
├── server/
│   ├── app.py             # FastAPI server & REST endpoints
│   ├── environment.py     # Core reset/step/state logic + reward shaping
│   ├── scenarios.py       # 5 deterministic task definitions
│   ├── rules.py           # Action effects, dependency logic, time degradation
│   ├── graders.py         # 6-component scoring [0.0, 1.0]
│   └── templates/
│       └── dashboard.html # Live incident war-room dashboard
└── tests/
    ├── test_env.py         # Environment lifecycle (90 tests, all passing ✅)
    ├── test_graders.py     # Grader determinism & range tests
    ├── test_tasks.py       # Scenario correctness tests
    ├── test_rules.py       # Action-effect rules engine tests
    └── test_integration.py # Full episode integration tests
```

---

## Design Decisions

**Why incident response?**
It's a multi-objective problem under time pressure — exactly where AI agents struggle most. Unlike toy environments, it requires diagnosis, sequencing, communication, and tradeoff management all at once.

**Why deterministic?**
When an agent scores 0.45, that's a repeatable measurement — not a lucky run. This makes IncidentOps suitable for benchmarking and comparing models fairly.

**Why dense rewards with penalties?**
Terminal-only scoring makes credit assignment near-impossible for RL agents. Dense per-step rewards give a learning gradient; the terminal grader gives the authoritative final score.

**Why 5 root causes?**
Each tests a different reasoning pattern — rollback vs restart, upstream vs downstream, memory vs network, direct vs reverse cascade. An agent that memorises one fix will fail on the others.

**Why dependency-aware rules?**
Restarting `payment_api` when `ledger` is broken should visibly fail. Scaling should increase DB pressure. These realistic side-effects force agents to reason about system architecture, not just pattern-match symptoms.

**Why resource metrics (CPU/memory/p99)?**
Status labels (healthy/degraded/down) are too coarse. CPU at 92% hints at a memory leak. p99 latency spikes suggest cascading retries. Richer observations → harder, more realistic reasoning.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Chetanareddy18/IncidentOps-OpenEnv.git
cd IncidentOps-OpenEnv/Incident_environment

# Set up virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run all tests (in another terminal)
pytest tests/ -v

# Run the baseline agent (needs HF_TOKEN in .env)
python inference.py
```

### Docker

```bash
docker build -t incidentops .
docker run -p 7860:7860 incidentops
```

---

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Info & available endpoints |
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Reset environment (`{"task_id": "single_service_outage"}`) |
| `POST` | `/step` | Take an action (`{"action_type": "...", "target_service": "..."}`) |
| `GET` | `/state` | Full internal state |
| `GET` | `/score` | Current grader score |
| `GET` | `/web` | Live incident war-room dashboard |
| `GET` | `/events` | SSE real-time state stream |

---

## Baseline Agent & Scores

The included baseline uses an LLM (Qwen2.5-72B-Instruct via HF Router) that reasons about the incident state and picks actions. Falls back to a deterministic heuristic when no HF_TOKEN is set.

**Strategy:**
1. Inspect the most critical service
2. Inspect logs to identify root cause
3. Fix DB first if database is unhealthy
4. Rollback if it's a deployment regression
5. Restart remaining degraded services
6. Communicate when complaints are high
7. Resolve when all services are healthy

### Scores (LLM: Qwen2.5-72B-Instruct)

| Task | Difficulty | Score | Steps |
|---|---|---|---|
| `single_service_outage` | 🟢 Easy | **0.93** | 7 |
| `dependency_degradation` | 🟡 Medium | **0.53** | 10 |
| `memory_leak_degradation` | 🟡 Medium | **1.00** | 4 |
| `multi_service_incident` | 🔴 Hard | **0.96** | 7 |
| `cascading_timeout_storm` | 🔴 Hard | **0.88** | 7 |
| **Average** | | **0.86** | |

---

## Example Episode (Easy Task)

```
[START] task=single_service_outage env=incidentops model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect_service('payment_api') reward=0.03 done=false error=null
[STEP] step=2 action=rollback_service('payment_api', 'v2.2.0') reward=0.18 done=false error=null
[STEP] step=3 action=send_status_update reward=0.08 done=false error=null
[STEP] step=4 action=inspect_service('ledger') reward=0.03 done=false error=null
[STEP] step=5 action=inspect_logs('ledger') reward=0.01 done=false error=null
[STEP] step=6 action=restart_service('ledger') reward=0.08 done=false error=null
[STEP] step=7 action=resolve_incident reward=0.13 done=true error=null
---

## Example Run

**Easy task — agent solves it in 7 steps:**

```
[START] task=single_service_outage
[STEP 1] inspect_service('payment_api')       → reward=0.03
[STEP 2] rollback_service('payment_api', v2.2.0) → reward=0.18
[STEP 3] send_status_update                    → reward=0.08
[STEP 4] inspect_service('ledger')             → reward=0.03
[STEP 5] inspect_logs('ledger')                → reward=0.01
[STEP 6] restart_service('ledger')             → reward=0.08
[STEP 7] resolve_incident                      → reward=0.13
[END] score=0.93 ✅
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENV_BASE_URL` | `http://localhost:7860` | Server URL |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `HF_TOKEN` | — | **Required** for LLM inference |

---

## What's next

- Stochastic mode — add noise to metrics for robustness testing
- Multi-agent — separate commander and communication roles
- Custom scenario builder — define your own incident templates
- Curriculum learning — auto-progress through difficulty levels

---

---

**Built with care by Chetana, Varshini & Nandini**

[⭐ Star on GitHub](https://github.com/Chetanareddy18/IncidentOps-OpenEnv) · [MIT License](LICENSE)
