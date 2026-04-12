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
  - openenv
  - incident-management
  - reinforcement-learning
  - benchmark
  - fintech
---

# IncidentOps OpenEnv

**A real-world incident command training environment for AI agents in fintech production operations.**

IncidentOps OpenEnv trains and evaluates AI incident commanders operating in a simulated fintech production system. Agents must triage outages, inspect service dependencies, choose mitigation actions, communicate with affected users, and minimise business impact — all under time pressure. The environment features typed models, deterministic graders, multi-task difficulty progression, dependency-aware action effects, and dense reward shaping with granular penalties.

---

## Motivation

Incident response is one of the highest-stakes workflows in production engineering. A single outage can cost thousands of dollars per minute, erode customer trust, and trigger compliance violations. Today, incident commanders must simultaneously:

- **Diagnose** root causes across dependent services
- **Prioritise** actions under time pressure
- **Communicate** with customers and stakeholders
- **Balance** technical recovery against business impact

IncidentOps captures this multi-objective decision problem as a step-based training environment, making it suitable for RL, LLM-agent, and heuristic evaluation.

---

## Environment Overview

| Property | Value |
|---|---|
| **Observation** | Structured JSON: incident metadata, service health (incl. CPU/memory/p99), customer/business impact, dependency graph, operations flags |
| **Action space** | 13 typed, parameterised actions (with `rollback_version`, `escalation_team`, `replica_delta`) |
| **Reward** | Dense per-step + final normalised score ∈ \[0.0, 1.0\] |
| **Tasks** | 5 (easy / medium / medium / hard / hard) |
| **Episode length** | 8 / 10 / 10 / 12 / 12 steps |
| **Deterministic** | Yes — identical actions produce identical outcomes |

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

## State Transition Mechanics

1. **Action validation** — invalid actions incur a −0.05 penalty
2. **Action effects** — dependency-aware (e.g. restarting `payment_api` won't fix a `ledger` root cause)
3. **Base time penalty** — −0.02 per step
4. **Progressive delay penalty** — extra −0.01 to −0.03 for steps beyond 70% of max_steps
5. **Communication urgency penalty** — −0.03 per step when complaints > 200 and status page not updated
6. **SLA breach proximity penalty** — −0.02 when SLA breach risk > 80%
7. **Time degradation** — every step, unresolved services worsen: complaints grow, sentiment drops, revenue accumulates, SLA risk increases, error rates climb via dependency cascading, CPU/memory increase
8. **Status updates** — healthy thresholds are recalculated after time effects

Anti-exploit protections:
- Repeated restarts penalised (−0.08 after 2nd attempt)
- False resolution strongly penalised (−0.20, +50 complaints)
- Communication alone cannot produce a high score without technical recovery
- Technical recovery alone cannot achieve max score on medium/hard tasks

---

## Reward Function

### Dense Per-Step Reward

```
reward = (
    diagnosis_reward          # +0.05–0.15 for inspecting affected services/logs
  + technical_improvement     # +0.08–0.20 for fixing root cause
  + communication_reward      # +0.06–0.10 for status/VIP updates
  + customer_impact_reward    # +0.04 for complaint reduction
  - base_time_penalty         # −0.02 per step
  - progressive_delay_penalty # −0.01 to −0.03 for late steps (>70% of max)
  - communication_neglect     # −0.03 if complaints > 200 and no status update
  - sla_breach_penalty        # −0.02 if SLA breach risk > 80%
  - invalid_action_penalty    # −0.05 for invalid actions
  - harmful_action_penalty    # −0.08 for repeated no-ops, −0.20 for false resolve
)
```

### Reward Shaping Details

| Reward Type | Value | Condition |
|---|---|---|
| Inspect new service | +0.05 | First inspection of a service |
| Identify root cause | +0.15 | `inspect_logs` on correct service |
| Root cause fix | +0.12–0.20 | Action that addresses actual root cause |
| Wrong service restart | −0.02–0.03 | Restarting a non-root-cause service |
| Status page update | +0.06–0.10 | Higher reward when complaints > 100 |
| VIP outreach | +0.04–0.10 | Higher reward when VIP affected > 5 |
| Correct resolution | +0.15 | All services healthy + communication done |
| False resolution | −0.20 | Resolving with unhealthy services |
| Repeated restart | −0.08 | 3rd+ restart of same service |
| Communication neglect | −0.03/step | Complaints > 200, no status page update |
| Progressive delay | −0.01–0.03 | Steps beyond 70% of episode budget |

### Terminal Grader (Normalised Score ∈ [0.0, 1.0])

| Component | Weight | What it measures |
|---|---|---|
| Technical recovery | 30% | Service health status, CPU/memory/latency restored |
| Customer impact reduction | 20% | Complaints, VIP affected, sentiment score |
| Communication correctness | 15% | Status page updated, VIP outreach sent |
| Action efficiency | 15% | Step count vs max steps |
| Resolution correctness | 10% | Properly resolved, not false-resolved |
| Business impact | 10% | SLA breach risk, revenue loss, compliance risk |

Additional modifiers:
- **Time penalty**: −0.00 to −0.10 based on minutes elapsed
- **Escalation bonus**: +0.03 for hard tasks with escalation

---

## Tasks & Difficulty Progression

| Dimension | Easy | Medium (1) | Medium (2) | Hard (1) | Hard (2) |
|---|---|---|---|---|---|
| **Task** | `single_service_outage` | `dependency_degradation` | `memory_leak_degradation` | `multi_service_incident` | `cascading_timeout_storm` |
| **Root Cause** | Deploy regression | Ledger overload + resource contention | Memory leak | DB saturation | Network congestion |
| **Impacted services** | 1 | 3 (competing priorities) | 1 (+DB pressure) | 3+ | 3+ (reverse cascade) |
| **Root cause obvious?** | Yes | Partly (resource overload misleads) | Misleading | No | No (reverse direction) |
| **Customer pressure** | Low | Medium–High | Medium | High | Very High |
| **Communication needed** | Simple | Important (evolving) | Important | Critical | Critical |
| **Escalation needed** | No | Maybe | No | Often yes | Often yes |
| **Max steps** | 8 | 10 | 10 | 12 | 12 |
| **Baseline score range** | 0.75–0.95 | 0.50–0.85 | 0.60–0.90 | 0.40–0.80 | 0.35–0.75 |

### Easy — `single_service_outage`
Payment API fails after a deploy. Notifications still work. Complaints are rising but manageable. Agent should inspect → rollback → communicate → resolve.

### Medium — `dependency_degradation`
Payment API is degraded, but the real issue is upstream ledger pressure compounded by resource overload. Notification service is also showing early degradation signs, creating a **competing priority** — the agent must decide which service to fix first. CPU/memory metrics on ledger (92% CPU, 85% memory) provide clues about the root cause. Agent must inspect dependencies, fix the root cause, manage competing services, communicate dynamically as the incident evolves, and avoid shallow fixes.

### Medium — `memory_leak_degradation`
Payment API is gradually degrading from a memory leak introduced in v2.4.0. Memory usage at 92% is a key diagnostic clue. DB connections are climbing from retries. Naïve agents may try rollback (won't help) or failover_database (premature). The correct fix is `restart_service(payment_api)` to clear the leaked memory, then communicate. Tests whether agents can distinguish memory leaks from deployment regressions using resource metrics.

### Hard — `multi_service_incident`
Payment down, ledger degraded, DB near saturation, social pressure rising. Agent must triage correctly, sequence actions (DB first), communicate without false reassurance, possibly escalate, and recover within step budget.

### Hard — `cascading_timeout_storm`
Notification service has a network failure. Payment API is retrying notifications → overwhelming ledger → saturating DB. This is a **reverse cascade** — the root cause is in an "independent" service (notification_service), not in the payment pipeline. Agents must trace the cascade backwards, fix notification_service first, then database, then let upstream services recover. Tests non-obvious root cause identification and reverse-dependency reasoning.

---

## Service Dependency Graph

```
payment_api ──→ ledger ──→ db
                           ↑
payment_api ──────────────┘
notification_service (independent)
```

- Restarting `payment_api` won't help if `ledger` is failing
- Scaling `payment_api` may worsen DB connection pressure
- `notification_service` works even when payments don't — so communication is always possible
- The dependency graph is included in every observation for agent reference

---

## Project Structure

```
Incident_environment/
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
├── models.py              # Typed Pydantic models (Action, State, Observation)
├── inference.py            # LLM-powered baseline agent (OpenAI client)
├── client.py               # Python client for the API
├── openenv.yaml            # OpenEnv metadata with success criteria
├── pyproject.toml          # Project config
├── requirements.txt        # Dependencies
├── Dockerfile              # Lean deployment with healthcheck
├── README.md               # This file
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI server (OpenEnv endpoints)
│   ├── environment.py      # Core reset/step/state logic + reward shaping
│   ├── scenarios.py        # 5 deterministic task definitions with resource metrics
│   ├── rules.py            # Action effects, dependency logic, time degradation
│   ├── graders.py          # Deterministic 6-component scoring [0.0, 1.0]
│   └── templates/
│       └── dashboard.html  # Incident war-room dashboard
└── tests/
    ├── __init__.py
    ├── test_env.py          # Environment lifecycle tests
    ├── test_graders.py      # Grader determinism & range tests
    ├── test_tasks.py        # Scenario correctness tests
    ├── test_rules.py        # Action-effect rules engine tests
    └── test_integration.py  # Full episode integration tests
```

---

## Design Decisions

**Why incident response?** Incident management is a multi-objective optimisation problem under time pressure — exactly the kind of task where AI agents struggle most. It requires diagnosis, sequencing, communication, and tradeoff management simultaneously. Unlike toy environments, the policy space is combinatorial and the reward signal is sparse without careful shaping.

**Why deterministic?** Deterministic environments isolate agent quality from environment noise. When an agent scores 0.45, that's a repeatable measurement, not a lucky run. This makes the environment suitable for benchmarking, debugging, and curriculum learning. A stochastic extension is planned.

**Why dense rewards with penalties?** Terminal-only scoring makes credit assignment extremely hard for RL agents. Our dense per-step rewards (diagnosis, technical, communication, penalty) give agents a gradient to learn from while the terminal grader provides the authoritative score. Penalties for communication neglect, progressive delays, and SLA proximity ensure agents learn prioritisation.

**Why 5 root causes?** Each root cause tests a fundamentally different reasoning pattern: rollback vs restart, upstream vs downstream, direct vs reverse cascade, memory vs network. An agent that memorises one fix pattern will fail on others.

**Why dependency-aware rules?** Restarting `payment_api` when `ledger` is the root cause should visibly fail. Scaling a service should increase DB pressure. These realistic side-effects force agents to reason about system architecture, not just symptom-matching.

**Why resource metrics (CPU/memory/p99)?** Service status alone (healthy/degraded/down) is too coarse. CPU and memory usage provide diagnostic clues (e.g., 92% memory on payment_api suggests a memory leak). This makes the observation space more realistic and actionable.

**Why anti-exploit protections?** Without them, agents learn degenerate policies: spam restarts, resolve immediately, or only communicate. Repeated-restart penalties, false-resolution penalties, and communication-only score caps ensure agents must actually solve the problem.

---

## Setup & Usage

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v

# Run baseline inference (in another terminal)
python inference.py
```

### Docker

```bash
docker build -t incidentops .
docker run -p 7860:7860 incidentops
```

### HF Space

Deploy the Dockerfile to a Hugging Face Space. The server starts on port 7860 and responds to `/reset` immediately.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Info & available endpoints |
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | List available tasks with metadata |
| `POST` | `/reset` | Reset environment (`{"task_id": "single_service_outage"}`) |
| `POST` | `/step` | Take an action (`{"action_type": "...", "target_service": "..."}`) |
| `GET` | `/state` | Full internal state |
| `GET` | `/score` | Current grader score |
| `GET` | `/web` | Live incident war-room dashboard |

---

## Baseline Inference

The baseline uses an LLM-powered agent (Qwen2.5-72B-Instruct via HF Router) that reasons about the incident state and selects actions. Falls back to a deterministic heuristic when HF_TOKEN is not set.

1. **Inspect** the most critical service
2. **Inspect logs** on the likely root-cause service
3. **Fix DB first** if database is unhealthy
4. **Rollback** if deployment regression detected
5. **Restart** remaining degraded services
6. **Communicate** when complaints are high
7. **Resolve** when all services are healthy

### Baseline Scores (LLM: Qwen2.5-72B-Instruct)

| Task | Difficulty | Expected Range | Baseline Score | Steps |
|---|---|---|---|---|
| `single_service_outage` | Easy | 0.75–0.95 | **0.93** | 7 |
| `dependency_degradation` | Medium | 0.50–0.85 | **0.53** | 10 |
| `memory_leak_degradation` | Medium | 0.60–0.90 | **1.00** | 4 |
| `multi_service_incident` | Hard | 0.40–0.80 | **0.96** | 7 |
| `cascading_timeout_storm` | Hard | 0.35–0.75 | **0.88** | 7 |
| **Average** | | | **0.86** | |

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
[END] success=true steps=7 score=0.93 rewards=0.03,0.18,0.08,0.03,0.01,0.08,0.13
```

### Example Episode (Medium Task — Dependency Degradation)

```
[START] task=dependency_degradation env=incidentops model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect_service('payment_api') reward=0.03 done=false error=null
[STEP] step=2 action=inspect_logs('ledger') reward=0.13 done=false error=null
[STEP] step=3 action=restart_service('ledger') reward=0.10 done=false error=null
[STEP] step=4 action=send_status_update reward=0.08 done=false error=null
[STEP] step=5 action=send_vip_update reward=0.08 done=false error=null
[STEP] step=6 action=resolve_incident reward=0.13 done=true error=null
[END] success=true steps=6 score=0.78 rewards=0.03,0.13,0.10,0.08,0.08,0.13
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | **Required.** Hugging Face API token for LLM inference |

---

## Future Extensions

- **Stochastic mode** — add noise to service metrics for robustness testing
- **Multi-agent** — separate commander and communication roles
- **Longer episodes** — post-incident review and RCA phases
- **Custom scenarios** — user-defined incident templates
- **Curriculum learning** — auto-progress through difficulty levels

---

*Built by Visionary Coders for the OpenEnv Hackathon.*
