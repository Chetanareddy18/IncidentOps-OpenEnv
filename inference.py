#!/usr/bin/env python3
"""IncidentOps OpenEnv - LLM-powered incident commander inference.

Uses the OpenAI client (via HF router) to reason about incident
state and choose optimal actions.  Falls back to a deterministic
heuristic when the LLM is unavailable.

Stdout format (validator-safe):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from dotenv import load_dotenv
load_dotenv()

import requests
from openai import OpenAI

# ---------------------------------------------------------------
# Environment variables - per hackathon checklist
# ---------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "incidentops"

TASKS = [
    "single_service_outage",
    "dependency_degradation",
    "multi_service_incident",
    "memory_leak_degradation",
    "cascading_timeout_storm",
]

# ---------------------------------------------------------------
# OpenAI client - configured via hackathon env vars
# ---------------------------------------------------------------

client = None
if HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI Incident Commander managing a live production outage at a fintech company.

Analyze the current incident state AND your previous action history to choose the SINGLE best next action.

DEPENDENCY GRAPH: payment_api depends on [ledger, db], ledger depends on [db], notification_service is independent.

ROOT CAUSE PATTERNS:
- "deployment" or "regression" → rollback_service(payment_api, rollback_version="v2.2.0")
- "ledger" overload → restart_service(ledger), then payment_api will cascade-recover
- "database" saturation → failover_database FIRST, then restart dependent services
- "memory leak" in payment_api → restart_service(payment_api) clears the leak
- "network" or "timeout storm" from notification_service → restart_service(notification_service), cascades fix payment_api and ledger

RESOURCE METRICS: Check cpu_usage_pct, memory_usage_pct, and response_time_p99 for clues:
- High memory (>80%) on payment_api suggests memory leak
- High CPU (>90%) on ledger suggests overload
- High p99 response times indicate cascading latency

STRATEGY:
1. Step 1-2: INSPECT the most critical service + inspect_logs on the likely root cause
2. Step 2-4: FIX the root cause (fix deepest dependency first: db → ledger → payment_api → notification)
3. Step 3-5: COMMUNICATE when complaints > 100 or VIP > 5 (send_status_update, send_vip_update)
4. Step 4+: After root cause fixed, restart any remaining degraded/down services
5. On multi-service incidents (3+ services): escalate_to_human early for bonus reward
6. RESOLVE only when ALL services healthy AND status page updated

CRITICAL RULES:
- Fix ROOT CAUSES not symptoms. Restarting payment_api won't help if ledger is the issue
- Do NOT repeat actions that already succeeded (check your action history below)
- False resolution is HEAVILY penalized (-0.20). Only resolve when truly ready
- For "db" issues, use failover_database, NOT restart_service(db)
- Be efficient — fewer steps = higher score. Don't waste steps on redundant inspections
- Never use "wait" — the situation always degrades
- COMMUNICATE EARLY — failing to communicate when complaints > 200 incurs a penalty each step

ACTIONS:
- inspect_service(target_service), inspect_logs(target_service)
- restart_service(target_service), rollback_service(target_service, rollback_version)
- scale_service(target_service, replica_delta), enable_autoscaling(target_service)
- failover_database
- send_status_update, send_vip_update, prioritize_customers
- escalate_to_human(escalation_team)
- resolve_incident

IMPORTANT: Valid target_service values are EXACTLY: payment_api, ledger, notification_service, db
Use "db" NOT "database". Use "payment_api" NOT "payment" or "payment-api".

RESPOND WITH ONLY a JSON object:
{"action_type": "...", "target_service": "...", "replica_delta": 2, "rollback_version": "v2.2.0", "escalation_team": "on-call SRE"}
Include only the fields needed for your chosen action."""


def build_observation_prompt(obs: dict, step: int, action_history: list[dict] | None = None) -> str:
    """Build a detailed prompt showing current incident state and action history."""
    summary = obs.get("incident_summary", "No summary")
    health = obs.get("service_health", {})
    cust = obs.get("customer_impact", {})
    biz = obs.get("business_impact", {})
    ops = obs.get("operations_status", {})
    deps = obs.get("dependency_graph", {})
    recent = obs.get("recent_events", [])
    max_steps = obs.get("max_steps", 12)
    avail = obs.get("available_actions", [])
    error = obs.get("last_action_error")

    remaining = max_steps - step
    urgency = "CRITICAL" if remaining <= 2 else "HIGH" if remaining <= 4 else "NORMAL"

    lines = [
        f"CURRENT INCIDENT STATE (Step {step + 1} of {max_steps}) — Urgency: {urgency}, {remaining} steps remaining:",
        "",
        "=== INCIDENT ===",
        summary,
        "",
        "=== SERVICE HEALTH (with resource metrics) ===",
        json.dumps(health, indent=2),
        "",
        "=== DEPENDENCY GRAPH ===",
        json.dumps(deps, indent=2) if deps else "payment_api → [ledger, db], ledger → [db], notification_service → []",
        "",
        "=== CUSTOMER IMPACT ===",
        f"- Complaints: {cust.get('complaint_count', 0)}",
        f"- VIP users affected: {cust.get('vip_users_affected', 0)}",
        f"- Sentiment: {cust.get('sentiment_score', 0)} (-1.0 to 1.0)",
        f"- Social noise: {cust.get('social_noise_score', 0)} (0 to 1.0)",
        "",
        "=== BUSINESS IMPACT ===",
        f"- Revenue loss/min: ${biz.get('revenue_loss_per_min', 0):,.0f}",
        f"- Total lost: ${biz.get('total_revenue_lost', 0):,.0f}",
        f"- SLA breach risk: {biz.get('sla_breach_risk', 0):.0%}",
        f"- Compliance risk: {biz.get('compliance_risk', 0):.0%}",
        "",
        "=== OPERATIONS STATUS ===",
        f"- Status page updated: {ops.get('status_page_updated', False)}",
        f"- VIP outreach sent: {ops.get('vip_outreach_sent', False)}",
        f"- Escalated: {ops.get('human_escalated', False)}",
        "",
        "=== AVAILABLE ACTIONS ===",
        ", ".join(avail) if avail else "all actions",
    ]

    # Action history with rewards — helps LLM learn from the episode
    if action_history:
        lines.append("")
        lines.append("=== YOUR ACTION HISTORY (do NOT repeat successful actions) ===")
        for entry in action_history:
            r = entry.get("reward", 0)
            outcome = "GOOD" if r > 0 else "BAD" if r < -0.03 else "NEUTRAL"
            lines.append(f"  Step {entry['step']}: {entry['action']} → reward={r:+.2f} ({outcome})")
    elif recent:
        lines.append("")
        lines.append("=== RECENT ACTIONS ===")
        lines.extend(recent)

    if error:
        lines.append(f"\nLAST ACTION ERROR: {error}")

    # Tactical guidance based on remaining steps
    if remaining <= 2:
        lines.append("\nURGENT: Very few steps left. If services are healthy, resolve now. If not, prioritise the single highest-impact action.")
    elif remaining <= 4:
        lines.append("\nReminder: Time is running short. Focus on resolution path: fix root cause → communicate → resolve.")

    lines.append("\nWhat is the single best action to take right now? Respond with ONLY a JSON object.")
    return "\n".join(lines)


# ---------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------

def choose_action_llm(obs: dict, step: int, action_history: list[dict] | None = None) -> dict | None:
    """Use the LLM to reason about the incident and choose an action."""
    if client is None:
        return None

    try:
        user_prompt = build_observation_prompt(obs, step, action_history)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        action = json.loads(content)

        # Auto-correct common LLM mistakes
        if action.get("target_service") == "database":
            action["target_service"] = "db"

        valid_actions = [
            "inspect_service", "inspect_logs", "restart_service",
            "rollback_service", "scale_service", "enable_autoscaling",
            "failover_database", "send_status_update", "send_vip_update",
            "prioritize_customers", "escalate_to_human", "resolve_incident",
            "wait",
        ]
        if action.get("action_type") not in valid_actions:
            return None

        return action

    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------
# Heuristic fallback agent
# ---------------------------------------------------------------

def choose_action_heuristic(obs: dict, step: int) -> dict:
    """Deterministic heuristic fallback when LLM is unavailable."""
    health = obs.get("service_health", {})
    cust = obs.get("customer_impact", {})
    ops = obs.get("operations_status", {})
    summary = obs.get("incident_summary", "")
    recent = obs.get("recent_events", [])

    down, degraded = [], []
    for name, info in health.items():
        if name == "database":
            continue
        st = info.get("status", "healthy")
        if st == "down":
            down.append(name)
        elif st == "degraded":
            degraded.append(name)

    db = health.get("database", {})
    db_bad = db.get("status", "healthy") != "healthy"

    past = [e.split(": ", 1)[1] if ": " in e else e for e in recent]
    did = lambda kw: any(kw in a for a in past)

    complaints = cust.get("complaint_count", 0)
    vip = cust.get("vip_users_affected", 0)
    comms_done = ops.get("status_page_updated", False)
    vip_done = ops.get("vip_outreach_sent", False)
    escalated = ops.get("human_escalated", False)

    # Phase 1: Inspect
    if step == 0 and not did("inspect"):
        tgt = (down + degraded + ["payment_api"])[0]
        return {"action_type": "inspect_service", "target_service": tgt}

    if step <= 1 and not did("inspect_logs"):
        if db_bad:
            tgt = "db"
        elif degraded and "ledger" in degraded:
            tgt = "ledger"
        else:
            tgt = (down + degraded + ["payment_api"])[0]
        return {"action_type": "inspect_logs", "target_service": tgt}

    # Phase 2: Fix database first
    if db_bad and not did("failover"):
        return {"action_type": "failover_database"}

    # Phase 3: Root-cause fix
    if "deployment" in summary.lower() or "regression" in summary.lower():
        if down and not did("rollback"):
            return {"action_type": "rollback_service", "target_service": down[0]}

    if "root cause" in summary.lower() and "deploy" in summary.lower():
        if down and not did("rollback"):
            return {"action_type": "rollback_service", "target_service": down[0]}

    for svc in degraded:
        if not did(f"restart_service({svc})"):
            return {"action_type": "restart_service", "target_service": svc}
    for svc in down:
        if not did(f"restart_service({svc})"):
            return {"action_type": "restart_service", "target_service": svc}

    # Phase 4: Communicate
    if complaints > 50 and not comms_done:
        return {"action_type": "send_status_update"}
    if vip > 3 and not vip_done:
        return {"action_type": "send_vip_update"}

    for svc in degraded:
        if not did(f"scale_service({svc})"):
            return {"action_type": "scale_service", "target_service": svc, "replica_delta": 2}

    # Phase 5: Resolve or escalate
    all_ok = not down and not degraded and not db_bad
    if all_ok:
        return {"action_type": "resolve_incident"}

    if not escalated and (len(down) + len(degraded)) >= 2:
        return {"action_type": "escalate_to_human", "escalation_team": "on-call SRE"}

    if not comms_done:
        return {"action_type": "send_status_update"}

    return {"action_type": "wait"}


# ---------------------------------------------------------------
# Combined action selection: LLM first, heuristic fallback
# ---------------------------------------------------------------

def choose_action(obs: dict, step: int, action_history: list[dict] | None = None) -> dict:
    """Try LLM-based reasoning first, fall back to heuristic."""
    action = choose_action_llm(obs, step, action_history)
    if action is not None:
        return action
    return choose_action_heuristic(obs, step)


# ---------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------

def _post(url: str, body: dict) -> dict | None:
    try:
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------

def run_task(task: str) -> None:
    rewards: list[str] = []
    steps = 0
    success = False
    score = 0.0
    action_history: list[dict] = []

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        resp = _post(f"{ENV_BASE_URL}/reset", {"task_id": task})
        if not resp:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            return

        obs = resp.get("observation", {})
        done = False
        max_steps = obs.get("max_steps", 12)

        for step_idx in range(max_steps):
            if done:
                break

            action = choose_action(obs, step_idx, action_history)
            step_resp = _post(f"{ENV_BASE_URL}/step", action)

            if not step_resp:
                rewards.append("0.00")
                steps = step_idx + 1
                act_str = action.get("action_type", "unknown")
                print(
                    f"[STEP] step={steps} action={act_str} "
                    f"reward=0.00 done=false error=connection_error",
                    flush=True,
                )
                break

            obs = step_resp.get("observation", {})
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            info = step_resp.get("info", {})
            error = info.get("error")

            rewards.append(f"{reward:.2f}")
            steps = step_idx + 1

            act_str = action.get("action_type", "unknown")
            tgt = action.get("target_service")
            ver = action.get("rollback_version")
            if tgt and ver:
                act_str += f"('{tgt}', '{ver}')"
            elif tgt:
                act_str += f"('{tgt}')"

            # Track action history for LLM context
            action_history.append({
                "step": steps,
                "action": act_str,
                "reward": reward,
            })

            err_str = str(error) if error else "null"
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={steps} action={act_str} "
                f"reward={reward:.2f} done={done_str} error={err_str}",
                flush=True,
            )

            if done:
                score = info.get("score", 0.0)
                success = True

        if not done:
            sr = _get(f"{ENV_BASE_URL}/score")
            if sr:
                score = sr.get("score", 0.0)

    except Exception:
        traceback.print_exc(file=sys.stderr)

    suc_str = "true" if success else "false"
    rew_str = ",".join(rewards)
    print(
        f"[END] success={suc_str} steps={steps} "
        f"score={score:.2f} rewards={rew_str}",
        flush=True,
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    if HF_TOKEN:
        print(f"Using LLM: {MODEL_NAME} via {API_BASE_URL}", file=sys.stderr)
    else:
        print("Warning: HF_TOKEN not set. Running heuristic-only baseline.", file=sys.stderr)

    for task in TASKS:
        try:
            run_task(task)
        except Exception:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)


if __name__ == "__main__":
    main()
