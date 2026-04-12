"""Core IncidentOps environment — reset / step / state.

Implements the standard OpenEnv lifecycle:
  reset(task_name)  → IncidentObservation
  step(action)      → (observation, reward, done, info)
  state()           → full internal dict

Reward shaping:
  - Granular partial rewards for diagnosis, technical fixes, communication
  - Penalties for wrong actions (restarting non-root-cause service)
  - Penalties for failing to communicate when complaints are rising
  - Progressive time penalties for taking too long
  - Clear progress signals toward task completion
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import EnvironmentState, IncidentAction, IncidentObservation
from server.scenarios import load_scenario, TASK_CONFIGS
from server.rules import validate_action, apply_action, apply_time_effects
from server.graders import grade_task


ALL_ACTIONS = [
    "inspect_service",
    "inspect_logs",
    "restart_service",
    "rollback_service",
    "scale_service",
    "enable_autoscaling",
    "failover_database",
    "send_status_update",
    "send_vip_update",
    "prioritize_customers",
    "escalate_to_human",
    "resolve_incident",
    "wait",
]


def _get_available_actions(state: EnvironmentState) -> list[str]:
    """Return dynamically filtered actions based on current state."""
    actions = []
    for a in ALL_ACTIONS:
        # Filter out already-completed one-time actions
        if a == "send_status_update" and state.operations.status_page_updated:
            continue
        if a == "send_vip_update" and state.operations.vip_outreach_sent:
            continue
        if a == "escalate_to_human" and state.operations.human_escalated:
            continue
        # Filter failover if DB is already healthy with low connections
        if a == "failover_database" and state.database.status == "healthy" and state.database.connections_pct < 0.60:
            continue
        actions.append(a)
    return actions


class IncidentEnvironment:
    """Step-based incident-command training environment."""

    def __init__(self) -> None:
        self._state: EnvironmentState | None = None
        self._task_name: str = ""

    # ── reset ────────────────────────────────────────────────
    def reset(self, task_name: str = "single_service_outage") -> IncidentObservation:
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(TASK_CONFIGS.keys())}"
            )
        self._task_name = task_name
        self._state = load_scenario(task_name)
        return self._observation()

    # ── step ─────────────────────────────────────────────────
    def step(
        self, action: IncidentAction,
    ) -> tuple[IncidentObservation, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            return self._observation(), 0.0, True, {"error": "Episode finished"}

        s = self._state

        # — validate —
        err = validate_action(s, action)
        if err:
            s.last_action_error = err
            s.step_count += 1
            penalty = -0.05
            s.rewards.append(penalty)
            apply_time_effects(s)
            if s.step_count >= s.max_steps:
                s.done = True
            info: dict = {"error": err, "step": s.step_count}
            if s.done:
                info["score"] = grade_task(self._task_name, s)
            return self._observation(), penalty, s.done, info

        # — apply action —
        message, reward = apply_action(s, action)
        s.last_action_error = None

        # — partial progress milestone bonus —
        # Reward when a service transitions to healthy for the first time
        for svc_name, svc in s.services.items():
            if (svc.status == "healthy"
                    and svc_name not in s.services_recovered):
                s.services_recovered.append(svc_name)
                reward += 0.03  # milestone bonus

        # — base time penalty (−0.02 per step) —
        reward -= 0.02

        # — progressive delay penalty: extra cost for late steps —
        step_ratio = s.step_count / max(s.max_steps, 1)
        if step_ratio > 0.7:
            delay_penalty = -0.03 * (step_ratio - 0.7) / 0.3
            reward += delay_penalty
            s.delay_penalty_total += abs(delay_penalty)

        # — communication urgency penalty —
        # If complaints > 200 and VIP > 5 but no status page update yet
        if (s.customers.complaint_count > 200
                and not s.operations.status_page_updated
                and action.action_type not in ("send_status_update", "send_vip_update")
                and s.step_count > 2):
            reward -= 0.03
            s.communication_overdue = True

        # — SLA breach proximity penalty —
        if s.business.sla_breach_risk > 0.80:
            reward -= 0.02

        # — time effects (skip if already done, e.g. resolve) —
        if not s.done:
            apply_time_effects(s)

        # — increment step —
        s.step_count += 1

        # — episode termination —
        if s.step_count >= s.max_steps and not s.done:
            s.done = True

        s.rewards.append(round(reward, 2))

        info = {"message": message, "step": s.step_count}
        if s.done:
            info["score"] = grade_task(self._task_name, s)

        return self._observation(), round(reward, 2), s.done, info

    # ── state ────────────────────────────────────────────────
    def state(self) -> dict:
        if self._state is None:
            return {}
        return self._state.model_dump()

    # ── score ────────────────────────────────────────────────
    def get_score(self) -> float:
        if self._state is None:
            return 0.0
        return grade_task(self._task_name, self._state)

    # ── observation builder ──────────────────────────────────
    def _observation(self) -> IncidentObservation:
        s = self._state
        assert s is not None

        # service health — now includes CPU, memory, p99
        health: dict = {}
        for name, svc in s.services.items():
            health[name] = {
                "status": svc.status,
                "latency_ms": round(svc.latency_ms, 1),
                "error_rate": round(svc.error_rate, 2),
                "replicas": svc.replicas,
                "cpu_usage_pct": round(svc.cpu_usage_pct, 1),
                "memory_usage_pct": round(svc.memory_usage_pct, 1),
                "response_time_p99": round(svc.response_time_p99, 1),
            }
        health["database"] = {
            "status": s.database.status,
            "connections_pct": round(s.database.connections_pct, 2),
            "replication_lag_ms": round(s.database.replication_lag_ms, 1),
        }

        # recent events (last 5)
        recent: list[str] = []
        start = max(0, len(s.actions_taken) - 5)
        for i in range(start, len(s.actions_taken)):
            recent.append(f"Step {i + 1}: {s.actions_taken[i]}")

        # recent action details (last 5) — includes reward impact
        recent_details: list[dict] = []
        start_d = max(0, len(s.action_details) - 5)
        for i in range(start_d, len(s.action_details)):
            detail = dict(s.action_details[i])
            if i < len(s.rewards):
                detail["reward"] = s.rewards[i]
            recent_details.append(detail)

        # incident summary
        parts = [
            f"[{s.incident.severity.upper()}] {s.incident.title}",
            f"Phase: {s.incident.phase} | Elapsed: {s.incident.minutes_elapsed}min",
        ]
        if s.root_cause_identified:
            parts.append(f"Root cause: {s.incident.root_cause_hint}")

        return IncidentObservation(
            task_name=s.task_name,
            step_count=s.step_count,
            max_steps=s.max_steps,
            incident_summary="\n".join(parts),
            service_health=health,
            customer_impact={
                "complaint_count": s.customers.complaint_count,
                "vip_users_affected": s.customers.vip_users_affected,
                "sentiment_score": round(s.customers.sentiment_score, 2),
                "social_noise_score": round(s.customers.social_noise_score, 2),
            },
            business_impact={
                "revenue_loss_per_min": s.business.revenue_loss_per_min,
                "total_revenue_lost": round(s.business.total_revenue_lost, 2),
                "sla_breach_risk": round(s.business.sla_breach_risk, 2),
                "compliance_risk": round(s.business.compliance_risk, 2),
            },
            operations_status={
                "status_page_updated": s.operations.status_page_updated,
                "vip_outreach_sent": s.operations.vip_outreach_sent,
                "human_escalated": s.operations.human_escalated,
                "incident_bridge_opened": s.operations.incident_bridge_opened,
            },
            dependency_graph=s.dependencies,
            available_actions=_get_available_actions(s),
            recent_events=recent,
            recent_action_details=recent_details,
            last_action_error=s.last_action_error,
        )
