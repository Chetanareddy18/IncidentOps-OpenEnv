"""Deterministic per-task graders returning scores in [0.0, 1.0].

Scoring rubric per task:
  30% — technical recovery (service health, CPU/memory/latency)
  20% — customer impact reduction (complaints, VIP, sentiment)
  15% — communication correctness (status page, VIP outreach)
  15% — action efficiency (step count)
  10% — resolution correctness (resolved, not false-resolved)
  10% — business impact (SLA breach, revenue, time penalties)
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import EnvironmentState


def _time_penalty(state: EnvironmentState, fast_mins: int, max_mins: int) -> float:
    """Return a negative penalty for slow resolution (0.0 to -0.10)."""
    elapsed = state.incident.minutes_elapsed
    if elapsed <= fast_mins:
        return 0.0
    if elapsed >= max_mins:
        return -0.10
    return -0.10 * (elapsed - fast_mins) / (max_mins - fast_mins)


def _business_impact_score(state: EnvironmentState) -> float:
    """Score business impact management (0.0 to 0.10)."""
    score = 0.0
    # SLA breach managed
    if state.business.sla_breach_risk < 0.50:
        score += 0.04
    elif state.business.sla_breach_risk < 0.75:
        score += 0.02

    # Revenue loss contained
    if state.business.revenue_loss_per_min < 200:
        score += 0.03
    elif state.business.revenue_loss_per_min < 500:
        score += 0.01

    # Compliance risk managed
    if state.business.compliance_risk < 0.15:
        score += 0.03
    elif state.business.compliance_risk < 0.30:
        score += 0.01

    return score


# ──────────────────────────────────────────────────────────────
# Easy: single_service_outage
# ──────────────────────────────────────────────────────────────

def grade_single_service_outage(state: EnvironmentState) -> float:
    score = 0.0

    # Technical recovery (30%)
    pay = state.services.get("payment_api")
    if pay:
        if pay.status == "healthy":
            score += 0.30
        elif pay.status == "degraded" and pay.error_rate < 0.15:
            score += 0.18
        elif pay.status == "degraded":
            score += 0.08

    # Customer impact (20%)
    if state.customers.complaint_count < 150:
        score += 0.10
    elif state.customers.complaint_count < 250:
        score += 0.05
    if state.customers.sentiment_score > -0.40:
        score += 0.10
    elif state.customers.sentiment_score > -0.60:
        score += 0.05

    # Communication (15%)
    if state.operations.status_page_updated:
        score += 0.15

    # Action efficiency (15%)
    if state.step_count <= 5:
        score += 0.15
    elif state.step_count <= 7:
        score += 0.08
    elif state.step_count <= 8:
        score += 0.04

    # Resolution correctness (10%)
    if state.resolved and not state.false_resolved:
        score += 0.10
    elif state.false_resolved:
        score -= 0.10

    # Business impact (10%)
    score += _business_impact_score(state)

    # Time penalty
    score += _time_penalty(state, 15, 30)

    # Conflicting / wrong-target deductions
    score -= 0.03 * state.conflicting_actions_count
    score -= 0.02 * state.wrong_target_actions

    return _clamp(score)


# ──────────────────────────────────────────────────────────────
# Medium: dependency_degradation
# ──────────────────────────────────────────────────────────────

def grade_dependency_degradation(state: EnvironmentState) -> float:
    score = 0.0

    # Technical recovery (30%)
    pay = state.services.get("payment_api")
    led = state.services.get("ledger")
    notif = state.services.get("notification_service")

    if led:
        if led.status == "healthy":
            score += 0.15
        elif led.status == "degraded" and led.error_rate < 0.10:
            score += 0.08
    if pay:
        if pay.status == "healthy":
            score += 0.10
        elif pay.status == "degraded" and pay.error_rate < 0.15:
            score += 0.05
    # Bonus for managing the competing priority (notification_service)
    if notif:
        if notif.status == "healthy":
            score += 0.05
        elif notif.status == "degraded" and notif.error_rate < 0.10:
            score += 0.02

    # Customer impact (20%)
    if state.customers.complaint_count < 250:
        score += 0.08
    elif state.customers.complaint_count < 400:
        score += 0.04
    if state.customers.vip_users_affected < 8:
        score += 0.07
    elif state.customers.vip_users_affected < 15:
        score += 0.03
    if state.customers.sentiment_score > -0.50:
        score += 0.05

    # Communication (15%)
    if state.operations.status_page_updated:
        score += 0.08
    if state.operations.vip_outreach_sent:
        score += 0.07

    # Action efficiency (15%)
    if state.step_count <= 7:
        score += 0.15
    elif state.step_count <= 9:
        score += 0.08
    elif state.step_count <= 10:
        score += 0.04

    # Resolution (10%)
    if state.resolved and not state.false_resolved:
        score += 0.10
    elif state.false_resolved:
        score -= 0.10

    # Root-cause identification bonus (medium task: agent must diagnose correctly)
    if state.root_cause_identified:
        score += 0.03

    # Business impact (10%)
    score += _business_impact_score(state)

    # Time penalty
    score += _time_penalty(state, 20, 40)

    # Conflicting / wrong-target deductions
    score -= 0.03 * state.conflicting_actions_count
    score -= 0.02 * state.wrong_target_actions

    return _clamp(score)


# ──────────────────────────────────────────────────────────────
# Hard: multi_service_incident
# ──────────────────────────────────────────────────────────────

def grade_multi_service_incident(state: EnvironmentState) -> float:
    score = 0.0

    # Technical recovery (30%)
    total = len(state.services)
    healthy = sum(1 for s in state.services.values() if s.status == "healthy")
    degraded = sum(1 for s in state.services.values() if s.status == "degraded")
    tech = (healthy * 1.0 + degraded * 0.3) / max(total, 1)
    score += tech * 0.20

    if state.database.status == "healthy":
        score += 0.10
    elif state.database.status == "degraded" and state.database.connections_pct < 0.70:
        score += 0.05

    # Customer impact (20%)
    if state.customers.complaint_count < 350:
        score += 0.07
    elif state.customers.complaint_count < 500:
        score += 0.03
    if state.customers.sentiment_score > -0.70:
        score += 0.06
    if state.customers.vip_users_affected < 12:
        score += 0.07
    elif state.customers.vip_users_affected < 18:
        score += 0.03

    # Communication (15%)
    if state.operations.status_page_updated:
        score += 0.08
    if state.operations.vip_outreach_sent:
        score += 0.07

    # Action efficiency (15%)
    if state.step_count <= 8:
        score += 0.15
    elif state.step_count <= 10:
        score += 0.08
    elif state.step_count <= 12:
        score += 0.04

    # Resolution (10%)
    if state.resolved and not state.false_resolved:
        score += 0.10
    elif state.false_resolved:
        score -= 0.15

    # Business impact (10%)
    score += _business_impact_score(state)

    # Time penalty
    score += _time_penalty(state, 25, 50)

    # Escalation bonus (hard task)
    if state.operations.human_escalated:
        score += 0.03

    # Conflicting / wrong-target deductions
    score -= 0.03 * state.conflicting_actions_count
    score -= 0.02 * state.wrong_target_actions

    return _clamp(score)


# ──────────────────────────────────────────────────────────────
# Medium: memory_leak_degradation
# ──────────────────────────────────────────────────────────────

def grade_memory_leak_degradation(state: EnvironmentState) -> float:
    score = 0.0

    # Technical recovery (30%)
    pay = state.services.get("payment_api")
    if pay:
        if pay.status == "healthy":
            score += 0.20
        elif pay.status == "degraded" and pay.error_rate < 0.15:
            score += 0.10
        elif pay.status == "degraded":
            score += 0.05

    # DB didn't get worse (10%)
    if state.database.status == "healthy":
        if state.database.connections_pct < 0.65:
            score += 0.10
        else:
            score += 0.05
    elif state.database.status == "degraded":
        score += 0.02

    # Customer impact (20%)
    if state.customers.complaint_count < 200:
        score += 0.08
    elif state.customers.complaint_count < 350:
        score += 0.04
    if state.customers.vip_users_affected < 8:
        score += 0.07
    elif state.customers.vip_users_affected < 12:
        score += 0.03
    if state.customers.sentiment_score > -0.45:
        score += 0.05

    # Communication (15%)
    if state.operations.status_page_updated:
        score += 0.08
    if state.operations.vip_outreach_sent:
        score += 0.07

    # Action efficiency (15%)
    if state.step_count <= 6:
        score += 0.15
    elif state.step_count <= 8:
        score += 0.08
    elif state.step_count <= 10:
        score += 0.04

    # Resolution (10%)
    if state.resolved and not state.false_resolved:
        score += 0.10
    elif state.false_resolved:
        score -= 0.10

    # Root-cause identification bonus
    if state.root_cause_identified:
        score += 0.03

    # Business impact (10%)
    score += _business_impact_score(state)

    # Time penalty
    score += _time_penalty(state, 20, 40)

    # Conflicting / wrong-target deductions
    score -= 0.03 * state.conflicting_actions_count
    score -= 0.02 * state.wrong_target_actions

    return _clamp(score)


# ──────────────────────────────────────────────────────────────
# Hard: cascading_timeout_storm
# ──────────────────────────────────────────────────────────────

def grade_cascading_timeout_storm(state: EnvironmentState) -> float:
    score = 0.0

    # Technical recovery (30%)
    total = len(state.services)
    healthy = sum(1 for s in state.services.values() if s.status == "healthy")
    degraded = sum(1 for s in state.services.values() if s.status == "degraded")
    tech = (healthy * 1.0 + degraded * 0.3) / max(total, 1)
    score += tech * 0.15

    # Notification service specifically (root cause)
    notif = state.services.get("notification_service")
    if notif and notif.status == "healthy":
        score += 0.08
    elif notif and notif.status == "degraded":
        score += 0.03

    # DB recovery
    if state.database.status == "healthy":
        score += 0.07
    elif state.database.status == "degraded" and state.database.connections_pct < 0.70:
        score += 0.03

    # Customer impact (20%)
    if state.customers.complaint_count < 350:
        score += 0.06
    elif state.customers.complaint_count < 500:
        score += 0.03
    if state.customers.sentiment_score > -0.65:
        score += 0.06
    if state.customers.vip_users_affected < 12:
        score += 0.05
    elif state.customers.vip_users_affected < 18:
        score += 0.02
    # Social noise (extra for this task — high noise scenario)
    if state.customers.social_noise_score < 0.50:
        score += 0.03

    # Communication (15%)
    if state.operations.status_page_updated:
        score += 0.08
    if state.operations.vip_outreach_sent:
        score += 0.07

    # Action efficiency (15%)
    if state.step_count <= 8:
        score += 0.15
    elif state.step_count <= 10:
        score += 0.08
    elif state.step_count <= 12:
        score += 0.04

    # Resolution (10%)
    if state.resolved and not state.false_resolved:
        score += 0.10
    elif state.false_resolved:
        score -= 0.15

    # Business impact (10%)
    score += _business_impact_score(state)

    # Time penalty
    score += _time_penalty(state, 25, 50)

    # Escalation bonus (hard task)
    if state.operations.human_escalated:
        score += 0.03

    # Conflicting / wrong-target deductions
    score -= 0.03 * state.conflicting_actions_count
    score -= 0.02 * state.wrong_target_actions

    return _clamp(score)


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

_GRADERS = {
    "single_service_outage": grade_single_service_outage,
    "dependency_degradation": grade_dependency_degradation,
    "multi_service_incident": grade_multi_service_incident,
    "memory_leak_degradation": grade_memory_leak_degradation,
    "cascading_timeout_storm": grade_cascading_timeout_storm,
}


def grade_task(task_name: str, state: EnvironmentState) -> float:
    grader = _GRADERS.get(task_name)
    if grader is None:
        raise ValueError(
            f"No grader for task '{task_name}'. "
            f"Available: {list(_GRADERS.keys())}"
        )
    return round(grader(state), 2)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
