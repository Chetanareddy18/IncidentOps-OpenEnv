"""Action-effect rules engine and time-based degradation logic.

Every action has realistic, dependency-aware consequences.
This is the heart of what makes IncidentOps a real benchmark.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import EnvironmentState, IncidentAction


# Root-cause → primary affected service mapping
ROOT_CAUSE_SERVICE: dict[str, str] = {
    "deployment_regression": "payment_api",
    "ledger_overload": "ledger",
    "database_saturation": "db",
    "memory_leak": "payment_api",
    "network_congestion": "notification_service",
}

# Services that can be targeted by actions
_SERVICE_ACTIONS = {
    "inspect_service", "inspect_logs", "restart_service",
    "rollback_service", "scale_service", "enable_autoscaling",
}


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_action(state: EnvironmentState, action: IncidentAction) -> str | None:
    """Return error message if invalid, else None."""
    valid_targets = list(state.services.keys()) + ["db"]

    if action.action_type in _SERVICE_ACTIONS:
        if not action.target_service:
            return f"Action '{action.action_type}' requires a target_service"
        if action.target_service not in valid_targets:
            return (
                f"Unknown target_service '{action.target_service}'. "
                f"Valid: {valid_targets}"
            )

    if action.action_type == "scale_service":
        delta = action.replica_delta or 2
        if delta < 1 or delta > 5:
            return "replica_delta must be between 1 and 5"

    return None


# ──────────────────────────────────────────────────────────────
# Action effects
# ──────────────────────────────────────────────────────────────

def apply_action(
    state: EnvironmentState,
    action: IncidentAction,
) -> tuple[str, float]:
    """Mutate *state* with the effect of *action*.
    Returns (event_message, step_reward).
    """
    reward = 0.0
    msg = ""
    at = action.action_type
    tgt = action.target_service
    rc = state.incident.root_cause
    rc_svc = ROOT_CAUSE_SERVICE.get(rc, "")

    # ── record action ────────────────────────────────────────
    label = at + (f"({tgt})" if tgt else "")
    state.actions_taken.append(label)
    state.action_details.append(action.model_dump())

    # ── detect conflicting actions ───────────────────────────
    # e.g. restart then rollback same service, or vice versa
    _CONFLICT_PAIRS = {
        ("restart_service", "rollback_service"),
        ("rollback_service", "restart_service"),
        ("scale_service", "enable_autoscaling"),
    }
    if len(state.action_details) >= 2:
        prev = state.action_details[-2]
        prev_at = prev.get("action_type", "")
        prev_tgt = prev.get("target_service")
        if (prev_at, at) in _CONFLICT_PAIRS and prev_tgt == tgt:
            state.conflicting_actions_count += 1
            reward -= 0.06
            msg = (
                f"Conflicting action: {at}({tgt}) contradicts "
                f"previous {prev_at}({prev_tgt}). Wasted step."
            )

    # ── wrong-target penalty (after root cause identified) ──
    if (state.root_cause_identified
            and at in ("restart_service", "rollback_service", "scale_service")
            and tgt and tgt != rc_svc
            and tgt != "db"):
        # Agent knows root cause but is fixing the wrong service
        svc_obj = state.services.get(tgt)
        if svc_obj and svc_obj.status == "healthy":
            # Targeting an already-healthy service after diagnosis
            state.wrong_target_actions += 1
            reward -= 0.05

    # ─────────────── inspect_service ─────────────────────────
    if at == "inspect_service":
        if tgt and tgt not in state.inspected_services:
            state.inspected_services.append(tgt)
            reward += 0.05
            if tgt in state.services:
                s = state.services[tgt]
                msg = (
                    f"Inspected {tgt}: status={s.status}, "
                    f"latency={s.latency_ms:.0f}ms, "
                    f"error_rate={s.error_rate:.2f}, "
                    f"version={s.version}, replicas={s.replicas}"
                )
            elif tgt == "db":
                d = state.database
                msg = (
                    f"Inspected database: status={d.status}, "
                    f"connections={d.connections_pct:.0%}, "
                    f"replication_lag={d.replication_lag_ms:.0f}ms"
                )
            if state.incident.phase == "detect":
                state.incident.phase = "inspect"
        else:
            msg = f"Service '{tgt}' already inspected or unknown."
            reward -= 0.02

    # ─────────────── inspect_logs ────────────────────────────
    elif at == "inspect_logs":
        state.diagnosis_actions += 1
        if state.diagnosis_actions > 3:
            reward -= 0.05
            msg = "Excessive log analysis. Consider taking action."
        elif tgt == rc_svc:
            if not state.root_cause_identified:
                state.root_cause_identified = True
                reward += 0.15
                msg = f"Root cause identified: {state.incident.root_cause_hint}"
            else:
                reward += 0.02
                msg = "Root cause already identified."
        else:
            reward += 0.03
            msg = (
                f"Logs for {tgt} show no anomalies. "
                "Root cause may be in a dependent service."
            )
        if state.incident.phase in ("detect", "inspect"):
            state.incident.phase = "diagnose"

    # ─────────────── restart_service ─────────────────────────
    elif at == "restart_service":
        if tgt and tgt in state.services:
            svc = state.services[tgt]
            n = state.restart_attempts.get(tgt, 0) + 1
            state.restart_attempts[tgt] = n

            if n > 2:
                reward -= 0.08
                msg = (
                    f"Repeated restart of {tgt} (attempt #{n}). "
                    "No further improvement."
                )
            elif rc == "deployment_regression" and tgt == "payment_api":
                svc.status = "degraded"
                svc.latency_ms = max(400, svc.latency_ms * 0.6)
                svc.error_rate = max(0.15, svc.error_rate * 0.5)
                svc.cpu_usage_pct = max(40.0, svc.cpu_usage_pct * 0.7)
                svc.memory_usage_pct = max(35.0, svc.memory_usage_pct * 0.8)
                svc.response_time_p99 = max(800, svc.response_time_p99 * 0.6)
                reward += 0.08
                msg = (
                    f"Restarted {tgt}. Temporary improvement — "
                    "underlying deployment regression persists."
                )
            elif rc == "ledger_overload" and tgt == "payment_api":
                svc.latency_ms = max(svc.latency_ms * 0.9, 800)
                reward -= 0.03
                msg = (
                    f"Restarted {tgt}. Minimal improvement — "
                    "issue appears to be upstream in ledger."
                )
            elif rc == "ledger_overload" and tgt == "ledger":
                svc.status = "healthy"
                svc.latency_ms = 150.0
                svc.error_rate = 0.02
                svc.cpu_usage_pct = 25.0
                svc.memory_usage_pct = 35.0
                svc.response_time_p99 = 300.0
                reward += 0.12
                msg = (
                    f"Restarted {tgt}. Service recovering. "
                    "Dependent services should stabilise."
                )
                # Cascade: payment_api recovers strongly
                if "payment_api" in state.services:
                    pay = state.services["payment_api"]
                    pay.latency_ms = 200.0
                    pay.error_rate = 0.04
                    pay.status = "healthy"
                    pay.cpu_usage_pct = 30.0
                    pay.memory_usage_pct = 40.0
                    pay.response_time_p99 = 400.0
                # Cascade: DB pressure relieved (ledger was hammering DB)
                if state.database.connections_pct > 0.50:
                    state.database.connections_pct = max(
                        0.35, state.database.connections_pct - 0.30,
                    )
                    state.database.replication_lag_ms = max(
                        10, state.database.replication_lag_ms * 0.4,
                    )
            elif rc == "database_saturation":
                if state.database.status == "healthy":
                    # DB already fixed — restart now fully effective
                    svc.status = "healthy"
                    svc.latency_ms = 150.0
                    svc.error_rate = 0.02
                    svc.cpu_usage_pct = 25.0
                    svc.memory_usage_pct = 35.0
                    svc.response_time_p99 = 300.0
                    reward += 0.10
                    msg = (
                        f"Restarted {tgt}. Service recovering well now that "
                        "database pressure is resolved."
                    )
                else:
                    svc.latency_ms = max(svc.latency_ms * 0.85, 300)
                    reward -= 0.02
                    msg = (
                        f"Restarted {tgt}. Limited improvement — "
                        "database pressure is the root cause."
                    )
            elif rc == "memory_leak" and tgt == "payment_api":
                # Restart clears leaked memory — full recovery
                svc.status = "healthy"
                svc.latency_ms = 150.0
                svc.cpu_usage_pct = 25.0
                svc.memory_usage_pct = 35.0
                svc.response_time_p99 = 300.0
                svc.error_rate = 0.02
                reward += 0.18
                msg = (
                    f"Restarted {tgt}. Memory leak cleared — "
                    "service fully recovered! Consider a code fix to prevent recurrence."
                )
                # DB pressure from retries eases
                if state.database.connections_pct > 0.60:
                    state.database.connections_pct = max(
                        0.40, state.database.connections_pct - 0.15,
                    )
            elif rc == "memory_leak" and tgt != "payment_api":
                reward -= 0.02
                msg = (
                    f"Restarted {tgt}. No improvement — "
                    "the memory leak is in payment_api."
                )
            elif rc == "network_congestion" and tgt == "notification_service":
                # Restart re-establishes network connections
                svc.status = "healthy"
                svc.latency_ms = 110.0
                svc.error_rate = 0.02
                svc.cpu_usage_pct = 15.0
                svc.memory_usage_pct = 30.0
                svc.response_time_p99 = 220.0
                reward += 0.15
                msg = (
                    f"Restarted {tgt}. Network connections re-established. "
                    "Timeout storm should subside."
                )
                # Cascade: payment_api stops retrying notifications
                if "payment_api" in state.services:
                    pay = state.services["payment_api"]
                    pay.latency_ms = 250.0
                    pay.error_rate = 0.04
                    pay.status = "healthy"
                    pay.cpu_usage_pct = 30.0
                    pay.memory_usage_pct = 40.0
                    pay.response_time_p99 = 500.0
                # Cascade: ledger also recovers
                if "ledger" in state.services:
                    led = state.services["ledger"]
                    led.latency_ms = 150.0
                    led.error_rate = 0.02
                    led.status = "healthy"
                    led.cpu_usage_pct = 25.0
                    led.memory_usage_pct = 35.0
                    led.response_time_p99 = 300.0
            elif rc == "network_congestion" and tgt != "notification_service":
                svc.latency_ms = max(svc.latency_ms * 0.85, 300)
                reward -= 0.02
                msg = (
                    f"Restarted {tgt}. Minimal improvement — "
                    "the timeout storm originates from notification_service."
                )
            else:
                svc.status = "degraded" if svc.status == "down" else "healthy"
                svc.latency_ms = max(200, svc.latency_ms * 0.5)
                svc.error_rate = max(0.05, svc.error_rate * 0.4)
                reward += 0.10
                msg = f"Restarted {tgt}. Service recovering."
            state.incident.phase = "mitigate"
        else:
            msg = f"Cannot restart '{tgt}' — not a known service."
            reward -= 0.03

    # ─────────────── rollback_service ────────────────────────
    elif at == "rollback_service":
        if tgt and tgt in state.services:
            svc = state.services[tgt]
            if rc == "deployment_regression" and tgt == "payment_api":
                ver = action.rollback_version or svc.previous_version or "v2.2.0"
                svc.status = "healthy"
                svc.latency_ms = 150.0
                svc.error_rate = 0.02
                svc.version = ver
                svc.cpu_usage_pct = 25.0
                svc.memory_usage_pct = 35.0
                svc.response_time_p99 = 300.0
                reward += 0.20
                msg = (
                    f"Rolled back {tgt} to {ver}. "
                    "Payment service fully recovered!"
                )
            elif svc.previous_version:
                svc.latency_ms = max(200, svc.latency_ms * 0.7)
                svc.error_rate = max(0.05, svc.error_rate * 0.6)
                reward += 0.05
                msg = f"Rolled back {tgt}. Marginal improvement."
            else:
                reward -= 0.03
                msg = (
                    f"Rollback of {tgt} had no effect — "
                    "no previous version available."
                )
            state.incident.phase = "mitigate"
        else:
            msg = f"Cannot rollback '{tgt}'."
            reward -= 0.03

    # ─────────────── scale_service ───────────────────────────
    elif at == "scale_service":
        if tgt and tgt in state.services:
            svc = state.services[tgt]
            delta = action.replica_delta or 2
            svc.replicas += delta
            svc.latency_ms = max(100, svc.latency_ms * (1 - 0.15 * delta))

            if state.database.connections_pct > 0.7:
                state.database.connections_pct = min(
                    1.0, state.database.connections_pct + 0.05 * delta,
                )
                if state.database.connections_pct > 0.95:
                    state.database.status = "degraded"
                reward += 0.02
                msg = (
                    f"Scaled {tgt} by +{delta} replicas. "
                    "Warning: DB connection pressure increased."
                )
            else:
                reward += 0.06
                msg = f"Scaled {tgt} by +{delta} replicas. Latency reduced."
            state.incident.phase = "mitigate"
        else:
            msg = f"Cannot scale '{tgt}'."
            reward -= 0.03

    # ─────────────── enable_autoscaling ──────────────────────
    elif at == "enable_autoscaling":
        if tgt and tgt in state.services:
            svc = state.services[tgt]
            svc.replicas = max(svc.replicas, 3)
            svc.latency_ms = max(100, svc.latency_ms * 0.7)
            reward += 0.05
            msg = f"Autoscaling enabled for {tgt}. Replicas adjusted."
            state.incident.phase = "mitigate"
        else:
            msg = f"Cannot enable autoscaling for '{tgt}'."
            reward -= 0.03

    # ─────────────── failover_database ───────────────────────
    elif at == "failover_database":
        db = state.database
        if db.status in ("degraded", "down"):
            db.status = "healthy"
            db.connections_pct = 0.35
            db.replication_lag_ms = 15.0
            reward += 0.15
            msg = (
                "Database failover complete. Connections reset. "
                "DB-dependent services should improve."
            )
            # Cascade improvement — strong recovery for db-dependent services
            for svc_name, deps in state.dependencies.items():
                if "db" in deps and svc_name in state.services:
                    svc = state.services[svc_name]
                    svc.latency_ms = max(150, svc.latency_ms * 0.3)
                    svc.error_rate = max(0.02, svc.error_rate * 0.2)
                    svc.cpu_usage_pct = max(20.0, svc.cpu_usage_pct * 0.5)
                    svc.memory_usage_pct = max(25.0, svc.memory_usage_pct * 0.6)
                    svc.response_time_p99 = max(300.0, svc.response_time_p99 * 0.3)
                    if svc.status == "down":
                        svc.status = "degraded"
                    elif svc.status == "degraded" and svc.error_rate < 0.10:
                        svc.status = "healthy"
        elif db.connections_pct > 0.7:
            db.connections_pct = 0.40
            db.replication_lag_ms = max(10, db.replication_lag_ms * 0.3)
            reward += 0.10
            msg = "Database failover complete. Connection pressure relieved."
        else:
            reward -= 0.03
            msg = "Database failover unnecessary — DB is healthy."
        state.incident.phase = "mitigate"

    # ─────────────── send_status_update ──────────────────────
    elif at == "send_status_update":
        if not state.operations.status_page_updated:
            state.operations.status_page_updated = True
            state.customers.sentiment_score = min(
                0.0, state.customers.sentiment_score + 0.12,
            )
            state.customers.social_noise_score = max(
                0.0, state.customers.social_noise_score - 0.10,
            )
            reward += 0.10 if state.customers.complaint_count > 100 else 0.06
            msg = "Status page updated. Customer sentiment improving."
            state.incident.phase = "communicate"
        else:
            reward -= 0.02
            msg = "Status page already updated."

    # ─────────────── send_vip_update ─────────────────────────
    elif at == "send_vip_update":
        if not state.operations.vip_outreach_sent:
            state.operations.vip_outreach_sent = True
            state.customers.vip_users_affected = max(
                0, state.customers.vip_users_affected - 5,
            )
            state.customers.sentiment_score = min(
                0.0, state.customers.sentiment_score + 0.08,
            )
            reward += 0.10 if state.customers.vip_users_affected > 5 else 0.04
            msg = "VIP customer outreach sent. VIP impact reduced."
            state.incident.phase = "communicate"
        else:
            reward -= 0.02
            msg = "VIP outreach already sent."

    # ─────────────── prioritize_customers ────────────────────
    elif at == "prioritize_customers":
        state.customers.complaint_count = max(
            0, int(state.customers.complaint_count * 0.7),
        )
        state.customers.sentiment_score = min(
            0.0, state.customers.sentiment_score + 0.05,
        )
        reward += 0.04
        msg = "Customer priority queue activated. Complaint backlog reduced."
        state.incident.phase = "communicate"

    # ─────────────── escalate_to_human ───────────────────────
    elif at == "escalate_to_human":
        if not state.operations.human_escalated:
            state.operations.human_escalated = True
            state.operations.incident_bridge_opened = True
            team = action.escalation_team or "on-call SRE"

            if state.task_name == "multi_service_incident":
                reward += 0.10
                for svc in state.services.values():
                    svc.error_rate = max(0.05, svc.error_rate * 0.7)
                msg = (
                    f"Escalated to {team}. "
                    "Additional support joining incident bridge."
                )
            elif state.task_name == "dependency_degradation":
                reward += 0.05
                msg = f"Escalated to {team}. Team acknowledges the incident."
            else:
                reward -= 0.05
                msg = (
                    f"Escalated to {team}. "
                    "Unnecessary escalation for this severity level."
                )
            state.incident.phase = "mitigate"
        else:
            reward -= 0.03
            msg = "Already escalated."

    # ─────────────── resolve_incident ────────────────────────
    elif at == "resolve_incident":
        all_healthy = all(
            s.status == "healthy" for s in state.services.values()
        )
        db_ok = state.database.status == "healthy"
        comms_ok = (
            state.operations.status_page_updated
            or state.customers.complaint_count < 30
        )

        # Count how many issues remain for partial-resolve clarity
        issues_remaining = (
            sum(1 for s in state.services.values() if s.status != "healthy")
            + (0 if db_ok else 1)
        )

        if all_healthy and db_ok and comms_ok:
            state.resolved = True
            state.done = True
            reward += 0.15
            msg = (
                "Incident resolved successfully. "
                "All services healthy and stakeholders informed."
            )
        elif all_healthy and db_ok:
            state.resolved = True
            state.done = True
            reward += 0.05
            msg = (
                "Incident resolved technically, but customer "
                "communication was insufficient."
            )
        else:
            state.false_resolved = True
            # Scale penalty by how many issues remain
            penalty = -0.20 - (0.03 * issues_remaining)
            reward += penalty
            state.customers.complaint_count += 50
            state.customers.sentiment_score = max(
                -1.0, state.customers.sentiment_score - 0.15,
            )
            msg = (
                f"FALSE RESOLUTION: {issues_remaining} system(s) still unhealthy. "
                "Incident re-opened with increased customer frustration."
            )
        state.incident.phase = "resolve"

    # ─────────────── wait ────────────────────────────────────
    elif at == "wait":
        reward -= 0.05
        msg = "Waiting… situation continues to degrade."

    return msg, reward


# ──────────────────────────────────────────────────────────────
# Time-based degradation (applied every step)
# ──────────────────────────────────────────────────────────────

def apply_time_effects(state: EnvironmentState) -> None:
    """Advance clock and worsen unresolved issues."""
    state.incident.minutes_elapsed += 3

    unhealthy = sum(
        1 for s in state.services.values() if s.status != "healthy"
    )
    if state.database.status != "healthy":
        unhealthy += 1

    if unhealthy > 0:
        # ── complaints grow ──
        growth = 20 + 15 * unhealthy
        if state.operations.status_page_updated:
            growth = int(growth * 0.5)
        state.customers.complaint_count += growth

        # ── mark communication overdue if complaints high ──
        if (state.customers.complaint_count > 200
                and not state.operations.status_page_updated):
            state.communication_overdue = True

        # ── sentiment decays ──
        decay = 0.03 * unhealthy
        if not state.operations.status_page_updated:
            decay *= 1.5
        state.customers.sentiment_score = max(
            -1.0, state.customers.sentiment_score - decay,
        )

        # ── social noise grows ──
        state.customers.social_noise_score = min(
            1.0, state.customers.social_noise_score + 0.03,
        )

        # ── revenue accumulates ──
        state.business.total_revenue_lost += (
            state.business.revenue_loss_per_min * 3
        )

        # ── SLA risk grows ──
        state.business.sla_breach_risk = min(
            1.0, state.business.sla_breach_risk + 0.04,
        )

        # ── compliance risk grows under prolonged incidents ──
        if state.incident.minutes_elapsed > 30:
            state.business.compliance_risk = min(
                1.0, state.business.compliance_risk + 0.02,
            )

        # ── services degrade further (including resource metrics) ──
        for svc_name, svc in state.services.items():
            if svc.status == "degraded":
                svc.error_rate = min(0.90, svc.error_rate + 0.02)
                svc.latency_ms = min(5000, svc.latency_ms * 1.05)
                svc.cpu_usage_pct = min(100.0, svc.cpu_usage_pct + 2.0)
                svc.memory_usage_pct = min(100.0, svc.memory_usage_pct + 1.5)
                svc.response_time_p99 = min(15000, svc.response_time_p99 * 1.08)
            elif svc.status == "down":
                svc.cpu_usage_pct = min(100.0, svc.cpu_usage_pct + 1.0)
                svc.memory_usage_pct = min(100.0, svc.memory_usage_pct + 0.5)

            # dependency cascade
            for dep in state.dependencies.get(svc_name, []):
                if dep == "db" and state.database.status != "healthy":
                    svc.latency_ms = min(5000, svc.latency_ms * 1.08)
                    svc.error_rate = min(0.95, svc.error_rate + 0.03)
                elif dep in state.services and state.services[dep].status != "healthy":
                    svc.latency_ms = min(5000, svc.latency_ms * 1.05)
                    svc.error_rate = min(0.95, svc.error_rate + 0.02)
    else:
        # all healthy — things slowly recover
        state.customers.complaint_count = max(
            0, state.customers.complaint_count - 15,
        )
        state.customers.sentiment_score = min(
            0.5, state.customers.sentiment_score + 0.02,
        )
        state.business.sla_breach_risk = max(
            0.0, state.business.sla_breach_risk - 0.05,
        )
        state.business.revenue_loss_per_min = max(
            0, state.business.revenue_loss_per_min * 0.3,
        )
        # Resource metrics recover
        for svc in state.services.values():
            svc.cpu_usage_pct = max(15.0, svc.cpu_usage_pct * 0.9)
            svc.memory_usage_pct = max(20.0, svc.memory_usage_pct * 0.95)
            svc.response_time_p99 = max(svc.latency_ms * 2, svc.response_time_p99 * 0.85)

    # ── update service status thresholds ──
    for svc in state.services.values():
        if svc.error_rate < 0.08 and svc.latency_ms < 400:
            svc.status = "healthy"
        elif svc.error_rate < 0.40 and svc.latency_ms < 2000:
            svc.status = "degraded"
        else:
            if svc.status != "down":
                svc.status = "down"

    # ── update DB status ──
    db = state.database
    if db.connections_pct < 0.80 and db.replication_lag_ms < 80:
        db.status = "healthy"
    elif db.connections_pct < 0.92 and db.replication_lag_ms < 250:
        db.status = "degraded"
    else:
        db.status = "down" if db.connections_pct > 0.95 else "degraded"
