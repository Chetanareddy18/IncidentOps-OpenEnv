"""Tests for action-effect rules engine — the most complex module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import IncidentAction, EnvironmentState
from server.scenarios import load_scenario
from server.rules import validate_action, apply_action, apply_time_effects


# ──────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────

def test_validate_missing_target():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="inspect_service")
    err = validate_action(state, action)
    assert err is not None
    assert "target_service" in err


def test_validate_invalid_target():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="inspect_service", target_service="nonexistent")
    err = validate_action(state, action)
    assert err is not None
    assert "nonexistent" in err


def test_validate_valid_action():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="inspect_service", target_service="payment_api")
    err = validate_action(state, action)
    assert err is None


def test_validate_scale_out_of_range():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="scale_service", target_service="payment_api", replica_delta=10)
    err = validate_action(state, action)
    assert err is not None


def test_validate_db_target():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="inspect_service", target_service="db")
    err = validate_action(state, action)
    assert err is None


# ──────────────────────────────────────────────────────────────
# Action effect tests — deployment_regression
# ──────────────────────────────────────────────────────────────

def test_rollback_fixes_deployment_regression():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="rollback_service", target_service="payment_api")
    msg, reward = apply_action(state, action)
    assert reward > 0
    assert state.services["payment_api"].status == "healthy"
    assert "recovered" in msg.lower() or "rolled back" in msg.lower()


def test_restart_partial_fix_for_deployment():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="restart_service", target_service="payment_api")
    msg, reward = apply_action(state, action)
    assert reward > 0
    assert state.services["payment_api"].status == "degraded"  # not full fix
    assert "regression" in msg.lower()


# ──────────────────────────────────────────────────────────────
# Action effect tests — ledger_overload
# ──────────────────────────────────────────────────────────────

def test_restart_ledger_fixes_overload():
    state = load_scenario("dependency_degradation")
    action = IncidentAction(action_type="restart_service", target_service="ledger")
    msg, reward = apply_action(state, action)
    assert reward > 0
    assert state.services["ledger"].status == "healthy"


def test_restart_ledger_cascades_to_payment():
    state = load_scenario("dependency_degradation")
    old_pay_latency = state.services["payment_api"].latency_ms
    action = IncidentAction(action_type="restart_service", target_service="ledger")
    apply_action(state, action)
    assert state.services["payment_api"].latency_ms < old_pay_latency


def test_restart_payment_useless_when_ledger_bad():
    state = load_scenario("dependency_degradation")
    action = IncidentAction(action_type="restart_service", target_service="payment_api")
    msg, reward = apply_action(state, action)
    assert reward < 0  # penalty — wrong target


# ──────────────────────────────────────────────────────────────
# Action effect tests — database_saturation
# ──────────────────────────────────────────────────────────────

def test_failover_fixes_saturated_db():
    state = load_scenario("multi_service_incident")
    action = IncidentAction(action_type="failover_database")
    msg, reward = apply_action(state, action)
    assert reward > 0
    assert state.database.status == "healthy"
    assert state.database.connections_pct < 0.50


def test_failover_cascades_to_services():
    state = load_scenario("multi_service_incident")
    action = IncidentAction(action_type="failover_database")
    apply_action(state, action)
    # Services depending on db should improve
    for name, deps in state.dependencies.items():
        if "db" in deps and name in state.services:
            assert state.services[name].latency_ms < 2400


# ──────────────────────────────────────────────────────────────
# Action effect tests — memory_leak (new task)
# ──────────────────────────────────────────────────────────────

def test_restart_payment_fixes_memory_leak():
    state = load_scenario("memory_leak_degradation")
    action = IncidentAction(action_type="restart_service", target_service="payment_api")
    msg, reward = apply_action(state, action)
    assert reward > 0.10
    assert state.services["payment_api"].status == "healthy"
    assert "memory leak" in msg.lower()


def test_restart_ledger_useless_for_memory_leak():
    state = load_scenario("memory_leak_degradation")
    action = IncidentAction(action_type="restart_service", target_service="ledger")
    msg, reward = apply_action(state, action)
    # Generic restart (ledger was healthy, so default case)
    assert state.services["payment_api"].status == "degraded"  # unchanged


# ──────────────────────────────────────────────────────────────
# Action effect tests — network_congestion (new task)
# ──────────────────────────────────────────────────────────────

def test_restart_notification_fixes_congestion():
    state = load_scenario("cascading_timeout_storm")
    action = IncidentAction(action_type="restart_service", target_service="notification_service")
    msg, reward = apply_action(state, action)
    assert reward > 0.10
    assert state.services["notification_service"].status == "healthy"


def test_network_congestion_cascades_fix():
    state = load_scenario("cascading_timeout_storm")
    action = IncidentAction(action_type="restart_service", target_service="notification_service")
    apply_action(state, action)
    # payment_api and ledger should improve from cascade
    assert state.services["payment_api"].latency_ms < 1600
    assert state.services["ledger"].status == "healthy"


# ──────────────────────────────────────────────────────────────
# Communication tests
# ──────────────────────────────────────────────────────────────

def test_status_update_improves_sentiment():
    state = load_scenario("single_service_outage")
    old_sent = state.customers.sentiment_score
    action = IncidentAction(action_type="send_status_update")
    msg, reward = apply_action(state, action)
    assert reward > 0
    assert state.operations.status_page_updated is True
    assert state.customers.sentiment_score > old_sent


def test_duplicate_status_update_penalized():
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="send_status_update"))
    _, reward = apply_action(state, IncidentAction(action_type="send_status_update"))
    assert reward < 0


def test_vip_update_reduces_affected():
    state = load_scenario("dependency_degradation")
    old_vip = state.customers.vip_users_affected
    action = IncidentAction(action_type="send_vip_update")
    apply_action(state, action)
    assert state.customers.vip_users_affected < old_vip


# ──────────────────────────────────────────────────────────────
# Escalation tests
# ──────────────────────────────────────────────────────────────

def test_escalation_bonus_hard_task():
    state = load_scenario("multi_service_incident")
    action = IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE")
    _, reward = apply_action(state, action)
    assert reward > 0
    assert state.operations.human_escalated is True
    assert state.operations.incident_bridge_opened is True


def test_escalation_penalty_easy_task():
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE")
    _, reward = apply_action(state, action)
    assert reward < 0  # unnecessary escalation


def test_duplicate_escalation_penalized():
    state = load_scenario("multi_service_incident")
    apply_action(state, IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE"))
    _, reward = apply_action(state, IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE"))
    assert reward < 0


# ──────────────────────────────────────────────────────────────
# Resolution tests
# ──────────────────────────────────────────────────────────────

def test_false_resolve_penalized():
    state = load_scenario("multi_service_incident")
    _, reward = apply_action(state, IncidentAction(action_type="resolve_incident"))
    assert reward < 0
    assert state.false_resolved is True
    assert state.customers.complaint_count > 285  # +50 penalty


def test_valid_resolve_after_fix():
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="rollback_service", target_service="payment_api"))
    apply_action(state, IncidentAction(action_type="send_status_update"))
    _, reward = apply_action(state, IncidentAction(action_type="resolve_incident"))
    assert reward > 0
    assert state.resolved is True
    assert state.done is True


# ──────────────────────────────────────────────────────────────
# Time degradation
# ──────────────────────────────────────────────────────────────

def test_time_effects_increase_complaints():
    state = load_scenario("multi_service_incident")
    old_complaints = state.customers.complaint_count
    apply_time_effects(state)
    assert state.customers.complaint_count > old_complaints


def test_time_effects_with_status_page_slower_growth():
    state = load_scenario("multi_service_incident")
    state.operations.status_page_updated = True
    old = state.customers.complaint_count
    apply_time_effects(state)
    growth_with_update = state.customers.complaint_count - old

    state2 = load_scenario("multi_service_incident")
    old2 = state2.customers.complaint_count
    apply_time_effects(state2)
    growth_without = state2.customers.complaint_count - old2

    assert growth_with_update < growth_without


def test_time_effects_worsens_sla():
    state = load_scenario("multi_service_incident")
    old_sla = state.business.sla_breach_risk
    apply_time_effects(state)
    assert state.business.sla_breach_risk > old_sla


def test_time_effects_all_healthy_reduces_complaints():
    state = load_scenario("single_service_outage")
    # Make everything healthy
    for svc in state.services.values():
        svc.status = "healthy"
        svc.error_rate = 0.01
        svc.latency_ms = 100
    state.customers.complaint_count = 100
    apply_time_effects(state)
    assert state.customers.complaint_count < 100


# ──────────────────────────────────────────────────────────────
# Repeated action penalties
# ──────────────────────────────────────────────────────────────

def test_repeated_restart_penalized():
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="restart_service", target_service="payment_api"))
    apply_action(state, IncidentAction(action_type="restart_service", target_service="payment_api"))
    _, reward = apply_action(state, IncidentAction(action_type="restart_service", target_service="payment_api"))
    assert reward < 0  # 3rd attempt penalized


def test_inspect_same_service_penalized():
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="inspect_service", target_service="payment_api"))
    _, reward = apply_action(state, IncidentAction(action_type="inspect_service", target_service="payment_api"))
    assert reward < 0  # duplicate inspect


# ──────────────────────────────────────────────────────────────
# Resource metric tests
# ──────────────────────────────────────────────────────────────

def test_time_effects_degrade_cpu():
    """CPU usage should increase for degraded services over time."""
    state = load_scenario("dependency_degradation")
    old_cpu = state.services["ledger"].cpu_usage_pct
    apply_time_effects(state)
    assert state.services["ledger"].cpu_usage_pct > old_cpu


def test_restart_resets_resource_metrics():
    """Fixing root cause should reset CPU/memory/p99."""
    state = load_scenario("dependency_degradation")
    action = IncidentAction(action_type="restart_service", target_service="ledger")
    apply_action(state, action)
    led = state.services["ledger"]
    assert led.cpu_usage_pct < 50.0
    assert led.memory_usage_pct < 50.0


def test_rollback_resets_resource_metrics():
    """Rollback should reset resource metrics on the fixed service."""
    state = load_scenario("single_service_outage")
    action = IncidentAction(action_type="rollback_service", target_service="payment_api")
    apply_action(state, action)
    pay = state.services["payment_api"]
    assert pay.cpu_usage_pct < 40.0
    assert pay.memory_usage_pct < 50.0


def test_notification_service_degraded_in_dependency():
    """Notification service should start degraded in dependency_degradation."""
    state = load_scenario("dependency_degradation")
    notif = state.services["notification_service"]
    assert notif.status == "degraded"
    assert notif.cpu_usage_pct >= 50.0


# ──────────────────────────────────────────────────────────────
# Conflicting action detection
# ──────────────────────────────────────────────────────────────

def test_conflicting_restart_then_rollback():
    """Restart then rollback same service is a conflicting action."""
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="restart_service", target_service="payment_api"))
    _, reward = apply_action(state, IncidentAction(action_type="rollback_service", target_service="payment_api"))
    assert state.conflicting_actions_count == 1
    # Rollback still applies, but the conflict penalty reduces net reward


def test_conflicting_rollback_then_restart():
    """Rollback then restart same service is a conflicting action."""
    state = load_scenario("single_service_outage")
    apply_action(state, IncidentAction(action_type="rollback_service", target_service="payment_api"))
    _, reward = apply_action(state, IncidentAction(action_type="restart_service", target_service="payment_api"))
    assert state.conflicting_actions_count == 1


def test_no_conflict_different_targets():
    """Restart one service then rollback a different one is NOT conflicting."""
    state = load_scenario("dependency_degradation")
    apply_action(state, IncidentAction(action_type="restart_service", target_service="ledger"))
    apply_action(state, IncidentAction(action_type="rollback_service", target_service="payment_api"))
    assert state.conflicting_actions_count == 0


# ──────────────────────────────────────────────────────────────
# Wrong-target penalty
# ──────────────────────────────────────────────────────────────

def test_wrong_target_after_diagnosis():
    """Fixing a healthy non-root-cause service after diagnosis is penalized."""
    state = load_scenario("dependency_degradation")
    state.root_cause_identified = True
    # notification_service is degraded, not healthy — so let's make one healthy first
    state.services["notification_service"].status = "healthy"
    state.services["notification_service"].error_rate = 0.01
    state.services["notification_service"].latency_ms = 100
    apply_action(state, IncidentAction(action_type="restart_service", target_service="notification_service"))
    assert state.wrong_target_actions == 1


# ──────────────────────────────────────────────────────────────
# False resolve severity scaling
# ──────────────────────────────────────────────────────────────

def test_false_resolve_scales_with_issues():
    """False resolve penalty scales with number of unhealthy systems."""
    state = load_scenario("multi_service_incident")
    _, reward = apply_action(state, IncidentAction(action_type="resolve_incident"))
    # payment_api (down) + ledger (degraded) + notif (degraded) + db (degraded) = 4 issues
    assert reward < -0.20  # Base + scaling
    assert state.false_resolved is True
