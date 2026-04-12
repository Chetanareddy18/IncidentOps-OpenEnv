"""Integration tests — full episode flows for all 5 tasks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import IncidentAction
from server.environment import IncidentEnvironment


# ──────────────────────────────────────────────────────────────
# Helper — run a scripted episode
# ──────────────────────────────────────────────────────────────

def _run_episode(task: str, actions: list[IncidentAction]) -> tuple:
    env = IncidentEnvironment()
    env.reset(task)
    total_reward = 0.0
    for action in actions:
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return env, total_reward, done, info


# ──────────────────────────────────────────────────────────────
# Easy: single_service_outage — optimal run
# ──────────────────────────────────────────────────────────────

def test_easy_optimal_episode():
    actions = [
        IncidentAction(action_type="inspect_service", target_service="payment_api"),
        IncidentAction(action_type="inspect_logs", target_service="payment_api"),
        IncidentAction(action_type="rollback_service", target_service="payment_api"),
        IncidentAction(action_type="send_status_update"),
        IncidentAction(action_type="resolve_incident"),
    ]
    env, total_reward, done, info = _run_episode("single_service_outage", actions)
    score = info.get("score", 0)
    assert done is True
    assert score >= 0.80, f"Optimal easy should score ≥0.80, got {score}"
    assert total_reward > 0


# ──────────────────────────────────────────────────────────────
# Medium: dependency_degradation — optimal run
# ──────────────────────────────────────────────────────────────

def test_medium_optimal_episode():
    actions = [
        IncidentAction(action_type="inspect_service", target_service="payment_api"),
        IncidentAction(action_type="inspect_logs", target_service="ledger"),
        IncidentAction(action_type="restart_service", target_service="ledger"),
        IncidentAction(action_type="restart_service", target_service="notification_service"),
        IncidentAction(action_type="send_status_update"),
        IncidentAction(action_type="send_vip_update"),
        IncidentAction(action_type="resolve_incident"),
    ]
    env, total_reward, done, info = _run_episode("dependency_degradation", actions)
    score = info.get("score", 0)
    assert done is True
    assert score >= 0.50, f"Optimal medium should score ≥0.50, got {score}"


# ──────────────────────────────────────────────────────────────
# Hard: multi_service_incident — optimal run
# ──────────────────────────────────────────────────────────────

def test_hard_optimal_episode():
    actions = [
        IncidentAction(action_type="inspect_service", target_service="payment_api"),
        IncidentAction(action_type="inspect_logs", target_service="db"),
        IncidentAction(action_type="failover_database"),
        IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE"),
        IncidentAction(action_type="restart_service", target_service="payment_api"),
        IncidentAction(action_type="restart_service", target_service="ledger"),
        IncidentAction(action_type="send_status_update"),
        IncidentAction(action_type="send_vip_update"),
        IncidentAction(action_type="resolve_incident"),
    ]
    env, total_reward, done, info = _run_episode("multi_service_incident", actions)
    score = info.get("score", 0)
    assert done is True
    assert score >= 0.45, f"Optimal hard should score ≥0.45, got {score}"


# ──────────────────────────────────────────────────────────────
# Medium: memory_leak_degradation — optimal run
# ──────────────────────────────────────────────────────────────

def test_memory_leak_optimal_episode():
    actions = [
        IncidentAction(action_type="inspect_service", target_service="payment_api"),
        IncidentAction(action_type="inspect_logs", target_service="payment_api"),
        IncidentAction(action_type="restart_service", target_service="payment_api"),
        IncidentAction(action_type="send_status_update"),
        IncidentAction(action_type="send_vip_update"),
        IncidentAction(action_type="resolve_incident"),
    ]
    env, total_reward, done, info = _run_episode("memory_leak_degradation", actions)
    score = info.get("score", 0)
    assert done is True
    assert score >= 0.70, f"Optimal memory_leak should score ≥0.70, got {score}"


# ──────────────────────────────────────────────────────────────
# Hard: cascading_timeout_storm — optimal run
# ──────────────────────────────────────────────────────────────

def test_cascading_timeout_optimal_episode():
    actions = [
        IncidentAction(action_type="inspect_service", target_service="notification_service"),
        IncidentAction(action_type="failover_database"),
        IncidentAction(action_type="restart_service", target_service="notification_service"),
        IncidentAction(action_type="escalate_to_human", escalation_team="on-call SRE"),
        IncidentAction(action_type="send_status_update"),
        IncidentAction(action_type="send_vip_update"),
        IncidentAction(action_type="resolve_incident"),
    ]
    env, total_reward, done, info = _run_episode("cascading_timeout_storm", actions)
    score = info.get("score", 0)
    assert done is True
    assert score >= 0.40, f"Optimal cascading should score ≥0.40, got {score}"


# ──────────────────────────────────────────────────────────────
# Cross-task properties
# ──────────────────────────────────────────────────────────────

def test_all_tasks_do_nothing_low_score():
    for task in ["single_service_outage", "dependency_degradation",
                 "multi_service_incident", "memory_leak_degradation",
                 "cascading_timeout_storm"]:
        env = IncidentEnvironment()
        env.reset(task)
        for _ in range(15):
            _, _, done, _ = env.step(IncidentAction(action_type="wait"))
            if done:
                break
        score = env.get_score()
        assert score < 0.30, f"Do-nothing {task} should score <0.30, got {score}"


def test_all_tasks_deterministic():
    for task in ["single_service_outage", "dependency_degradation",
                 "multi_service_incident", "memory_leak_degradation",
                 "cascading_timeout_storm"]:
        env1 = IncidentEnvironment()
        env1.reset(task)
        env1.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
        s1 = env1.get_score()

        env2 = IncidentEnvironment()
        env2.reset(task)
        env2.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
        s2 = env2.get_score()

        assert s1 == s2, f"{task} not deterministic: {s1} vs {s2}"


def test_wrong_fix_still_allows_recovery():
    """Even if agent makes mistakes early, partial recovery is possible."""
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    # Waste a step
    env.step(IncidentAction(action_type="inspect_service", target_service="ledger"))
    # Then fix correctly
    env.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="send_status_update"))
    _, _, done, info = env.step(IncidentAction(action_type="resolve_incident"))
    assert done is True
    assert info.get("score", 0) > 0.50


def test_episode_reward_sum_consistency():
    """Verify reward list matches step rewards."""
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    collected = []
    for _ in range(3):
        _, reward, done, _ = env.step(
            IncidentAction(action_type="inspect_service", target_service="payment_api")
        )
        collected.append(reward)
        if done:
            break
    state = env.state()
    assert len(state["rewards"]) == len(collected)


# ──────────────────────────────────────────────────────────────
# Partial progress milestones
# ──────────────────────────────────────────────────────────────

def test_partial_progress_milestone_tracked():
    """services_recovered should track newly healthy services."""
    env = IncidentEnvironment()
    env.reset("dependency_degradation")
    env.step(IncidentAction(action_type="restart_service", target_service="ledger"))
    state = env.state()
    # Ledger + payment_api cascade-recover to healthy
    assert "ledger" in state["services_recovered"]


def test_partial_progress_milestone_reward():
    """Recovering a service should give a milestone bonus in the reward."""
    env = IncidentEnvironment()
    env.reset("dependency_degradation")
    # First inspect (no recovery → no milestone)
    _, r1, _, _ = env.step(IncidentAction(action_type="inspect_service", target_service="ledger"))
    # Fix root cause (recovery → milestone)
    _, r2, _, _ = env.step(IncidentAction(action_type="restart_service", target_service="ledger"))
    # The fix step should have a higher reward than the inspect step
    assert r2 > r1


# ──────────────────────────────────────────────────────────────
# Medium task complexity
# ──────────────────────────────────────────────────────────────

def test_dependency_degradation_db_near_saturation():
    """dependency_degradation should start with DB connections >= 0.80."""
    env = IncidentEnvironment()
    env.reset("dependency_degradation")
    state = env.state()
    assert state["database"]["connections_pct"] >= 0.80


def test_dependency_degradation_notification_requires_triage():
    """Notification service should be degraded enough to require attention."""
    env = IncidentEnvironment()
    env.reset("dependency_degradation")
    state = env.state()
    notif = state["services"]["notification_service"]
    assert notif["error_rate"] >= 0.10
    assert notif["cpu_usage_pct"] >= 60.0
