"""Tests for deterministic graders — scores must be in [0.0, 1.0]."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import IncidentAction
from server.environment import IncidentEnvironment
from server.graders import grade_task


def _run_optimal_easy():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    env.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="inspect_logs", target_service="payment_api"))
    env.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="send_status_update"))
    env.step(IncidentAction(action_type="resolve_incident"))
    return env


def _run_do_nothing(task):
    env = IncidentEnvironment()
    env.reset(task)
    for _ in range(15):
        obs, _, done, _ = env.step(IncidentAction(action_type="wait"))
        if done:
            break
    return env


def test_optimal_easy_high_score():
    env = _run_optimal_easy()
    score = env.get_score()
    assert score >= 0.75, f"Optimal easy should score ≥0.75, got {score}"


def test_do_nothing_easy_low_score():
    env = _run_do_nothing("single_service_outage")
    score = env.get_score()
    assert score < 0.30, f"Do-nothing easy should score <0.30, got {score}"


def test_do_nothing_hard_low_score():
    env = _run_do_nothing("multi_service_incident")
    score = env.get_score()
    assert score < 0.30, f"Do-nothing hard should score <0.30, got {score}"


def test_scores_are_deterministic():
    s1 = _run_optimal_easy().get_score()
    s2 = _run_optimal_easy().get_score()
    assert s1 == s2, "Same actions must produce same score"


def test_all_scores_in_range():
    for task in ["single_service_outage", "dependency_degradation",
                 "multi_service_incident", "memory_leak_degradation",
                 "cascading_timeout_storm"]:
        env = _run_do_nothing(task)
        score = env.get_score()
        assert 0.0 <= score <= 1.0, f"{task} score {score} out of range"


def test_grade_task_unknown_raises():
    from server.scenarios import load_scenario
    state = load_scenario("single_service_outage")
    try:
        grade_task("nonexistent_task", state)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ──────────────────────────────────────────────────────────────
# Enhanced grader tests
# ──────────────────────────────────────────────────────────────

def test_time_penalty_function():
    from server.graders import _time_penalty
    from server.scenarios import load_scenario
    state = load_scenario("single_service_outage")

    # Fast resolution — no penalty
    state.incident.minutes_elapsed = 10
    assert _time_penalty(state, 15, 30) == 0.0

    # Slow resolution — max penalty
    state.incident.minutes_elapsed = 35
    assert _time_penalty(state, 15, 30) == -0.10


def test_business_impact_score_function():
    from server.graders import _business_impact_score
    from server.scenarios import load_scenario
    state = load_scenario("single_service_outage")

    # All healthy — good business impact
    state.business.sla_breach_risk = 0.10
    state.business.revenue_loss_per_min = 100
    state.business.compliance_risk = 0.05
    score = _business_impact_score(state)
    assert score >= 0.08  # should get most of 0.10


def test_optimal_easy_better_than_suboptimal():
    """Fast + correct actions should score higher than slow + correct."""
    env1 = _run_optimal_easy()
    score1 = env1.get_score()

    # Suboptimal: waste steps before fixing
    env2 = IncidentEnvironment()
    env2.reset("single_service_outage")
    env2.step(IncidentAction(action_type="wait"))
    env2.step(IncidentAction(action_type="wait"))
    env2.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
    env2.step(IncidentAction(action_type="inspect_logs", target_service="payment_api"))
    env2.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env2.step(IncidentAction(action_type="send_status_update"))
    env2.step(IncidentAction(action_type="resolve_incident"))
    score2 = env2.get_score()

    assert score1 > score2, f"Optimal {score1} should beat suboptimal {score2}"


# ──────────────────────────────────────────────────────────────
# Conflicting-action and wrong-target grader deductions
# ──────────────────────────────────────────────────────────────

def test_conflicting_actions_reduce_score():
    """Conflicting actions during the episode should reduce final score."""
    env_clean = IncidentEnvironment()
    env_clean.reset("single_service_outage")
    env_clean.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
    env_clean.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env_clean.step(IncidentAction(action_type="send_status_update"))
    env_clean.step(IncidentAction(action_type="resolve_incident"))
    score_clean = env_clean.get_score()

    # Now with a conflicting pair: restart then rollback same service
    env_conflict = IncidentEnvironment()
    env_conflict.reset("single_service_outage")
    env_conflict.step(IncidentAction(action_type="restart_service", target_service="payment_api"))
    env_conflict.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env_conflict.step(IncidentAction(action_type="send_status_update"))
    env_conflict.step(IncidentAction(action_type="resolve_incident"))
    score_conflict = env_conflict.get_score()

    assert score_clean > score_conflict, (
        f"Clean {score_clean} should beat conflicting {score_conflict}"
    )


def test_root_cause_diagnosis_bonus():
    """Medium/hard tasks get a grading bonus for identifying root cause."""
    # With diagnosis
    env1 = IncidentEnvironment()
    env1.reset("dependency_degradation")
    env1.step(IncidentAction(action_type="inspect_logs", target_service="ledger"))
    state1 = env1.state()
    assert state1["root_cause_identified"] is True

    # Without diagnosis (skip to fix)
    env2 = IncidentEnvironment()
    env2.reset("dependency_degradation")
    env2.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))

    # Run same remaining steps for both
    for env in (env1, env2):
        env.step(IncidentAction(action_type="restart_service", target_service="ledger"))
        env.step(IncidentAction(action_type="send_status_update"))
        env.step(IncidentAction(action_type="send_vip_update"))
        env.step(IncidentAction(action_type="resolve_incident"))

    # Diagnosis path should yield a higher or equal score
    assert env1.get_score() >= env2.get_score()
