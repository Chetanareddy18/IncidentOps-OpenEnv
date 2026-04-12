"""Tests for environment lifecycle: reset, step, state, scoring."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import IncidentAction
from server.environment import IncidentEnvironment


def test_reset_returns_observation():
    env = IncidentEnvironment()
    obs = env.reset("single_service_outage")
    assert obs.task_name == "single_service_outage"
    assert obs.step_count == 0
    assert obs.max_steps == 8
    assert len(obs.available_actions) > 0


def test_reset_all_tasks():
    env = IncidentEnvironment()
    for task in ["single_service_outage", "dependency_degradation",
                 "multi_service_incident", "memory_leak_degradation",
                 "cascading_timeout_storm"]:
        obs = env.reset(task)
        assert obs.task_name == task


def test_step_returns_tuple():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    action = IncidentAction(action_type="inspect_service", target_service="payment_api")
    obs, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert obs.step_count == 1


def test_invalid_action_returns_error():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    action = IncidentAction(action_type="inspect_service")  # missing target
    obs, reward, done, info = env.step(action)
    assert reward < 0
    assert "error" in info


def test_episode_terminates_at_max_steps():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    for _ in range(20):
        action = IncidentAction(action_type="wait")
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done is True
    assert "score" in info


def test_resolve_when_healthy():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    # Rollback to fix payment
    env.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="send_status_update"))
    obs, reward, done, info = env.step(IncidentAction(action_type="resolve_incident"))
    assert done is True
    assert info.get("score", 0) > 0


def test_false_resolve_penalty():
    env = IncidentEnvironment()
    env.reset("multi_service_incident")
    obs, reward, done, info = env.step(
        IncidentAction(action_type="resolve_incident"),
    )
    assert reward < 0  # penalty for false resolution
    assert done is False or env.state().get("false_resolved") is True


def test_state_returns_dict():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    s = env.state()
    assert isinstance(s, dict)
    assert "incident" in s
    assert "services" in s
    assert "customers" in s
    assert "business" in s


def test_score_range():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    # Take some actions
    env.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="rollback_service", target_service="payment_api"))
    env.step(IncidentAction(action_type="send_status_update"))
    env.step(IncidentAction(action_type="resolve_incident"))
    score = env.get_score()
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────
# Enhanced feature tests
# ──────────────────────────────────────────────────────────────

def test_observation_includes_dependency_graph():
    env = IncidentEnvironment()
    obs = env.reset("single_service_outage")
    assert "payment_api" in obs.dependency_graph
    assert "db" in obs.dependency_graph["payment_api"]


def test_observation_includes_resource_metrics():
    env = IncidentEnvironment()
    obs = env.reset("single_service_outage")
    pay = obs.service_health["payment_api"]
    assert "cpu_usage_pct" in pay
    assert "memory_usage_pct" in pay
    assert "response_time_p99" in pay


def test_available_actions_dynamic():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    obs1, _, _, _ = env.step(IncidentAction(action_type="send_status_update"))
    # After sending status update, it should be removed from available actions
    assert "send_status_update" not in obs1.available_actions


def test_observation_includes_recent_action_details():
    env = IncidentEnvironment()
    env.reset("single_service_outage")
    obs, _, _, _ = env.step(IncidentAction(action_type="inspect_service", target_service="payment_api"))
    assert len(obs.recent_action_details) >= 1


def test_progressive_delay_penalty():
    """Steps beyond 70% of max should have higher penalty."""
    env = IncidentEnvironment()
    env.reset("single_service_outage")  # max_steps=8
    rewards = []
    for i in range(8):
        _, reward, done, _ = env.step(IncidentAction(action_type="wait"))
        rewards.append(reward)
        if done:
            break
    # Later steps should have worse rewards (more penalty)
    assert rewards[-1] <= rewards[0]


def test_communication_urgency_tracked():
    """State should track communication_overdue flag."""
    env = IncidentEnvironment()
    env.reset("multi_service_incident")
    state = env.state()
    assert "communication_overdue" in state
