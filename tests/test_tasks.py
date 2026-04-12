"""Tests for task scenarios — initial state correctness."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.scenarios import load_scenario, TASK_CONFIGS, SCENARIO_BUILDERS


def test_all_tasks_loadable():
    for name in TASK_CONFIGS:
        state = load_scenario(name)
        assert state.task_name == name
        assert state.max_steps > 0
        assert len(state.services) >= 1


def test_single_service_outage_initial():
    s = load_scenario("single_service_outage")
    assert s.incident.severity == "sev2"
    assert s.incident.root_cause == "deployment_regression"
    assert s.services["payment_api"].status == "down"
    assert s.services["ledger"].status == "healthy"
    assert s.max_steps == 8


def test_dependency_degradation_initial():
    s = load_scenario("dependency_degradation")
    assert s.incident.severity == "sev1"
    assert s.incident.root_cause == "ledger_overload"
    assert s.services["payment_api"].status == "degraded"
    assert s.services["ledger"].status == "degraded"
    assert s.services["notification_service"].status == "degraded"
    assert s.max_steps == 10


def test_dependency_degradation_resource_metrics():
    """Ledger should show high CPU/memory as root cause signal."""
    s = load_scenario("dependency_degradation")
    led = s.services["ledger"]
    assert led.cpu_usage_pct >= 90.0
    assert led.memory_usage_pct >= 80.0
    # Notification service should show moderate resource stress (competing priority)
    notif = s.services["notification_service"]
    assert notif.cpu_usage_pct >= 50.0


def test_multi_service_incident_initial():
    s = load_scenario("multi_service_incident")
    assert s.incident.severity == "sev1"
    assert s.incident.root_cause == "database_saturation"
    assert s.services["payment_api"].status == "down"
    assert s.database.status == "degraded"
    assert s.database.connections_pct == 0.94
    assert s.max_steps == 12


def test_dependencies_present():
    for name in TASK_CONFIGS:
        s = load_scenario(name)
        assert "payment_api" in s.dependencies
        assert "db" in s.dependencies["payment_api"]


def test_unknown_task_raises():
    try:
        load_scenario("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_scenario_builders_match_configs():
    assert set(SCENARIO_BUILDERS.keys()) == set(TASK_CONFIGS.keys())


def test_all_scenarios_have_resource_metrics():
    """Every service in every scenario must have CPU, memory, p99 fields."""
    for name in TASK_CONFIGS:
        s = load_scenario(name)
        for svc_name, svc in s.services.items():
            assert hasattr(svc, "cpu_usage_pct"), f"{name}/{svc_name} missing cpu_usage_pct"
            assert hasattr(svc, "memory_usage_pct"), f"{name}/{svc_name} missing memory_usage_pct"
            assert hasattr(svc, "response_time_p99"), f"{name}/{svc_name} missing response_time_p99"
            assert 0.0 <= svc.cpu_usage_pct <= 100.0, f"{name}/{svc_name} cpu_usage_pct out of range"
            assert 0.0 <= svc.memory_usage_pct <= 100.0, f"{name}/{svc_name} memory_usage_pct out of range"


def test_memory_leak_initial_metrics():
    """Memory leak task should have high memory usage on payment_api."""
    s = load_scenario("memory_leak_degradation")
    pay = s.services["payment_api"]
    assert pay.memory_usage_pct >= 85.0


def test_cascading_timeout_initial():
    s = load_scenario("cascading_timeout_storm")
    assert s.incident.root_cause == "network_congestion"
    notif = s.services["notification_service"]
    assert notif.status == "down"


def test_task_configs_have_success_criteria():
    """Each task config should have success_criteria and expected_score_range."""
    for name, cfg in TASK_CONFIGS.items():
        assert "success_criteria" in cfg, f"{name} missing success_criteria"
        assert "expected_score_range" in cfg, f"{name} missing expected_score_range"
