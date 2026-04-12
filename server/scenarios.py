"""Deterministic scenario definitions for the five IncidentOps tasks."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    EnvironmentState, IncidentMeta, ServiceState,
    DatabaseState, CustomerState, BusinessState, OperationsState,
)


# ──────────────────────────────────────────────────────────────
# Task metadata (exposed by GET /tasks)
# ──────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "single_service_outage": {
        "description": (
            "Easy — Payment API outage after a recent deployment. "
            "Agent must inspect, rollback/restart, communicate, and resolve."
        ),
        "difficulty": "easy",
        "max_steps": 8,
        "success_criteria": (
            "Restore payment_api to healthy, update status page, "
            "resolve within 5 steps for full efficiency bonus."
        ),
        "expected_score_range": "0.75–0.95",
    },
    "dependency_degradation": {
        "description": (
            "Medium — Payment API degraded due to upstream ledger pressure "
            "compounded by resource overload. Notification service shows early "
            "degradation signs requiring triage. Database connection pool is "
            "approaching saturation from ledger retries. Agent must perform "
            "dependency analysis, fix the root cause (ledger), manage competing "
            "service priorities, prevent DB overload, and communicate dynamically "
            "as the incident evolves."
        ),
        "difficulty": "medium",
        "max_steps": 10,
        "success_criteria": (
            "Identify ledger as root cause (not payment_api), restart ledger, "
            "manage notification service degradation, prevent DB saturation, "
            "send status + VIP updates, resolve within 7 steps for full "
            "efficiency bonus."
        ),
        "expected_score_range": "0.50–0.85",
    },
    "multi_service_incident": {
        "description": (
            "Hard — Cascading multi-service failure from database saturation. "
            "Complex triage, sequencing, escalation, and communication required."
        ),
        "difficulty": "hard",
        "max_steps": 12,
        "success_criteria": (
            "Failover database first, restart dependent services in correct order, "
            "escalate to human team, communicate to status page and VIPs, "
            "resolve within 8 steps."
        ),
        "expected_score_range": "0.40–0.80",
    },
    "memory_leak_degradation": {
        "description": (
            "Medium — Payment API gradually degrading from a memory leak. "
            "Agent must identify the leak pattern, restart the right service, "
            "manage rising DB pressure, and communicate proactively."
        ),
        "difficulty": "medium",
        "max_steps": 10,
        "success_criteria": (
            "Identify memory leak in payment_api (not deployment regression), "
            "restart payment_api to clear leak, manage DB connection pressure, "
            "communicate proactively. Resolve within 6 steps."
        ),
        "expected_score_range": "0.60–0.90",
    },
    "cascading_timeout_storm": {
        "description": (
            "Hard — Notification service network failure cascading into "
            "payment retries overwhelming ledger and saturating the database. "
            "Agent must trace the reverse-cascade, fix root cause first, "
            "stabilise all services, escalate, and manage high social noise."
        ),
        "difficulty": "hard",
        "max_steps": 12,
        "success_criteria": (
            "Trace reverse cascade to notification_service, fix root cause, "
            "failover database if needed, escalate, manage social noise with "
            "status + VIP updates, resolve within 8 steps."
        ),
        "expected_score_range": "0.35–0.75",
    },
}


# ──────────────────────────────────────────────────────────────
# Shared dependency graph
# ──────────────────────────────────────────────────────────────

DEPENDENCY_GRAPH = {
    "payment_api": ["ledger", "db"],
    "ledger": ["db"],
    "notification_service": [],
}


# ──────────────────────────────────────────────────────────────
# Scenario builders
# ──────────────────────────────────────────────────────────────

def build_single_service_outage() -> EnvironmentState:
    return EnvironmentState(
        task_name="single_service_outage",
        incident=IncidentMeta(
            incident_id="INC-1001",
            title="Payment gateway outage after deploy",
            severity="sev2",
            root_cause="deployment_regression",
            root_cause_hint="Recent v2.3.1 deployment introduced a regression in payment processing",
            minutes_elapsed=4,
            phase="detect",
        ),
        services={
            "payment_api": ServiceState(
                status="down", latency_ms=2400.0, error_rate=0.71,
                version="v2.3.1", replicas=2, previous_version="v2.2.0",
                cpu_usage_pct=85.0, memory_usage_pct=60.0, response_time_p99=4800.0,
            ),
            "ledger": ServiceState(
                status="healthy", latency_ms=120.0, error_rate=0.02,
                version="v1.8.0", replicas=2,
                cpu_usage_pct=25.0, memory_usage_pct=35.0, response_time_p99=240.0,
            ),
            "notification_service": ServiceState(
                status="healthy", latency_ms=110.0, error_rate=0.02,
                version="v3.0.5", replicas=1,
                cpu_usage_pct=15.0, memory_usage_pct=30.0, response_time_p99=220.0,
            ),
        },
        database=DatabaseState(
            status="healthy", connections_pct=0.45, replication_lag_ms=10.0,
        ),
        dependencies=DEPENDENCY_GRAPH,
        customers=CustomerState(
            complaint_count=85, vip_users_affected=3,
            sentiment_score=-0.25, social_noise_score=0.15,
        ),
        business=BusinessState(
            revenue_loss_per_min=800.0, total_revenue_lost=3200.0,
            sla_breach_risk=0.35, compliance_risk=0.05,
        ),
        operations=OperationsState(),
        max_steps=8,
    )


def build_dependency_degradation() -> EnvironmentState:
    return EnvironmentState(
        task_name="dependency_degradation",
        incident=IncidentMeta(
            incident_id="INC-2001",
            title="Payment degradation from upstream ledger pressure with resource overload",
            severity="sev1",
            root_cause="ledger_overload",
            root_cause_hint="Ledger service is experiencing connection exhaustion and increased latency due to resource overload",
            minutes_elapsed=8,
            phase="detect",
        ),
        services={
            "payment_api": ServiceState(
                status="degraded", latency_ms=1800.0, error_rate=0.45,
                version="v2.3.1", replicas=2, previous_version="v2.2.0",
                cpu_usage_pct=78.0, memory_usage_pct=72.0, response_time_p99=3600.0,
            ),
            "ledger": ServiceState(
                status="degraded", latency_ms=900.0, error_rate=0.18,
                version="v1.8.0", replicas=2, previous_version="v1.7.0",
                cpu_usage_pct=92.0, memory_usage_pct=85.0, response_time_p99=1800.0,
            ),
            "notification_service": ServiceState(
                status="degraded", latency_ms=450.0, error_rate=0.12,
                version="v3.0.5", replicas=1,
                cpu_usage_pct=62.0, memory_usage_pct=55.0, response_time_p99=900.0,
            ),
        },
        database=DatabaseState(
            status="healthy", connections_pct=0.82, replication_lag_ms=55.0,
        ),
        dependencies=DEPENDENCY_GRAPH,
        customers=CustomerState(
            complaint_count=185, vip_users_affected=12,
            sentiment_score=-0.42, social_noise_score=0.38,
        ),
        business=BusinessState(
            revenue_loss_per_min=1200.0, total_revenue_lost=9600.0,
            sla_breach_risk=0.55, compliance_risk=0.12,
        ),
        operations=OperationsState(),
        max_steps=10,
    )


def build_multi_service_incident() -> EnvironmentState:
    return EnvironmentState(
        task_name="multi_service_incident",
        incident=IncidentMeta(
            incident_id="INC-3001",
            title="Multi-service cascading failure from database saturation",
            severity="sev1",
            root_cause="database_saturation",
            root_cause_hint="Database connections at capacity with replication lag spiking",
            minutes_elapsed=12,
            phase="detect",
        ),
        services={
            "payment_api": ServiceState(
                status="down", latency_ms=2400.0, error_rate=0.71,
                version="v2.3.1", replicas=2, previous_version="v2.2.0",
                cpu_usage_pct=95.0, memory_usage_pct=88.0, response_time_p99=5000.0,
            ),
            "ledger": ServiceState(
                status="degraded", latency_ms=900.0, error_rate=0.18,
                version="v1.8.0", replicas=2, previous_version="v1.7.0",
                cpu_usage_pct=80.0, memory_usage_pct=70.0, response_time_p99=1800.0,
            ),
            "notification_service": ServiceState(
                status="degraded", latency_ms=300.0, error_rate=0.05,
                version="v3.0.5", replicas=1,
                cpu_usage_pct=40.0, memory_usage_pct=35.0, response_time_p99=600.0,
            ),
        },
        database=DatabaseState(
            status="degraded", connections_pct=0.94, replication_lag_ms=200.0,
        ),
        dependencies=DEPENDENCY_GRAPH,
        customers=CustomerState(
            complaint_count=285, vip_users_affected=19,
            sentiment_score=-0.62, social_noise_score=0.58,
        ),
        business=BusinessState(
            revenue_loss_per_min=1700.0, total_revenue_lost=20400.0,
            sla_breach_risk=0.81, compliance_risk=0.20,
        ),
        operations=OperationsState(),
        max_steps=12,
    )


def build_memory_leak_degradation() -> EnvironmentState:
    """Medium — payment_api memory leak with rising DB pressure."""
    return EnvironmentState(
        task_name="memory_leak_degradation",
        incident=IncidentMeta(
            incident_id="INC-4001",
            title="Payment API memory leak causing progressive degradation",
            severity="sev2",
            root_cause="memory_leak",
            root_cause_hint="payment_api heap usage growing unbounded — classic memory leak after v2.4.0 deploy",
            minutes_elapsed=15,
            phase="detect",
        ),
        services={
            "payment_api": ServiceState(
                status="degraded", latency_ms=1200.0, error_rate=0.35,
                version="v2.4.0", replicas=2, previous_version="v2.3.1",
                cpu_usage_pct=70.0, memory_usage_pct=92.0, response_time_p99=2400.0,
            ),
            "ledger": ServiceState(
                status="healthy", latency_ms=140.0, error_rate=0.04,
                version="v1.8.0", replicas=2,
                cpu_usage_pct=30.0, memory_usage_pct=38.0, response_time_p99=280.0,
            ),
            "notification_service": ServiceState(
                status="healthy", latency_ms=100.0, error_rate=0.01,
                version="v3.0.5", replicas=1,
                cpu_usage_pct=12.0, memory_usage_pct=25.0, response_time_p99=200.0,
            ),
        },
        database=DatabaseState(
            status="healthy", connections_pct=0.72, replication_lag_ms=35.0,
        ),
        dependencies=DEPENDENCY_GRAPH,
        customers=CustomerState(
            complaint_count=120, vip_users_affected=6,
            sentiment_score=-0.30, social_noise_score=0.22,
        ),
        business=BusinessState(
            revenue_loss_per_min=600.0, total_revenue_lost=9000.0,
            sla_breach_risk=0.40, compliance_risk=0.08,
        ),
        operations=OperationsState(),
        max_steps=10,
    )


def build_cascading_timeout_storm() -> EnvironmentState:
    """Hard — notification_service network failure cascading everywhere."""
    return EnvironmentState(
        task_name="cascading_timeout_storm",
        incident=IncidentMeta(
            incident_id="INC-5001",
            title="Notification timeout storm cascading across all services",
            severity="sev1",
            root_cause="network_congestion",
            root_cause_hint="notification_service network interface saturated — timeout retries cascading upstream",
            minutes_elapsed=10,
            phase="detect",
        ),
        services={
            "payment_api": ServiceState(
                status="degraded", latency_ms=1600.0, error_rate=0.40,
                version="v2.3.1", replicas=2, previous_version="v2.2.0",
                cpu_usage_pct=82.0, memory_usage_pct=65.0, response_time_p99=3200.0,
            ),
            "ledger": ServiceState(
                status="degraded", latency_ms=700.0, error_rate=0.15,
                version="v1.8.0", replicas=2, previous_version="v1.7.0",
                cpu_usage_pct=75.0, memory_usage_pct=60.0, response_time_p99=1400.0,
            ),
            "notification_service": ServiceState(
                status="down", latency_ms=5000.0, error_rate=0.85,
                version="v3.0.5", replicas=1,
                cpu_usage_pct=98.0, memory_usage_pct=90.0, response_time_p99=10000.0,
            ),
        },
        database=DatabaseState(
            status="degraded", connections_pct=0.88, replication_lag_ms=120.0,
        ),
        dependencies=DEPENDENCY_GRAPH,
        customers=CustomerState(
            complaint_count=240, vip_users_affected=15,
            sentiment_score=-0.55, social_noise_score=0.65,
        ),
        business=BusinessState(
            revenue_loss_per_min=1500.0, total_revenue_lost=15000.0,
            sla_breach_risk=0.70, compliance_risk=0.18,
        ),
        operations=OperationsState(),
        max_steps=12,
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

SCENARIO_BUILDERS = {
    "single_service_outage": build_single_service_outage,
    "dependency_degradation": build_dependency_degradation,
    "multi_service_incident": build_multi_service_incident,
    "memory_leak_degradation": build_memory_leak_degradation,
    "cascading_timeout_storm": build_cascading_timeout_storm,
}


def load_scenario(task_name: str) -> EnvironmentState:
    builder = SCENARIO_BUILDERS.get(task_name)
    if builder is None:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available: {list(SCENARIO_BUILDERS.keys())}"
        )
    return builder()
