from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Action Model — typed, parameterised action space
# ──────────────────────────────────────────────────────────────

class IncidentAction(BaseModel):
    action_type: Literal[
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
    target_service: Optional[str] = None
    message_template: Optional[str] = None
    replica_delta: Optional[int] = None
    rollback_version: Optional[str] = None
    escalation_team: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# State component models
# ──────────────────────────────────────────────────────────────

class ServiceState(BaseModel):
    status: Literal["healthy", "degraded", "down"] = "healthy"
    latency_ms: float = 100.0
    error_rate: float = 0.0
    version: str = "v1.0.0"
    replicas: int = 2
    previous_version: Optional[str] = None
    cpu_usage_pct: float = 30.0
    memory_usage_pct: float = 40.0
    response_time_p99: float = 200.0


class DatabaseState(BaseModel):
    status: Literal["healthy", "degraded", "down"] = "healthy"
    connections_pct: float = 0.50
    replication_lag_ms: float = 10.0


class CustomerState(BaseModel):
    complaint_count: int = 0
    vip_users_affected: int = 0
    sentiment_score: float = 0.0        # −1.0 … 1.0
    social_noise_score: float = 0.0     # 0.0 … 1.0


class BusinessState(BaseModel):
    revenue_loss_per_min: float = 0.0
    total_revenue_lost: float = 0.0
    sla_breach_risk: float = 0.0
    compliance_risk: float = 0.0


class OperationsState(BaseModel):
    status_page_updated: bool = False
    vip_outreach_sent: bool = False
    human_escalated: bool = False
    incident_bridge_opened: bool = False


class IncidentMeta(BaseModel):
    incident_id: str = "INC-0000"
    title: str = ""
    severity: Literal["sev1", "sev2", "sev3"] = "sev3"
    root_cause: str = ""
    root_cause_hint: str = ""
    minutes_elapsed: int = 0
    phase: str = "detect"   # detect→inspect→diagnose→mitigate→communicate→resolve


# ──────────────────────────────────────────────────────────────
# Full internal environment state
# ──────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_name: str = ""
    incident: IncidentMeta = Field(default_factory=IncidentMeta)
    services: dict[str, ServiceState] = Field(default_factory=dict)
    database: DatabaseState = Field(default_factory=DatabaseState)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    customers: CustomerState = Field(default_factory=CustomerState)
    business: BusinessState = Field(default_factory=BusinessState)
    operations: OperationsState = Field(default_factory=OperationsState)

    # Tracking
    actions_taken: list[str] = Field(default_factory=list)
    action_details: list[dict] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    step_count: int = 0
    max_steps: int = 8
    done: bool = False
    resolved: bool = False
    false_resolved: bool = False

    # Diagnosis state
    inspected_services: list[str] = Field(default_factory=list)
    root_cause_identified: bool = False
    diagnosis_actions: int = 0
    restart_attempts: dict[str, int] = Field(default_factory=dict)

    # Conflict & wrong-target tracking
    conflicting_actions_count: int = 0
    wrong_target_actions: int = 0

    # Communication tracking
    communication_overdue: bool = False
    delay_penalty_total: float = 0.0

    # Progress milestones
    services_recovered: list[str] = Field(default_factory=list)

    # Reward history
    rewards: list[float] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Observation model — what the agent sees
# ──────────────────────────────────────────────────────────────

class IncidentObservation(BaseModel):
    task_name: str
    step_count: int
    max_steps: int
    incident_summary: str
    service_health: dict
    customer_impact: dict
    business_impact: dict
    operations_status: dict
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    available_actions: list[str]
    recent_events: list[str]
    recent_action_details: list[dict] = Field(default_factory=list)
    last_action_error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# API request / response models
# ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "single_service_outage"


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


class StateResponse(BaseModel):
    state: dict
