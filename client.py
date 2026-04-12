"""IncidentOps OpenEnv — Python client."""
from __future__ import annotations

import requests
from typing import Optional


class IncidentOpsClient:
    """Thin wrapper around the IncidentOps HTTP API."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "single_service_outage") -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, **kwargs) -> dict:
        payload = {"action_type": action_type, **kwargs}
        r = requests.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def score(self) -> float:
        r = requests.get(f"{self.base_url}/score")
        r.raise_for_status()
        return r.json().get("score", 0.0)

    def tasks(self) -> dict:
        r = requests.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    c = IncidentOpsClient()
    print("Tasks:", c.tasks())
    print("Reset:", c.reset("single_service_outage"))
