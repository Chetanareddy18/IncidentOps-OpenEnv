"""IncidentOps OpenEnv — FastAPI server.

Endpoints:
  GET  /           health / info
  GET  /health     liveness probe
  GET  /tasks      list available tasks
  POST /reset      reset environment for a task
  POST /step       take an action
  GET  /state      full internal state
  GET  /score      current grader score
  GET  /web        incident war-room dashboard
  GET  /events     SSE real-time state stream
  POST /webhooks/notify  notification webhook
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from models import IncidentAction, ResetRequest
from server.environment import IncidentEnvironment
from server.scenarios import TASK_CONFIGS

app = FastAPI(
    title="IncidentOps OpenEnv",
    description=(
        "A real-world incident command training environment for AI agents "
        "in fintech production operations."
    ),
    version="1.1.0",
)

env = IncidentEnvironment()
webhook_subscribers: List[str] = []


# ──────────────────────────────────────────────────────────────
# Info / health
# ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "IncidentOps OpenEnv",
        "version": "1.1.0",
        "status": "ok",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/score", "/web", "/events", "/webhooks/notify"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────
# Task listing
# ──────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_CONFIGS}


# ──────────────────────────────────────────────────────────────
# Environment lifecycle
# ──────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    task_id = (req.task_id if req and req.task_id else "single_service_outage")
    try:
        obs = env.reset(task_id)
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: IncidentAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state():
    return {"state": env.state()}


@app.get("/score")
def get_score():
    return {"score": env.get_score()}


# ──────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
def dashboard():
    # Auto-initialize with default scenario if no task is active
    if not env.state():
        env.reset("single_service_outage")

    tpl = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    try:
        with open(tpl, encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard template not found</h1>", status_code=500)

    state_json = json.dumps(env.state(), default=str)
    tasks_json = json.dumps(TASK_CONFIGS, default=str)
    html = html.replace("__STATE_JSON__", state_json)
    html = html.replace("__TASKS_JSON__", tasks_json)
    return HTMLResponse(html)


# ──────────────────────────────────────────────────────────────
# SSE real-time stream
# ──────────────────────────────────────────────────────────────

@app.get("/events")
async def sse_events(request: Request):
    """Server-Sent Events stream for real-time dashboard updates."""
    async def event_generator():
        prev_state = None
        while True:
            if await request.is_disconnected():
                break
            current = env.state()
            score = env.get_score()
            state_json = json.dumps({"state": current, "score": score}, default=str)
            if state_json != prev_state:
                yield f"event: state\ndata: {state_json}\n\n"
                prev_state = state_json
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────
# Notification webhook
# ──────────────────────────────────────────────────────────────

class WebhookSubscription(BaseModel):
    url: str
    events: List[str] = ["all"]


@app.post("/webhooks/notify")
def register_webhook(sub: WebhookSubscription):
    """Register a webhook URL for incident notifications.

    Compatible with SendGrid/Twilio-style integrations.
    Events: service_down, sla_breach, resolved, escalation, all.
    """
    if sub.url not in webhook_subscribers:
        webhook_subscribers.append(sub.url)
    return {"status": "subscribed", "url": sub.url, "events": sub.events}


@app.get("/webhooks/notify")
def list_webhooks():
    return {"subscribers": webhook_subscribers}


# ──────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
