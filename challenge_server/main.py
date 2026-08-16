"""
Behavioral challenge test server.

Hosts randomized human verification tasks and proxies ML scoring to the
detection backend (default http://127.0.0.1:8010).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge, validate_completion

DETECTION_API = os.getenv("BEHAVIORAL_DETECTION_API", "http://127.0.0.1:8010")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CHALLENGE_FRONTEND = os.path.join(ROOT_DIR, "frontend", "challenge")

app = FastAPI(
    title="Behavioral Challenge Test Server",
    description="Randomized mouse challenge tasks with human/bot detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, Dict[str, Any]] = {}


class NewSessionPayload(BaseModel):
    seed: Optional[int] = None
    challenge_type: Optional[str] = Field(None, description="shape_trace, button_sequence, maze_navigate, pin_entry, path_gates")


class VerifyPayload(BaseModel):
    session_id: str
    events: List[Dict[str, Any]]
    participant: str = Field("human", description="human or bot label for logging")


def _normalize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for event in events:
        raw_type = str(event.get("type", event.get("event_type", "mousemove"))).lower()
        if raw_type in ("mouse_move", "0"):
            evt_type = "mousemove"
        elif raw_type in ("mouse_click_down", "mousedown", "1"):
            evt_type = "mousedown"
        elif raw_type in ("mouse_click_up", "mouseup", "2"):
            evt_type = "mouseup"
        else:
            evt_type = raw_type
        normalized.append({
            "x": float(event["x"]),
            "y": float(event["y"]),
            "t": float(event.get("t", 0)),
            "type": evt_type,
        })
    return normalized


def _proxy_detection(events: List[Dict[str, Any]], challenge_id: str) -> Dict[str, Any]:
    payload = json.dumps({"events": events, "challenge_id": challenge_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{DETECTION_API}/api/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {
            "evaluation": {
                "decision": "UNKNOWN",
                "is_bot": None,
                "risk_score": None,
                "model_type": "unavailable",
                "flags": [f"Detection API unreachable: {exc}"],
            },
            "features": {},
        }


@app.get("/api/health")
def health_check():
    detection_status = "unknown"
    try:
        with urllib.request.urlopen(f"{DETECTION_API}/api/health", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            detection_status = data.get("status", "unknown")
    except urllib.error.URLError:
        detection_status = "offline"

    return {
        "status": "healthy",
        "engine": "Behavioral Challenge Test Server",
        "detection_api": DETECTION_API,
        "detection_status": detection_status,
        "challenge_types": list(CHALLENGE_TYPES),
        "active_sessions": len(_sessions),
    }


@app.get("/api/challenge/types")
def list_challenge_types():
    return {"challenge_types": list(CHALLENGE_TYPES)}


@app.post("/api/challenge/new")
def create_challenge(payload: NewSessionPayload):
    challenge = generate_challenge(seed=payload.seed, challenge_type=payload.challenge_type)
    _sessions[challenge["session_id"]] = challenge
    return challenge


@app.post("/api/challenge/verify")
def verify_challenge(payload: VerifyPayload):
    challenge = _sessions.get(payload.session_id)
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge session not found or expired.")

    if not payload.events:
        raise HTTPException(status_code=400, detail="Empty telemetry events payload.")

    events = _normalize_events(payload.events)
    task_result = validate_completion(challenge, events)
    detection = _proxy_detection(events, payload.session_id)

    evaluation = detection.get("evaluation", {})
    return {
        "timestamp": time.time(),
        "session_id": payload.session_id,
        "participant": payload.participant,
        "challenge_type": challenge["challenge_type"],
        "task_result": task_result,
        "features": detection.get("features", {}),
        "evaluation": evaluation,
        "human_confidence": None if evaluation.get("bot_probability") is None else round(1.0 - float(evaluation["bot_probability"]), 4),
    }


if os.path.exists(CHALLENGE_FRONTEND):
    @app.get("/")
    def challenge_home():
        return FileResponse(os.path.join(CHALLENGE_FRONTEND, "index.html"))

    @app.get("/{asset_name}")
    def challenge_asset(asset_name: str):
        if asset_name.startswith("api"):
            raise HTTPException(status_code=404, detail="Not found")
        target = os.path.join(CHALLENGE_FRONTEND, asset_name)
        if os.path.exists(target) and os.path.isfile(target):
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CHALLENGE_HOST", "127.0.0.1")
    port = int(os.getenv("CHALLENGE_PORT", "8020"))
    uvicorn.run("challenge_server.main:app", host=host, port=port, reload=True)
