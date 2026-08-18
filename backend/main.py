import os
import json
import math
import time
import random
import secrets
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.model import BehavioralClassifier
from backend.active_window import DEFAULT_ACTIVE_WINDOW_CONFIG, segment_episode_into_windows, _split_on_gaps, aggregate_window_evaluations
from bots.dashboard_bots import list_dashboard_bots, run_dashboard_bot
from bots.event_utils import normalize_bot_events
from challenge_server.generator import CHALLENGE_TYPES, CHALLENGE_CATALOG, generate_challenge, validate_completion
from backend.challenge_features import extract_challenge_features
from backend.challenge_security import (
    CAPTCHA_STRICT,
    ingest_live_pointer_events,
    ingest_sealed_pointer_events,
    is_open_telemetry_blocked,
    live_events_for_session,
    validate_verify_request,
)
from backend.human_capture import origin_bucket, sanitize_ip, write_human_capture
from backend.hz60 import downsample_events_to_hz
from backend.telemetry_crypto import new_telemetry_key, telemetry_key_to_b64, unseal_events

app = FastAPI(
    title="Behavioral CAPTCHA & Biometric Bot Detection Engine",
    description="Real-time biometric mouse/keyboard tracking & bot detection server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = BehavioralClassifier()
_challenge_sessions: Dict[str, Dict[str, Any]] = {}

class TelemetryPayload(BaseModel):
    events: List[Dict[str, Any]]
    challenge_id: Optional[str] = "demo-challenge"

class BotSimPayload(BaseModel):
    bot_type: str = Field(..., description="linear, bezier, humanoid, jitter, teleport")
    start_x: float = 100.0
    start_y: float = 100.0
    end_x: float = 600.0
    end_y: float = 400.0
    steps: int = 35

class ModelActivationPayload(BaseModel):
    model_name: str = Field(..., description="random_forest_humanlike, isolation_forest, or random_forest")


class NewChallengePayload(BaseModel):
    seed: Optional[int] = None
    challenge_type: Optional[str] = None


class ChallengeVerifyPayload(BaseModel):
    session_id: str
    events: Optional[List[Dict[str, Any]]] = None
    sealed_telemetry: Optional[Dict[str, Any]] = None
    participant: str = "human"
    verify_token: Optional[str] = None
    client_started_ms: Optional[float] = None
    client_submitted_ms: Optional[float] = None
    source: Optional[str] = None
    record_hz: Optional[float] = None


class LiveTelemetryPayload(BaseModel):
    session_id: str
    verify_token: str
    events: List[Dict[str, Any]]


class RunChallengeBotPayload(BaseModel):
    session_id: str
    bot: str = Field("wide_arc", description="wide_arc, minjerk, or simulated")
    record_hz: Optional[float] = None


class SaveHumanCapturePayload(BaseModel):
    session_id: str
    verify_token: Optional[str] = None
    events: Optional[List[Dict[str, Any]]] = None
    challenge_type: Optional[str] = None
    seed: Optional[int] = None
    spec: Optional[Dict[str, Any]] = None
    record_hz: Optional[float] = None


def _normalize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return normalize_bot_events(events)


def _maybe_record_hz(events: List[Dict[str, Any]], record_hz: Optional[float]) -> List[Dict[str, Any]]:
    if not record_hz or float(record_hz) <= 0:
        return events
    return downsample_events_to_hz(events, float(record_hz))


def _events_for_active_model(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    meta = classifier.metadata if isinstance(getattr(classifier, "metadata", None), dict) else {}
    hz = meta.get("target_hz")
    if hz:
        return downsample_events_to_hz(events, float(hz))
    return events


def _evaluate_events_dynamic(events: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    """Score using dynamic active windows; falls back to full stream if needed."""
    cfg = DEFAULT_ACTIVE_WINDOW_CONFIG
    sorted_events = sorted(_events_for_active_model(events), key=lambda e: float(e["t"]))
    episodes = _split_on_gaps(sorted_events, cfg.afk_gap_ms)

    evaluations: List[Dict[str, Any]] = []
    primary_features: Dict[str, Any] = {}

    for episode in episodes:
        for window in segment_episode_into_windows(episode, cfg):
            features = classifier.features_from_events(window)
            evaluation = classifier.evaluate_window(window)
            evaluations.append(evaluation)
            if not primary_features or evaluation.get("risk_score", 0) >= primary_features.get("_risk", 0):
                primary_features = {**features, "_risk": evaluation.get("risk_score", 0)}

    if not evaluations:
        features = classifier.features_from_events(sorted_events)
        return classifier.evaluate_window(sorted_events), features, 1

    primary_features.pop("_risk", None)
    aggregated = aggregate_window_evaluations(evaluations, strategy="max_risk")
    return aggregated, primary_features, len(evaluations)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return sanitize_ip(forwarded.split(",")[0])
    if request.client and request.client.host:
        return sanitize_ip(request.client.host)
    return "unknown"


def _remember_client(session: Dict[str, Any], ip: str) -> None:
    if not session or not ip or ip == "unknown":
        return
    session["client_ip"] = ip
    session["origin"] = origin_bucket(ip)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "Behavioral CAPTCHA Dashboard",
        "model_trained": classifier.is_trained,
        "active_model": classifier.model_type,
        "active_model_label": classifier.display_name,
        "available_models": classifier.available_models(),
        "challenge_types": list(CHALLENGE_TYPES),
        "active_challenge_sessions": len(_challenge_sessions),
    }

@app.get("/api/qmodels")
def list_qmodels():
    return classifier.qmodel_catalog()


@app.get("/api/models")
def list_models():
    return {
        "active_model": classifier.model_type,
        "active_model_label": classifier.display_name,
        "available_models": classifier.available_models(),
        "qmodels": classifier.qmodel_catalog(),
    }


@app.post("/api/models/activate")
def activate_model(payload: ModelActivationPayload):
    if not classifier.load_model(payload.model_name):
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unable to activate model '{payload.model_name}'.",
                "available_models": classifier.available_models(),
            },
        )

    return {
        "status": "activated",
        "active_model": classifier.model_type,
        "active_model_label": classifier.display_name,
        "available_models": classifier.available_models(),
        "qmodels": classifier.qmodel_catalog(),
    }


@app.post("/api/models/import")
async def import_model(file: UploadFile = File(...)):
    name = file.filename or "model.joblib"
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".joblib", ".pkl"):
        raise HTTPException(status_code=400, detail="Only .joblib or .pkl files are accepted.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > 80 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 80 MB).")
    try:
        imported = classifier.import_joblib_bytes(raw, name, activate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "imported",
        **imported,
        "active_model": classifier.model_type,
        "active_model_label": classifier.display_name,
        "available_models": classifier.available_models(),
        "qmodels": classifier.qmodel_catalog(),
    }


@app.get("/api/challenge/bots")
def list_challenge_bots():
    return {"bots": list_dashboard_bots()}

@app.post("/api/verify")
def verify_telemetry(payload: TelemetryPayload):
    if is_open_telemetry_blocked():
        raise HTTPException(
            status_code=403,
            detail="Open telemetry verify is disabled. Complete challenges through the dashboard.",
        )
    if not payload.events:
        raise HTTPException(status_code=400, detail="Empty telemetry events payload")

    evaluation, features, _ = _evaluate_events_dynamic(_normalize_events(payload.events))

    return {
        "timestamp": time.time(),
        "challenge_id": payload.challenge_id,
        "features": features,
        "evaluation": evaluation
    }

@app.post("/api/simulate-bot")
def simulate_bot(payload: BotSimPayload):
    if is_open_telemetry_blocked():
        raise HTTPException(
            status_code=403,
            detail="Bot simulation API is disabled in strict captcha mode.",
        )
    bot_type = payload.bot_type.lower()
    x0, y0 = payload.start_x, payload.start_y
    x1, y1 = payload.end_x, payload.end_y
    n = max(5, payload.steps)
    t0 = time.time() * 1000.0

    events = []

    if bot_type == "linear":
        dt_step = 12.0
        for i in range(n):
            alpha = i / (n - 1)
            cx = x0 + alpha * (x1 - x0)
            cy = y0 + alpha * (y1 - y0)
            events.append({"x": round(cx, 2), "y": round(cy, 2), "t": round(t0 + i * dt_step, 2), "type": "mousemove"})

    elif bot_type == "bezier":
        ctrl_x1 = x0 + (x1 - x0) * 0.25 + random.uniform(60.0, 120.0)
        ctrl_y1 = y0 + (y1 - y0) * 0.25 - random.uniform(40.0, 100.0)
        ctrl_x2 = x0 + (x1 - x0) * 0.75 - random.uniform(40.0, 90.0)
        ctrl_y2 = y0 + (y1 - y0) * 0.75 + random.uniform(50.0, 110.0)
        dt_step = 14.0
        for i in range(n):
            u = i / (n - 1)
            cx = (1-u)**3 * x0 + 3*(1-u)**2 * u * ctrl_x1 + 3*(1-u) * u**2 * ctrl_x2 + u**3 * x1
            cy = (1-u)**3 * y0 + 3*(1-u)**2 * u * ctrl_y1 + 3*(1-u) * u**2 * ctrl_y2 + u**3 * y1
            events.append({"x": round(cx, 2), "y": round(cy, 2), "t": round(t0 + i * dt_step, 2), "type": "mousemove"})

    elif bot_type == "humanoid":
        # Minimum Jerk 5th degree polynomial + Organic Joint Curvature Arc/S-Curve + 8Hz Tremor
        dt_step = 15.0
        f1 = 8.5
        dist = math.hypot(x1 - x0, y1 - y0)
        perp_x = -(y1 - y0) / (dist + 1e-6)
        perp_y = (x1 - x0) / (dist + 1e-6)

        curve_sign = random.choice([-1.0, 1.0])
        curve_amp = 0.16 * dist * curve_sign
        curve_style = random.choice(["arc", "s_curve", "elbow_swing"])

        for i in range(n):
            tau = i / (n - 1)
            # Minimum Jerk Polynomial
            poly = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            base_x = x0 + (x1 - x0) * poly
            base_y = y0 + (y1 - y0) * poly

            if curve_style == "arc":
                curve_offset = math.sin(math.pi * tau) * curve_amp
            elif curve_style == "s_curve":
                curve_offset = math.sin(2 * math.pi * tau) * curve_amp * 0.65
            else:
                curve_offset = 4 * tau * (1.0 - tau) * curve_amp

            curved_x = base_x + perp_x * curve_offset
            curved_y = base_y + perp_y * curve_offset

            tremor_x = 0.8 * math.sin(2 * math.pi * f1 * (i * 0.015))
            tremor_y = 0.8 * math.cos(2 * math.pi * f1 * (i * 0.015))

            events.append({
                "x": round(curved_x + tremor_x, 2),
                "y": round(curved_y + tremor_y, 2),
                "t": round(t0 + i * dt_step, 2),
                "type": "mousemove"
            })

    elif bot_type == "jitter":
        dt_step = 10.0
        for i in range(n):
            alpha = i / (n - 1)
            cx = x0 + alpha * (x1 - x0) + random.uniform(-18.0, 18.0)
            cy = y0 + alpha * (y1 - y0) + random.uniform(-18.0, 18.0)
            events.append({"x": round(cx, 2), "y": round(cy, 2), "t": round(t0 + i * dt_step, 2), "type": "mousemove"})

    elif bot_type == "teleport":
        events = [
            {"x": x0, "y": y0, "t": t0, "type": "mousemove"},
            {"x": x1, "y": y1, "t": t0 + 2.0, "type": "mousemove"}
        ]

    else:
        raise HTTPException(status_code=400, detail=f"Unknown bot_type: {bot_type}")

    end_t = events[-1]["t"]
    events.append({"x": x1, "y": y1, "t": end_t + 10.0, "type": "mousedown"})
    events.append({"x": x1, "y": y1, "t": end_t + 13.0, "type": "mouseup"})

    return {"bot_type": bot_type, "event_count": len(events), "events": events}


@app.get("/api/challenge/types")
def list_challenge_types():
    return {
        "challenge_types": list(CHALLENGE_TYPES),
        "catalog": CHALLENGE_CATALOG,
    }


@app.post("/api/challenge/new")
def create_challenge(payload: NewChallengePayload, request: Request):
    challenge = generate_challenge(seed=payload.seed, challenge_type=payload.challenge_type)
    verify_token = secrets.token_urlsafe(24)
    telemetry_key = new_telemetry_key()
    session_doc = {
        **challenge,
        "verify_token": verify_token,
        "telemetry_key": telemetry_key,
        "capture_mode": "local",
        "created_at": time.time(),
        "verified": False,
        "verify_attempts": 0,
        "live_events": [],
        "live_recv": [],
    }
    _remember_client(session_doc, _client_ip(request))
    _challenge_sessions[challenge["session_id"]] = session_doc
    response = {
        **challenge,
        "verify_token": verify_token,
        "telemetry_key": telemetry_key_to_b64(telemetry_key),
        "capture_mode": "local_sealed",
    }
    return response


@app.post("/api/challenge/live")
def challenge_live_http(payload: LiveTelemetryPayload, request: Request):
    session = _challenge_sessions.get(payload.session_id)
    if not session or payload.verify_token != session.get("verify_token"):
        raise HTTPException(status_code=403, detail="Invalid live telemetry session.")
    _remember_client(session, _client_ip(request))
    ok, reason, count = ingest_live_pointer_events(session, payload.events)
    if not ok:
        raise HTTPException(
            status_code=403,
            detail={"message": reason or "Live telemetry rejected.", "count": count},
        )
    return {"ok": True, "count": count}


@app.websocket("/ws/challenge/live")
async def challenge_live_telemetry(websocket: WebSocket, session_id: str = "", verify_token: str = ""):
    await websocket.accept()
    session = _challenge_sessions.get(session_id)
    if not session or not verify_token or verify_token != session.get("verify_token"):
        await websocket.close(code=4403)
        return
    if websocket.client and websocket.client.host:
        _remember_client(session, sanitize_ip(websocket.client.host))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"ok": False, "error": "invalid json", "count": 0})
                continue
            if payload.get("flush"):
                await websocket.send_json({"ok": True, "count": len(session.get("live_events") or [])})
                continue
            batch = payload.get("events")
            if batch is None and isinstance(payload, dict) and "x" in payload:
                batch = [payload]
            if not isinstance(batch, list):
                await websocket.send_json({"ok": False, "error": "expected events[]", "count": 0})
                continue
            ok, reason, count = ingest_live_pointer_events(session, batch)
            await websocket.send_json({"ok": ok, "error": reason, "count": count})
            if not ok and "too large" in reason:
                await websocket.close(code=4403)
                return
    except WebSocketDisconnect:
        return


@app.post("/api/challenge/verify")
def verify_challenge(payload: ChallengeVerifyPayload, request: Request):
    session = _challenge_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Challenge session not found.")
    _remember_client(session, _client_ip(request))

    supplied = payload.events or []
    if CAPTCHA_STRICT and supplied:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Plaintext telemetry is not accepted. Submit the encrypted local capture.",
                "security_flags": ["json_event_payload_rejected"],
            },
        )

    events_raw: List[Dict[str, Any]] = []
    if payload.sealed_telemetry:
        key = session.get("telemetry_key")
        if not key:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "This session has no local capture key. Start a new challenge.",
                    "security_flags": ["missing_telemetry_key"],
                },
            )
        try:
            unsealed = unseal_events(key, payload.sealed_telemetry)
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail={"message": str(exc), "security_flags": ["sealed_telemetry_invalid"]},
            ) from exc
        ok_ingest, reason, count = ingest_sealed_pointer_events(session, unsealed)
        if not ok_ingest:
            raise HTTPException(
                status_code=403,
                detail={"message": reason, "security_flags": ["sealed_telemetry_rejected"], "count": count},
            )
        events_raw = live_events_for_session(session)
    elif CAPTCHA_STRICT:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "No encrypted local capture. Finish the challenge on the canvas, then submit.",
                "security_flags": ["missing_sealed_telemetry"],
            },
        )
    else:
        live = live_events_for_session(session)
        events_raw = live if len(live) >= 4 else supplied

    if not events_raw:
        raise HTTPException(status_code=400, detail="Empty telemetry events payload.")

    ok, reason, security_flags = validate_verify_request(
        session,
        events_raw,
        payload.verify_token,
        payload.client_started_ms,
        payload.client_submitted_ms,
        payload.source,
        payload.participant,
    )
    session["verify_attempts"] = int(session.get("verify_attempts", 0)) + 1
    if not ok:
        raise HTTPException(
            status_code=403,
            detail={
                "message": reason,
                "security_flags": security_flags,
            },
        )

    events = _maybe_record_hz(_normalize_events(events_raw), payload.record_hz)
    evaluation, features, window_count = _evaluate_events_dynamic(events)
    task_result = validate_completion(session, events)
    bot_prob = evaluation.get("bot_probability")
    human_confidence = round(1.0 - float(bot_prob), 4) if bot_prob is not None else None

    if task_result.get("passed"):
        session["verified"] = True

    session["last_events"] = events
    session["last_participant"] = payload.participant
    session["last_task_passed"] = bool(task_result.get("passed"))
    session["last_features"] = features

    return {
        "timestamp": time.time(),
        "session_id": payload.session_id,
        "participant": payload.participant,
        "challenge_type": session["challenge_type"],
        "task_result": task_result,
        "features": features,
        "evaluation": evaluation,
        "active_windows_scored": window_count,
        "human_confidence": human_confidence,
        "security_flags": security_flags,
    }


@app.post("/api/challenge/run-bot")
def run_bot_on_challenge(payload: RunChallengeBotPayload):
    challenge = _challenge_sessions.get(payload.session_id)
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge session not found.")

    bot_name = payload.bot.lower()
    try:
        stamped = run_dashboard_bot(challenge, bot_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    events = _maybe_record_hz(_normalize_events(stamped), payload.record_hz)
    evaluation, features, window_count = _evaluate_events_dynamic(events)
    task_result = validate_completion(challenge, events)
    bot_prob = evaluation.get("bot_probability")
    human_confidence = round(1.0 - float(bot_prob), 4) if bot_prob is not None else None

    challenge["last_events"] = events
    challenge["last_participant"] = f"bot_{bot_name}"
    challenge["last_task_passed"] = bool(task_result.get("passed"))

    return {
        "timestamp": time.time(),
        "session_id": payload.session_id,
        "participant": f"bot_{bot_name}",
        "bot": bot_name,
        "challenge_type": challenge["challenge_type"],
        "event_count": len(events),
        "events": events,
        "task_result": task_result,
        "features": features,
        "evaluation": evaluation,
        "active_windows_scored": window_count,
        "human_confidence": human_confidence,
    }


@app.post("/api/challenge/save-human")
def save_human_capture(payload: SaveHumanCapturePayload, request: Request):
    session = _challenge_sessions.get(payload.session_id)
    client_events = payload.events or []
    request_ip = _client_ip(request)

    if session:
        _remember_client(session, request_ip)
        expected_token = session.get("verify_token")
        if expected_token and payload.verify_token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid or missing verify token.")
        participant = str(session.get("last_participant") or "human")
        if participant.lower().startswith("bot") or "evasion" in participant.lower():
            raise HTTPException(status_code=400, detail="Bot runs cannot be saved as human data.")
        events = session.get("last_events") or []
        challenge_type = str(session.get("challenge_type") or payload.challenge_type or "unknown")
        seed = session.get("seed", payload.seed)
        task_passed = session.get("last_task_passed")
        features = session.get("last_features") or {}
        spec = session.get("spec") or payload.spec or {}
        already = session.get("human_saved_path")
    else:
        events = client_events
        challenge_type = str(payload.challenge_type or "unknown")
        seed = payload.seed
        task_passed = None
        features = {}
        spec = payload.spec or {}
        already = None

    if len(events) < 4:
        raise HTTPException(status_code=400, detail="No captured telemetry to save for this session.")

    if already:
        return {
            "status": "already_saved",
            "filename": os.path.basename(str(already)),
            "relative_dir": os.path.join(
                "challenges",
                os.path.basename(os.path.dirname(str(already))),
            ).replace("\\", "/"),
            "event_count": len(events),
            "challenge_type": challenge_type,
            "client_ip": session.get("client_ip") if session else request_ip,
            "origin": session.get("origin") if session else origin_bucket(request_ip),
        }

    saved = write_human_capture(
        events,
        session_id=payload.session_id,
        challenge_type=challenge_type,
        seed=seed,
        task_passed=task_passed,
        features=features,
        spec=spec,
        challenge_features=extract_challenge_features(challenge_type, spec, events),
        client_ip=(session.get("client_ip") if session else None) or request_ip,
        target_hz=payload.record_hz,
    )
    if session is not None:
        session["human_saved_path"] = saved["csv_path"]
    return {
        "status": "saved",
        "label": "human",
        "filename": saved["filename"],
        "relative_dir": saved["relative_dir"],
        "event_count": saved["event_count"],
        "challenge_type": saved["challenge_type"],
        "client_ip": saved.get("client_ip"),
        "origin": saved.get("origin"),
    }


frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
dashboard_path = os.path.join(frontend_path, "dashboard")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def read_root():
        dashboard_index = os.path.join(dashboard_path, "index.html")
        if os.path.exists(dashboard_index):
            return FileResponse(dashboard_index)
        live_demo_path = os.path.join(frontend_path, "live_demo.html")
        if os.path.exists(live_demo_path):
            return FileResponse(live_demo_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/dashboard/{asset_name}")
    def read_dashboard_asset(asset_name: str):
        target = os.path.join(dashboard_path, asset_name)
        if os.path.exists(target) and os.path.isfile(target):
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="File not found")

    @app.get("/{file_name}")
    def read_frontend_file(file_name: str):
        if file_name.startswith("api"):
            raise HTTPException(status_code=404, detail="File not found")
        dash_target = os.path.join(dashboard_path, file_name)
        if os.path.exists(dash_target) and os.path.isfile(dash_target):
            return FileResponse(dash_target)
        target = os.path.join(frontend_path, file_name)
        if os.path.exists(target) and os.path.isfile(target):
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BEHAVIORAL_HOST", "0.0.0.0")
    port = int(os.getenv("BEHAVIORAL_PORT", "8010"))
    # Reload on Windows wedges the dashboard when two watchers share 8010 or
    # StatReload respawns while a request is in flight. Opt in with BEHAVIORAL_RELOAD=1.
    use_reload = os.getenv("BEHAVIORAL_RELOAD", "0") == "1"
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=use_reload,
        reload_dirs=[backend_dir] if use_reload else None,
    )
