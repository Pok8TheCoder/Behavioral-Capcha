"""Evasion Arena — live GUI for the closed-loop Xmodel attack / retrain cycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from queue import Empty

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_evasion.evasion_engine import ENGINE, WORKER_DEFAULT, WORKER_MAX, WORKER_SAFE
from backend.xmodel_registry import catalog_for_api
from bots.dashboard_bots import list_dashboard_bots

FRONTEND = os.path.join(ROOT, "frontend", "evasion_arena")
API = os.getenv("BEHAVIORAL_API", "http://127.0.0.1:8010")
ARENA_HOST = os.getenv("EVASION_ARENA_HOST", "127.0.0.1")
ARENA_PORT = int(os.getenv("EVASION_ARENA_PORT", "8011"))

app = FastAPI(title="Xmodel Evasion Arena")


class StartPayload(BaseModel):
    model_id: Optional[str] = None
    workers: Optional[int] = None


class WorkersPayload(BaseModel):
    workers: int


class WindowTargetPayload(BaseModel):
    windows_target: int


class TrainNowPayload(BaseModel):
    model_id: Optional[str] = None


class SelectModelPayload(BaseModel):
    model_id: str


class ProbePayload(BaseModel):
    model_id: Optional[str] = None
    count: int = 2


class ProbeDatasetPayload(BaseModel):
    dataset_id: str
    model_id: Optional[str] = None


class TrainFailsPayload(BaseModel):
    model_id: Optional[str] = None


def backend_health() -> Optional[dict]:
    try:
        with urllib.request.urlopen(API + "/api/health", timeout=2) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def ensure_backend() -> dict:
    health = backend_health()
    if health:
        return health
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8010"],
        cwd=ROOT,
    )
    for _ in range(40):
        time.sleep(0.5)
        health = backend_health()
        if health:
            return health
    raise RuntimeError("Behavioral API on :8010 did not start.")


@app.get("/api/arena/bootstrap")
def bootstrap():
    health = ensure_backend()
    catalog = catalog_for_api()
    return {
        "backend": health,
        "xmodels": catalog,
        "bots": list_dashboard_bots(),
        "state": ENGINE.snapshot(),
        "win_hc": 0.65,
        "mark_hc": 0.60,
        "window_target": ENGINE.state.get("windows_target", 5000),
        "stall_sec": ENGINE.state.get("stall_sec", 1800),
        "workers_default": WORKER_DEFAULT,
        "workers_safe": WORKER_SAFE,
        "workers_max": WORKER_MAX,
        "datasets": ENGINE.list_datasets(),
    }


@app.get("/api/arena/state")
def get_state():
    return ENGINE.snapshot()


@app.post("/api/arena/start")
def start_loop(payload: StartPayload):
    catalog = catalog_for_api()
    model_id = (payload.model_id or catalog.get("latest") or "xmodel_v0.3").strip()
    try:
        ensure_backend()
        ENGINE.start(model_id, workers=payload.workers)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "started", "state": ENGINE.snapshot()}


@app.post("/api/arena/pause")
def pause_loop():
    ENGINE.pause()
    return {"status": "paused"}


@app.post("/api/arena/resume")
def resume_loop():
    ENGINE.resume()
    return {"status": "resumed"}


@app.post("/api/arena/workers")
def set_workers(payload: WorkersPayload):
    n = ENGINE.set_workers(payload.workers)
    return {"status": "ok", "workers": n, "state": ENGINE.snapshot()}


@app.post("/api/arena/windows-target")
def set_windows_target(payload: WindowTargetPayload):
    n = ENGINE.set_windows_target(payload.windows_target)
    return {"status": "ok", "windows_target": n, "state": ENGINE.snapshot()}


@app.post("/api/arena/select-model")
def select_model(payload: SelectModelPayload):
    try:
        model_id = ENGINE.select_model(payload.model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "model_id": model_id, "state": ENGINE.snapshot()}


@app.post("/api/arena/train-now")
def train_now(payload: TrainNowPayload):
    try:
        out = ENGINE.train_now(parent_id=payload.model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "trained", **out}


@app.post("/api/arena/probe-recent")
def probe_recent(payload: Optional[ProbePayload] = None):
    payload = payload or ProbePayload()
    try:
        probe = ENGINE.probe_recent_episodes(payload.count or 2, model_id=payload.model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "probe": probe, "state": ENGINE.snapshot()}


@app.get("/api/arena/datasets")
def get_datasets():
    return {"datasets": ENGINE.list_datasets()}


@app.post("/api/arena/probe-dataset")
def probe_dataset(payload: ProbeDatasetPayload):
    try:
        probe = ENGINE.probe_dataset(payload.dataset_id, model_id=payload.model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "probe": probe, "state": ENGINE.snapshot()}


@app.post("/api/arena/train-fails")
def train_fails(payload: Optional[TrainFailsPayload] = None):
    payload = payload or TrainFailsPayload()
    try:
        out = ENGINE.train_on_failed_windows(parent_id=payload.model_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "trained", **out}


@app.post("/api/arena/stop")
def stop_loop():
    ENGINE.stop("stopped by operator")
    return {"status": "stopped"}


@app.get("/api/arena/stream")
def stream():
    q = ENGINE.subscribe()

    def gen():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'state': ENGINE.snapshot()})}\n\n"
            while True:
                try:
                    item = q.get(timeout=1.0)
                    yield f"data: {json.dumps(item, default=str)}\n\n"
                except Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'state': ENGINE.snapshot()})}\n\n"
        finally:
            ENGINE.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


@app.get("/static/{name}")
def static_file(name: str):
    allowed = {"arena.css", "arena.js"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(os.path.join(FRONTEND, name))


def main() -> None:
    import uvicorn

    url = f"http://{ARENA_HOST}:{ARENA_PORT}/"
    threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    print(f"Evasion Arena -> {url}")
    print("Scoring API   ->", API)
    restored = ENGINE.rehydrate()
    if restored:
        print(
            "Restored running cycle: "
            f"{ENGINE.state.get('windows_cycle')}/{ENGINE.state.get('windows_target')} windows"
        )
    uvicorn.run(app, host=ARENA_HOST, port=ARENA_PORT, log_level="info")


if __name__ == "__main__":
    main()
