import os
import math
import time
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.feature_extractor import extract_features
from backend.model import BehavioralClassifier

app = FastAPI(
    title="Behavioral CAPTCHA & Biometric Bot Detection Engine",
    description="Real-time biometric mouse/keyboard tracking & bot detection server",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = BehavioralClassifier()

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

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "Behavioral CAPTCHA Core",
        "model_trained": classifier.is_trained
    }

@app.post("/api/verify")
def verify_telemetry(payload: TelemetryPayload):
    """
    Receives user interaction telemetry events, computes biometric features,
    and returns human/bot risk evaluation.
    """
    if not payload.events:
        raise HTTPException(status_code=400, detail="Empty telemetry events payload")

    features = extract_features(payload.events)
    evaluation = classifier.evaluate(features)

    return {
        "timestamp": time.time(),
        "challenge_id": payload.challenge_id,
        "features": features,
        "evaluation": evaluation
    }

@app.post("/api/simulate-bot")
def simulate_bot(payload: BotSimPayload):
    """
    Generates synthetic bot telemetry data (Linear, Bezier, Humanoid, Jitter, Teleport)
    for model testing and client-side simulation.
    """
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
        ctrl_x1 = x0 + (x1 - x0) * 0.25 + 80.0
        ctrl_y1 = y0 + (y1 - y0) * 0.25 - 60.0
        ctrl_x2 = x0 + (x1 - x0) * 0.75 - 50.0
        ctrl_y2 = y0 + (y1 - y0) * 0.75 + 70.0
        dt_step = 14.0
        for i in range(n):
            u = i / (n - 1)
            cx = (1-u)**3 * x0 + 3*(1-u)**2 * u * ctrl_x1 + 3*(1-u) * u**2 * ctrl_x2 + u**3 * x1
            cy = (1-u)**3 * y0 + 3*(1-u)**2 * u * ctrl_y1 + 3*(1-u) * u**2 * ctrl_y2 + u**3 * y1
            events.append({"x": round(cx, 2), "y": round(cy, 2), "t": round(t0 + i * dt_step, 2), "type": "mousemove"})

    elif bot_type == "humanoid":
        # Minimum Jerk 5th degree polynomial + 8Hz tremor
        dt_step = 15.0
        f1 = 8.5
        for i in range(n):
            tau = i / (n - 1)
            poly = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            base_x = x0 + (x1 - x0) * poly
            base_y = y0 + (y1 - y0) * poly
            tremor_x = 0.8 * math.sin(2 * math.pi * f1 * (i * 0.015))
            tremor_y = 0.8 * math.cos(2 * math.pi * f1 * (i * 0.015))
            events.append({"x": round(base_x + tremor_x, 2), "y": round(base_y + tremor_y, 2), "t": round(t0 + i * dt_step, 2), "type": "mousemove"})

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

# Mount Frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def read_root():
        live_demo_path = os.path.join(frontend_path, "live_demo.html")
        if os.path.exists(live_demo_path):
            return FileResponse(live_demo_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{file_name}")
    def read_frontend_file(file_name: str):
        target = os.path.join(frontend_path, file_name)
        if os.path.exists(target) and os.path.isfile(target):
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
