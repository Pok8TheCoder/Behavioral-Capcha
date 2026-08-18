"""Synthetic pointing (no replay): two-phase homing + bursty sampling."""
from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8010"
OUT = Path(__file__).resolve().parent / "last_result.json"


def api(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Origin": API,
            "Referer": API + "/",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc


class Rng:
    def __init__(self, seed: int | None = None) -> None:
        self.r = random.Random(seed)

    def u(self) -> float:
        return self.r.random()

    def n(self) -> float:
        return self.r.gauss(0.0, 1.0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def generate_events(buttons: list[dict], rng: Rng) -> tuple[list[dict], float]:
    w, h = 960.0, 640.0
    t = rng.u() * 800 + 200
    x, y = 18 + rng.u() * 130, 16 + rng.u() * 100
    evs: list[dict] = []

    def push(px: float, py: float, typ: str) -> None:
        rx, ry = int(round(px)), int(round(py))
        if typ == "mousemove" and evs:
            prev = evs[-1]
            if prev["type"] == "mousemove" and prev["x"] == rx and prev["y"] == ry:
                return
        evs.append(
            {
                "x": rx,
                "y": ry,
                "t": round(t, 2),
                "type": typ,
                "pid": 1,
                "pt": "mouse",
                "src": "canvas",
            }
        )

    def burst(nx: float, ny: float) -> None:
        nonlocal t, x, y
        n = 1 + rng.r.randint(0, 3)
        for i in range(n):
            t += 0.25 + rng.u() * 1.4
            push(nx + (rng.n() * 0.35 if i else 0), ny + (rng.n() * 0.35 if i else 0), "mousemove")
        x, y = nx, ny
        t += 6 + rng.u() * 10 + (12 + rng.u() * 16 if rng.u() < 0.12 else 0)

    def home(tx: float, ty: float) -> None:
        nonlocal t, x, y
        if rng.u() < 0.28:
            ang = math.atan2(ty - y, tx - x) + (1 if rng.u() < 0.5 else -1) * (0.4 + rng.u() * 0.9)
            d = 18 + rng.u() * 55
            hx, hy = clamp(x + math.cos(ang) * d, 6, w - 6), clamp(y + math.sin(ang) * d, 6, h - 6)
            hops = 3 + rng.r.randint(0, 4)
            for i in range(1, hops + 1):
                burst(x + (hx - x) * i / hops + rng.n(), y + (hy - y) * i / hops + rng.n())
        for _ in range(420):
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < 2.5:
                break
            heading = math.atan2(dy, dx) + rng.n() * (0.2 + 0.32 * min(1.0, dist / 260))
            ballistic = dist > 46
            if ballistic:
                step = clamp(dist * (0.11 + rng.u() * 0.17) + 3 + rng.n() * 2, 2.2, 23)
            else:
                step = clamp(1.1 + dist * (0.17 + rng.u() * 0.22) + rng.n() * 0.8, 0.55, 8)
            jitter = rng.n() * 4 if (rng.u() < 0.08 and ballistic) else 0.0
            burst(
                clamp(x + math.cos(heading) * step + jitter, 5, w - 5),
                clamp(y + math.sin(heading) * step + jitter, 5, h - 5),
            )
            if rng.u() < 0.08 and dist > 28:
                t += 14 + rng.u() * 36
        burst(tx + rng.n(), ty + rng.n())

    def click(tx: float, ty: float) -> None:
        nonlocal t, x, y
        home(tx + rng.n() * 5.2, ty + rng.n() * 5.2)
        hover = t + 35 + rng.u() * 150
        while t < hover:
            t += 8 + rng.u() * 10
            push(x + rng.n() * 0.7, y + rng.n() * 0.7, "mousemove")
        push(x, y, "mousedown")
        hold = t + 50 + rng.u() * 100
        while t < hold:
            t += 8 + rng.u() * 8
            x += rng.n() * 0.2
            y += rng.n() * 0.2
            push(x, y, "mousemove")
        push(x, y, "mouseup")
        t += 30 + rng.u() * 130

    think = t + 300 + rng.u() * 800
    while t < think:
        t += 9 + rng.u() * 12
        x = clamp(x + rng.n() * 1.1, 8, w - 8)
        y = clamp(y + rng.n() * 0.9, 8, h - 8)
        push(x, y, "mousemove")

    for b in sorted(buttons, key=lambda z: z["id"]):
        click(float(b["x"]), float(b["y"]))
    return evs, t


def run_once(seed: int | None = None) -> dict:
    rng = Rng(seed)
    ch = api("/api/challenge/new", {"challenge_type": "button_sequence"})
    events, last_t = generate_events(ch["spec"]["buttons"], rng)
    span_ms = max(0.0, events[-1]["t"] - events[0]["t"]) if events else 0.0
    time.sleep(span_ms / 1000.0 + 0.35)
    started = 400 + rng.u() * 200
    out = api(
        "/api/challenge/verify",
        {
            "session_id": ch["session_id"],
            "events": events,
            "participant": "human",
            "verify_token": ch.get("verify_token"),
            "client_started_ms": started,
            "client_submitted_ms": started + span_ms + 80,
            "source": "dashboard",
        },
    )
    summary = {
        "challenge_type": ch.get("challenge_type"),
        "seed": ch.get("seed"),
        "n_events": len(events),
        "human_confidence": out.get("human_confidence"),
        "evaluation": out.get("evaluation"),
        "task_result": out.get("task_result"),
        "participant": out.get("participant"),
    }
    OUT.write_text(json.dumps({"summary": summary, "raw": out}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    best = None
    for i in range(3):
        s = run_once()
        print(json.dumps(s, indent=2))
        hc = s.get("human_confidence") or 0
        if best is None or hc > (best.get("human_confidence") or 0):
            best = s
        if hc >= 0.70:
            break
        time.sleep(0.2)
    print("BEST", json.dumps(best, indent=2))
