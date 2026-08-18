"""Synthetic min-jerk dense overflow bot (Issue 006 — events above V0.7 #18 cap)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

PLAYWRIGHT_HOP_DT_MS = (6.0, 10.0)


@dataclass
class MinJerkConfig:
    spacing_px: tuple[float, float] = (7.0, 12.0)
    min_points: int = 24
    arc_px: tuple[float, float] = (10.0, 24.0)
    tremor_px: float = 0.9
    aim_sigma: float = 1.8
    dwell_ms: tuple[float, float] = (70.0, 130.0)
    pause_ms: tuple[float, float] = (120.0, 280.0)
    min_jerk: bool = True
    s_curve: bool = False
    name: str = "minjerk_direct"


FAMILY_PRESETS = {
    "minjerk_direct": MinJerkConfig(),
    "dense_smooth": MinJerkConfig(
        spacing_px=(4.0, 7.0), min_points=40, arc_px=(14.0, 30.0),
        tremor_px=0.7, aim_sigma=1.6, dwell_ms=(60.0, 120.0),
        pause_ms=(100.0, 220.0), name="dense_smooth",
    ),
    "curved_idle": MinJerkConfig(
        spacing_px=(6.0, 10.0), min_points=28, arc_px=(28.0, 55.0),
        tremor_px=1.2, aim_sigma=2.1, dwell_ms=(80.0, 160.0),
        pause_ms=(220.0, 550.0), s_curve=True, name="curved_idle",
    ),
    "slow_settle": MinJerkConfig(
        spacing_px=(8.0, 14.0), min_points=20, arc_px=(16.0, 36.0),
        tremor_px=1.4, aim_sigma=2.4, dwell_ms=(90.0, 180.0),
        pause_ms=(180.0, 400.0), min_jerk=False, s_curve=True, name="slow_settle",
    ),
}

FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def _min_jerk(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _append(events: list[dict], x: float, y: float, t: float, etype: str) -> float:
    events.append({
        "x": round(x),
        "y": round(y),
        "t": round(t, 2),
        "type": etype,
        "pid": 1,
        "pt": "mouse",
        "src": "canvas",
    })
    return t


def travel_along(
    events: list[dict],
    start: tuple[float, float],
    end: tuple[float, float],
    t0: float,
    cfg: MinJerkConfig,
) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, end[0], end[1], t0 + random.uniform(*PLAYWRIGHT_HOP_DT_MS), "mousemove")
        return t0, end

    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    n = max(cfg.min_points, int(dist / random.uniform(*cfg.spacing_px)))
    for i in range(1, n + 1):
        raw = i / n
        poly = _min_jerk(raw) if cfg.min_jerk else raw
        wave = math.sin(math.pi * raw) * arc
        if cfg.s_curve:
            wave += math.sin(2 * math.pi * raw) * arc * 0.35
        tremor = random.gauss(0.0, cfg.tremor_px)
        t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
        t0 = _append(
            events,
            start[0] + dx * poly + nx * wave + tremor,
            start[1] + dy * poly + ny * wave + tremor * 0.5,
            t0,
            "mousemove",
        )
    t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
    t0 = _append(events, end[0], end[1], t0, "mousemove")
    return t0, end


def click_at(events: list[dict], x: float, y: float, t0: float, cfg: MinJerkConfig) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 += random.uniform(40.0, 100.0)
    t0 = _append(events, ax, ay, t0, "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    t0 = _append(events, ax, ay, t0 + random.uniform(*cfg.dwell_ms), "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: MinJerkConfig | None = None) -> list[dict]:
    cfg = cfg or MinJerkConfig()
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: list[dict] = []
    t = random.uniform(40.0, 90.0)
    pos = (area["x"] + random.uniform(40, 140), area["y"] + random.uniform(40, 140))
    t = _append(events, pos[0], pos[1], t, "mousemove")

    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        t, pos = travel_along(events, pos, target, t, cfg)
        t = click_at(events, target[0], target[1], t, cfg)
        pos = target
        t += random.uniform(*cfg.pause_ms)
    return events


def winning_feature_sample() -> dict[str, float]:
    """Issue 006 minjerk_direct winning live run vs Qmodel V0.7."""
    return {
        "straightness_index": 0.1303,
        "jitter_ratio": 0.162,
        "velocity_mean": 0.8602,
        "velocity_max": 8.0,
        "jerk_mean": 0.168,
        "acceleration_mean": 0.246,
        "avg_click_dwell_ms": 100.0,
        "duration_ms": 4200.0,
        "event_count": 360,
    }
