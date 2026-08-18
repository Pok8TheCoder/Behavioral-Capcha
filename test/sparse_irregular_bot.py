"""Synthetic sparse irregular min-jerk family (Issue 007 vs Qmodel V0.8.1G)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

HOP_DT_MS = (4.0, 28.0)


@dataclass
class SparseConfig:
    spacing_px: tuple[float, float] = (13.0, 22.0)
    min_points: int = 10
    max_points: int = 26
    arc_px: tuple[float, float] = (10.0, 24.0)
    tremor_px: float = 1.5
    aim_sigma: float = 2.3
    dwell_ms: tuple[float, float] = (50.0, 110.0)
    pause_ms: tuple[float, float] = (60.0, 140.0)
    hop_dt_ms: tuple[float, float] = HOP_DT_MS
    s_curve: bool = True
    irregular: bool = True
    name: str = "sparse_irregular"


FAMILY_PRESETS = {
    "sparse_irregular": SparseConfig(),
    "sparse_fast": SparseConfig(
        spacing_px=(16.0, 26.0), min_points=8, max_points=22,
        arc_px=(6.0, 16.0), tremor_px=1.35, aim_sigma=2.2,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 120.0),
        s_curve=False, irregular=False, name="sparse_fast",
    ),
    "minjerk_fast": SparseConfig(
        spacing_px=(12.0, 20.0), min_points=10, max_points=28,
        arc_px=(8.0, 18.0), tremor_px=1.15, aim_sigma=2.0,
        s_curve=False, irregular=False, name="minjerk_fast",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def _min_jerk(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _append(events: list[dict], x: float, y: float, t: float, etype: str) -> float:
    events.append({
        "x": round(x), "y": round(y), "t": round(t, 2),
        "type": etype, "pid": 1, "pt": "mouse", "src": "canvas",
    })
    return t


def travel_along(events, start, end, t0, cfg: SparseConfig):
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, end[0], end[1], t0 + random.uniform(*HOP_DT_MS), "mousemove")
        return t0, end
    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    n = max(cfg.min_points, int(dist / random.uniform(*cfg.spacing_px)))
    n = min(n, cfg.max_points)
    if cfg.irregular:
        raw_steps = [random.uniform(0.4, 1.8) for _ in range(n)]
        total = sum(raw_steps)
        running = 0.0
        fractions = []
        for s in raw_steps:
            running += s
            fractions.append(running / total)
    else:
        fractions = [(i / n) for i in range(1, n + 1)]
    for raw in fractions:
        poly = _min_jerk(raw)
        wave = math.sin(math.pi * raw) * arc
        if cfg.s_curve:
            wave += math.sin(2 * math.pi * raw) * arc * 0.35
        tremor = random.gauss(0.0, cfg.tremor_px)
        t0 += random.uniform(*cfg.hop_dt_ms)
        t0 = _append(
            events,
            start[0] + dx * poly + nx * wave + tremor,
            start[1] + dy * poly + ny * wave + tremor * 0.5,
            t0,
            "mousemove",
        )
    t0 += random.uniform(*cfg.hop_dt_ms)
    t0 = _append(events, end[0], end[1], t0, "mousemove")
    return t0, end


def click_at(events, x, y, t0, cfg: SparseConfig) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 += random.uniform(30.0, 90.0)
    t0 = _append(events, ax, ay, t0, "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    t0 = _append(events, ax, ay, t0 + random.uniform(*cfg.dwell_ms), "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: SparseConfig | None = None) -> list[dict]:
    cfg = cfg or SparseConfig()
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: list[dict] = []
    t = random.uniform(40.0, 90.0)
    pos = (area["x"] + random.uniform(40, 160), area["y"] + random.uniform(40, 160))
    t = _append(events, pos[0], pos[1], t, "mousemove")
    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        t, pos = travel_along(events, pos, target, t, cfg)
        t = click_at(events, target[0], target[1], t, cfg)
        pos = target
        t += random.uniform(*cfg.pause_ms)
    return events


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.1417,
        "jitter_ratio": 0.2414,
        "velocity_mean": 1.4335,
        "velocity_max": 18.2867,
        "jerk_mean": 0.19221112,
        "acceleration_mean": 0.466679,
        "event_count": 60,
        "duration_ms": 2225.9,
        "avg_click_dwell_ms": 80.83,
        "idle_ratio": 0.4299,
    }
