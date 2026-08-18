"""Synthetic fast_direct dart-hop bot (Issue 005 — high jitter, elevated velocity)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

PLAYWRIGHT_HOP_DT_MS = (6.0, 10.0)
MIN_STEP_PX = 3.2


@dataclass
class DartHopConfig:
    step_px: tuple[float, float] = (14.0, 26.0)
    arc_px: tuple[float, float] = (6.0, 16.0)
    dart_px: tuple[float, float] = (7.0, 14.0)
    dart_p: float = 0.28
    tremor_px: float = 1.0
    aim_sigma: float = 2.0
    dwell_ms: tuple[float, float] = (40.0, 80.0)
    pause_ms: tuple[float, float] = (40.0, 110.0)
    name: str = "fast_direct"


FAMILY_PRESETS = {
    "fast_direct": DartHopConfig(),
    "direct_wobble": DartHopConfig(
        step_px=(8.0, 15.0), arc_px=(8.0, 22.0), dart_px=(8.0, 16.0),
        dart_p=0.22, tremor_px=0.9, aim_sigma=1.8, name="direct_wobble",
    ),
    "human_turns": DartHopConfig(
        step_px=(7.0, 13.0), arc_px=(18.0, 38.0), dart_px=(10.0, 20.0),
        dart_p=0.32, tremor_px=1.1, aim_sigma=2.2, name="human_turns",
    ),
    "event_flood": DartHopConfig(
        step_px=(4.5, 8.0), arc_px=(10.0, 24.0), dart_px=(6.0, 12.0),
        dart_p=0.18, tremor_px=0.8, aim_sigma=1.7, name="event_flood",
    ),
}

FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def default_fast_direct_cfg() -> DartHopConfig:
    return DartHopConfig(tremor_px=random.uniform(0.85, 1.15), aim_sigma=random.uniform(1.7, 2.3))


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
    cfg: DartHopConfig,
) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, end[0], end[1], t0 + random.uniform(*PLAYWRIGHT_HOP_DT_MS), "mousemove")
        return t0, end

    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    traveled = 0.0
    while traveled < dist:
        remain = dist - traveled
        step = min(remain, random.uniform(*cfg.step_px))
        step = max(step, MIN_STEP_PX)
        if step > remain:
            step = remain
        traveled += step
        raw = min(1.0, traveled / dist)
        base_x = start[0] + dx * raw
        base_y = start[1] + dy * raw
        wave = math.sin(math.pi * raw) * arc
        if random.random() < cfg.dart_p and remain > 18:
            dart = random.uniform(*cfg.dart_px) * random.choice([-1.0, 1.0])
            t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
            t0 = _append(events, base_x + nx * (wave + dart), base_y + ny * (wave + dart), t0, "mousemove")
        tremor = random.gauss(0.0, cfg.tremor_px)
        t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
        t0 = _append(
            events,
            base_x + nx * wave + tremor,
            base_y + ny * wave + tremor * 0.5,
            t0,
            "mousemove",
        )

    t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
    t0 = _append(events, end[0], end[1], t0, "mousemove")
    return t0, end


def click_at(events: list[dict], x: float, y: float, t0: float, cfg: DartHopConfig) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 += random.uniform(30.0, 70.0)
    t0 = _append(events, ax, ay, t0, "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    t0 = _append(events, ax, ay, t0 + random.uniform(*cfg.dwell_ms), "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: DartHopConfig | None = None) -> list[dict]:
    cfg = cfg or default_fast_direct_cfg()
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
    """Issue 005 fast_direct winning live run vs Qmodel V0.6."""
    return {
        "straightness_index": 0.1643,
        "jitter_ratio": 0.5046,
        "velocity_mean": 1.2404,
        "velocity_max": 8.0,
        "jerk_mean": 0.153,
        "acceleration_mean": 0.326,
        "avg_click_dwell_ms": 60.0,
        "duration_ms": 2800.0,
        "event_count": 111,
    }
