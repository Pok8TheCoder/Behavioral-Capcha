"""Synthetic hi_dyn Playwright bot (Issue 003 — fast alternating hops, flat dynamics)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

PLAYWRIGHT_HOP_DT_MS = (6.0, 10.0)


def ease_out_in(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


@dataclass
class HiDynConfig:
    step_px: tuple[float, float] = (22.0, 48.0)
    small_px: tuple[float, float] = (4.0, 9.0)
    arc_px: tuple[float, float] = (14.0, 32.0)
    tremor_px: float = 0.9
    aim_sigma: float = 2.0
    dwell_ms: tuple[float, float] = (45.0, 85.0)
    pause_ms: tuple[float, float] = (50.0, 140.0)
    correct_p: float = 0.22


def default_hi_dyn_cfg() -> HiDynConfig:
    return HiDynConfig(tremor_px=random.uniform(0.7, 1.1), aim_sigma=random.uniform(1.7, 2.3))


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
    cfg: HiDynConfig,
) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, end[0], end[1], t0 + random.uniform(*PLAYWRIGHT_HOP_DT_MS), "mousemove")
        return t0, end

    nx, ny = (-dy / dist, dx / dist)
    ux, uy = dx / dist, dy / dist
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    traveled = 0.0
    big = True
    while traveled < dist:
        remain = dist - traveled
        if big:
            step = min(remain, random.uniform(*cfg.step_px))
        else:
            step = min(remain, random.uniform(*cfg.small_px))
        big = not big
        traveled += step
        raw = min(1.0, traveled / dist)
        eased = ease_out_in(raw)
        base_x = start[0] + dx * eased
        base_y = start[1] + dy * eased
        wave = math.sin(math.pi * raw) * arc
        tremor = random.gauss(0.0, cfg.tremor_px)
        px = base_x + nx * wave + tremor
        py = base_y + ny * wave + tremor * 0.6
        t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
        t0 = _append(events, px, py, t0, "mousemove")
        if cfg.correct_p > 0 and random.random() < cfg.correct_p and remain > 20:
            back = random.uniform(4.0, 10.0)
            t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
            t0 = _append(events, px - ux * back, py - uy * back, t0, "mousemove")

    t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
    t0 = _append(events, end[0], end[1], t0, "mousemove")
    return t0, end


def click_at(events: list[dict], x: float, y: float, t0: float, cfg: HiDynConfig) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 += random.uniform(30.0, 80.0)
    t0 = _append(events, ax, ay, t0, "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    dwell = random.uniform(*cfg.dwell_ms)
    t0 = _append(events, ax, ay, t0 + dwell, "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: HiDynConfig | None = None) -> list[dict]:
    cfg = cfg or default_hi_dyn_cfg()
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: list[dict] = []
    t = random.uniform(40.0, 100.0)
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
    return {
        "straightness_index": 0.3107,
        "jitter_ratio": 0.4029,
        "velocity_mean": 1.0798,
        "velocity_max": 6.0,
        "jerk_mean": 0.0558,
        "acceleration_mean": 0.1699,
        "avg_click_dwell_ms": 65.0,
        "duration_ms": 3800.0,
        "event_count": 141,
    }
