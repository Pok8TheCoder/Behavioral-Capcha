"""Synthetic Playwright hop+jiggle bot (Issue 002 os_direct_v2 family).

Generates event streams that mimic live Playwright mouse motion:
large hops + 1px micro-samples → high jitter, low jerk/acceleration.
Used for offline evaluation and evasion dataset export — not JSON replay attacks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

PLAYWRIGHT_HOP_DT_MS = (28.0, 72.0)
MICRO_JIGGLE_DT_MS = (0.3, 1.2)


def ease_out_in(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


@dataclass
class HopConfig:
    step_px: tuple[float, float] = (18.0, 42.0)
    arc_px: tuple[float, float] = (22.0, 55.0)
    tremor_px: float = 1.2
    aim_sigma: float = 1.8
    dwell_ms: tuple[float, float] = (30.0, 70.0)
    pause_ms: tuple[float, float] = (180.0, 420.0)
    micro_jiggles: tuple[int, int] = (1, 3)
    tiny_step_prob: float = 0.18


def default_hop_cfg() -> HopConfig:
    return HopConfig(
        tremor_px=random.uniform(0.8, 1.6),
        aim_sigma=random.uniform(1.4, 2.4),
    )


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
    cfg: HopConfig,
) -> tuple[float, tuple[float, float]]:
    x, y = start
    tx, ty = end
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, tx, ty, t0 + random.uniform(*PLAYWRIGHT_HOP_DT_MS), "mousemove")
        return t0, end

    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    traveled = 0.0
    while traveled < dist:
        remain = dist - traveled
        if random.random() < cfg.tiny_step_prob:
            step = min(remain, random.uniform(2.0, 6.0))
        else:
            step = min(remain, random.uniform(*cfg.step_px))
        traveled += step
        raw = min(1.0, traveled / dist)
        eased = ease_out_in(raw)
        base_x = start[0] + dx * eased
        base_y = start[1] + dy * eased
        wave = math.sin(math.pi * raw) * arc
        tremor = random.gauss(0.0, cfg.tremor_px)
        px = base_x + nx * wave + tremor
        py = base_y + ny * wave + tremor * 0.65
        if random.random() < 0.05:
            t0 += random.uniform(12.0, 40.0)
        else:
            t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
        t0 = _append(events, px, py, t0, "mousemove")
        for _ in range(random.randint(*cfg.micro_jiggles)):
            t0 += random.uniform(*MICRO_JIGGLE_DT_MS)
            t0 = _append(
                events,
                px + random.uniform(-1.4, 1.4),
                py + random.uniform(-1.4, 1.4),
                t0,
                "mousemove",
            )

    t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
    t0 = _append(events, tx, ty, t0, "mousemove")
    return t0, end


def click_at(
    events: list[dict],
    x: float,
    y: float,
    t0: float,
    cfg: HopConfig,
) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 = _append(events, ax, ay, t0 + random.uniform(40.0, 120.0), "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    dwell = random.uniform(*cfg.dwell_ms)
    if random.random() < 0.55:
        mid = t0 + dwell * 0.5
        t0 = _append(events, ax + random.uniform(-1.2, 1.2), ay + random.uniform(-1.2, 1.2), mid, "mousemove")
    t0 = _append(events, ax, ay, t0 + dwell, "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: HopConfig | None = None) -> list[dict]:
    cfg = cfg or default_hop_cfg()
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: list[dict] = []
    t = random.uniform(50.0, 120.0)
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
    """Fingerprint from Issue 002 winning live run."""
    return {
        "straightness_index": 0.2683,
        "jitter_ratio": 0.7888,
        "velocity_mean": 0.6014,
        "velocity_max": 5.0,
        "jerk_mean": 0.0620,
        "acceleration_mean": 0.1428,
        "avg_click_dwell_ms": 50.0,
        "duration_ms": 4500.0,
        "event_count": 163,
    }
