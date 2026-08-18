"""Synthetic Playwright dense-hop family bots (Issue 004 — mid_dense and relatives).

Families: mid_dense, dense_alt, contrast_hop, straighter_dense.
Linear chord + alternating hops, all steps > 2px (no micro-jiggle).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List

PLAYWRIGHT_HOP_DT_MS = (6.0, 10.0)
MIN_STEP_PX = 3.1

FAMILY_PRESETS: Dict[str, Dict[str, Any]] = {
    "mid_dense": {
        "step_px": (12.0, 22.0),
        "small_px": (4.0, 9.0),
        "arc_px": (18.0, 40.0),
        "tremor_px": 1.15,
        "aim_sigma": 2.0,
        "dwell_ms": (40.0, 85.0),
        "pause_ms": (50.0, 140.0),
        "stutter_p": 0.08,
    },
    "dense_alt": {
        "step_px": (10.0, 18.0),
        "small_px": (3.5, 7.5),
        "arc_px": (12.0, 28.0),
        "tremor_px": 0.85,
        "aim_sigma": 1.9,
        "dwell_ms": (45.0, 90.0),
        "pause_ms": (50.0, 130.0),
        "stutter_p": 0.08,
    },
    "contrast_hop": {
        "step_px": (36.0, 62.0),
        "small_px": (3.4, 8.0),
        "arc_px": (16.0, 36.0),
        "tremor_px": 1.0,
        "aim_sigma": 2.1,
        "dwell_ms": (40.0, 80.0),
        "pause_ms": (40.0, 110.0),
        "stutter_p": 0.08,
    },
    "straighter_dense": {
        "step_px": (8.0, 14.0),
        "small_px": (3.3, 6.5),
        "arc_px": (5.0, 14.0),
        "tremor_px": 0.7,
        "aim_sigma": 1.6,
        "dwell_ms": (50.0, 100.0),
        "pause_ms": (60.0, 160.0),
        "stutter_p": 0.0,
    },
}

FAMILY_NAMES = list(FAMILY_PRESETS.keys())


@dataclass
class DenseFamilyConfig:
    step_px: tuple[float, float]
    small_px: tuple[float, float]
    arc_px: tuple[float, float]
    tremor_px: float
    aim_sigma: float
    dwell_ms: tuple[float, float]
    pause_ms: tuple[float, float]
    stutter_p: float = 0.08
    name: str = "mid_dense"


def cfg_from_preset(name: str) -> DenseFamilyConfig:
    p = FAMILY_PRESETS[name]
    return DenseFamilyConfig(
        step_px=tuple(p["step_px"]),
        small_px=tuple(p["small_px"]),
        arc_px=tuple(p["arc_px"]),
        tremor_px=p["tremor_px"],
        aim_sigma=p["aim_sigma"],
        dwell_ms=tuple(p["dwell_ms"]),
        pause_ms=tuple(p["pause_ms"]),
        stutter_p=p.get("stutter_p", 0.08),
        name=name,
    )


def random_family_cfg() -> DenseFamilyConfig:
    return cfg_from_preset(random.choice(FAMILY_NAMES))


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
    cfg: DenseFamilyConfig,
) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        t0 = _append(events, end[0], end[1], t0 + random.uniform(*PLAYWRIGHT_HOP_DT_MS), "mousemove")
        return t0, end

    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg.arc_px) * random.choice([-1.0, 1.0])
    traveled = 0.0
    big = True
    while traveled < dist:
        remain = dist - traveled
        step = min(remain, random.uniform(*(cfg.step_px if big else cfg.small_px)))
        big = not big
        step = max(step, MIN_STEP_PX)
        if step > remain:
            step = remain
        traveled += step
        raw = min(1.0, traveled / dist)
        base_x = start[0] + dx * raw
        base_y = start[1] + dy * raw
        wave = math.sin(math.pi * raw) * arc
        tremor = random.gauss(0.0, cfg.tremor_px)
        px = base_x + nx * wave + tremor
        py = base_y + ny * wave + tremor * 0.55
        if cfg.stutter_p > 0 and random.random() < cfg.stutter_p:
            t0 += random.uniform(12.0, 35.0)
        else:
            t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
        t0 = _append(events, px, py, t0, "mousemove")

    t0 += random.uniform(*PLAYWRIGHT_HOP_DT_MS)
    t0 = _append(events, end[0], end[1], t0, "mousemove")
    return t0, end


def click_at(events: list[dict], x: float, y: float, t0: float, cfg: DenseFamilyConfig) -> float:
    ax = x + random.gauss(0.0, cfg.aim_sigma)
    ay = y + random.gauss(0.0, cfg.aim_sigma)
    t0 += random.uniform(25.0, 70.0)
    t0 = _append(events, ax, ay, t0, "mousemove")
    t0 = _append(events, ax, ay, t0, "mousedown")
    dwell = random.uniform(*cfg.dwell_ms)
    t0 = _append(events, ax, ay, t0 + dwell, "mouseup")
    return t0


def generate_events(challenge: dict[str, Any], cfg: DenseFamilyConfig | None = None) -> list[dict]:
    cfg = cfg or random_family_cfg()
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
    """Issue 004 mid_dense winning live run."""
    return {
        "straightness_index": 0.216,
        "jitter_ratio": 0.143,
        "velocity_mean": 0.945,
        "velocity_max": 5.0,
        "jerk_mean": 0.172,
        "acceleration_mean": 0.280,
        "avg_click_dwell_ms": 62.0,
        "duration_ms": 4200.0,
        "event_count": 156,
    }


def family_feature_samples() -> List[dict[str, float]]:
    """Representative bands across the dense-hop family for rule tuning."""
    return [
        winning_feature_sample(),
        {
            "straightness_index": 0.12,
            "jitter_ratio": 0.11,
            "velocity_mean": 0.88,
            "velocity_max": 8.0,
            "jerk_mean": 0.14,
            "acceleration_mean": 0.24,
            "avg_click_dwell_ms": 60.0,
            "duration_ms": 3500.0,
            "event_count": 180,
        },
        {
            "straightness_index": 0.28,
            "jitter_ratio": 0.16,
            "velocity_mean": 1.05,
            "velocity_max": 12.0,
            "jerk_mean": 0.20,
            "acceleration_mean": 0.32,
            "avg_click_dwell_ms": 55.0,
            "duration_ms": 2800.0,
            "event_count": 95,
        },
    ]
