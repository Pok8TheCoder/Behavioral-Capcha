"""Fitts-law + minimum-jerk motor synthesis for evasion bots."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def minimum_jerk(tau: float) -> float:
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


@dataclass
class FittsMotorConfig:
    fitts_a: float = 0.11
    fitts_b: float = 0.17
    target_width: float = 42.0
    tremor_scale: float = 0.68
    curvature_scale: float = 0.14
    pause_probability: float = 0.20
    overshoot_probability: float = 0.28
    hesitation_ms: Tuple[float, float] = (28.0, 95.0)
    sample_hz: Tuple[float, float] = (72.0, 108.0)
    dwell_ms: Tuple[float, float] = (95.0, 210.0)
    wander_strength: float = 4.2
    pause_ms: Tuple[float, float] = (220.0, 480.0)
    pre_click_delay_ms: Tuple[float, float] = (45.0, 175.0)


def default_motor() -> FittsMotorConfig:
    return FittsMotorConfig()


def motor_from_dict(d: Dict[str, Any]) -> FittsMotorConfig:
    return FittsMotorConfig(
        fitts_a=d.get("fitts_a", 0.11),
        fitts_b=d.get("fitts_b", 0.17),
        target_width=d.get("target_width", 42.0),
        tremor_scale=d.get("tremor_scale", 0.68),
        curvature_scale=d.get("curvature_scale", 0.14),
        pause_probability=d.get("pause_probability", 0.20),
        overshoot_probability=d.get("overshoot_probability", 0.28),
        hesitation_ms=tuple(d.get("hesitation_ms", (28.0, 95.0))),
        sample_hz=tuple(d.get("sample_hz", (72.0, 108.0))),
        dwell_ms=tuple(d.get("dwell_ms", (95.0, 210.0))),
        wander_strength=d.get("wander_strength", 4.2),
        pause_ms=tuple(d.get("pause_ms", (220.0, 480.0))),
        pre_click_delay_ms=tuple(d.get("pre_click_delay_ms", (45.0, 175.0))),
    )


def mutate_motor(cfg: FittsMotorConfig, rng: random.Random, scale: float = 1.0) -> FittsMotorConfig:
    return FittsMotorConfig(
        fitts_a=max(0.07, cfg.fitts_a + rng.uniform(-0.02, 0.02) * scale),
        fitts_b=max(0.10, cfg.fitts_b + rng.uniform(-0.03, 0.03) * scale),
        target_width=max(20.0, cfg.target_width + rng.uniform(-8, 8) * scale),
        tremor_scale=max(0.25, cfg.tremor_scale + rng.uniform(-0.15, 0.15) * scale),
        curvature_scale=max(0.05, cfg.curvature_scale + rng.uniform(-0.05, 0.05) * scale),
        pause_probability=min(0.45, max(0.05, cfg.pause_probability + rng.uniform(-0.06, 0.06) * scale)),
        overshoot_probability=min(0.55, max(0.08, cfg.overshoot_probability + rng.uniform(-0.08, 0.08) * scale)),
        hesitation_ms=(
            max(12.0, cfg.hesitation_ms[0] + rng.uniform(-12, 12) * scale),
            max(40.0, cfg.hesitation_ms[1] + rng.uniform(-20, 20) * scale),
        ),
        sample_hz=(
            max(50.0, cfg.sample_hz[0] + rng.uniform(-8, 8) * scale),
            max(70.0, cfg.sample_hz[1] + rng.uniform(-12, 12) * scale),
        ),
        dwell_ms=(
            max(70.0, cfg.dwell_ms[0] + rng.uniform(-18, 18) * scale),
            max(110.0, cfg.dwell_ms[1] + rng.uniform(-30, 30) * scale),
        ),
        wander_strength=max(2.0, cfg.wander_strength + rng.uniform(-1.0, 1.0) * scale),
        pause_ms=(
            max(140.0, cfg.pause_ms[0] + rng.uniform(-35, 35) * scale),
            max(220.0, cfg.pause_ms[1] + rng.uniform(-50, 50) * scale),
        ),
        pre_click_delay_ms=(
            max(25.0, cfg.pre_click_delay_ms[0] + rng.uniform(-20, 20) * scale),
            max(50.0, cfg.pre_click_delay_ms[1] + rng.uniform(-35, 35) * scale),
        ),
    )


def _add_move(events: List[Dict[str, Any]], x: float, y: float, t: float) -> None:
    events.append({
        "x": round(x + random.uniform(-0.6, 0.6)),
        "y": round(y + random.uniform(-0.6, 0.6)),
        "t": round(t, 2),
        "type": "mousemove",
    })


def wander(
    events: List[Dict[str, Any]],
    pos: Tuple[float, float],
    t0: float,
    ms: float,
    strength: float,
) -> float:
    steps = max(2, int(ms / random.uniform(16, 28)))
    for i in range(1, steps + 1):
        _add_move(
            events,
            pos[0] + random.uniform(-strength, strength),
            pos[1] + random.uniform(-strength, strength),
            t0 + ms * (i / steps),
        )
    return t0 + ms


def fitts_move_to(
    events: List[Dict[str, Any]],
    start: Tuple[float, float],
    end: Tuple[float, float],
    t0: float,
    motor: FittsMotorConfig,
    depth: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    if depth > 2:
        _add_move(events, end[0], end[1], t0)
        return t0, end

    sx, sy = start
    tx, ty = end
    distance = math.hypot(tx - sx, ty - sy)
    if distance < 1.0:
        _add_move(events, tx, ty, t0)
        return t0, end

    fitts_id = math.log2(1.0 + distance / max(5.0, motor.target_width))
    total_duration_s = motor.fitts_a + motor.fitts_b * fitts_id + random.uniform(-0.02, 0.04)
    total_duration_s = max(0.16, total_duration_s)
    total_duration_ms = total_duration_s * 1000.0

    ballistic_ratio = random.uniform(0.86, 0.94)
    bx = sx + (tx - sx) * ballistic_ratio + random.uniform(-4.0, 4.0)
    by = sy + (ty - sy) * ballistic_ratio + random.uniform(-4.0, 4.0)

    dx_dir = bx - sx
    dy_dir = by - sy
    perp_x = -dy_dir / (distance + 1e-6)
    perp_y = dx_dir / (distance + 1e-6)
    curve_style = random.choice(["arc", "s_curve", "elbow"])
    curve_amp = (
        random.uniform(0.05, 0.20) * distance * motor.curvature_scale * random.choice([-1.0, 1.0])
    )

    phase1_ms = total_duration_ms * random.uniform(0.74, 0.82)
    hz = random.uniform(*motor.sample_hz)
    steps1 = max(14, int(phase1_ms / 1000.0 * hz))
    dt1 = phase1_ms / steps1

    f1, f2 = random.uniform(7.5, 9.5), random.uniform(10.5, 12.5)
    phi1, phi2 = random.uniform(0.0, math.tau), random.uniform(0.0, math.tau)
    tremor = motor.tremor_scale
    t = t0
    t_accum = 0.0

    for i in range(steps1):
        tau = (i + 1) / steps1
        poly = minimum_jerk(tau)
        base_x = sx + (bx - sx) * poly
        base_y = sy + (by - sy) * poly
        if curve_style == "arc":
            curve_offset = math.sin(math.pi * tau) * curve_amp
        elif curve_style == "s_curve":
            curve_offset = math.sin(2.0 * math.pi * tau) * curve_amp * 0.6
        else:
            curve_offset = 4.0 * tau * (1.0 - tau) * curve_amp
        tremor_x = tremor * (
            0.8 * math.sin(math.tau * f1 * t_accum + phi1)
            + 0.4 * math.cos(math.tau * f2 * t_accum + phi2)
        )
        tremor_y = tremor * (
            0.8 * math.cos(math.tau * f1 * t_accum + phi1)
            + 0.4 * math.sin(math.tau * f2 * t_accum + phi2)
        )
        t += dt1
        _add_move(
            events,
            base_x + perp_x * curve_offset + tremor_x,
            base_y + perp_y * curve_offset + tremor_y,
            t,
        )
        t_accum += phase1_ms / steps1 / 1000.0

    if random.random() < motor.pause_probability:
        t += random.uniform(25.0, 130.0)

    phase2_ms = total_duration_ms - phase1_ms
    steps2 = max(8, int(phase2_ms / 1000.0 * hz))
    dt2 = phase2_ms / steps2
    last_x, last_y = events[-1]["x"], events[-1]["y"]

    for i in range(steps2):
        tau = (i + 1) / steps2
        poly = minimum_jerk(tau)
        cx = last_x + (tx - last_x) * poly
        cy = last_y + (ty - last_y) * poly
        tremor_x = tremor * 0.45 * math.sin(math.tau * f1 * t_accum + phi1)
        tremor_y = tremor * 0.45 * math.cos(math.tau * f1 * t_accum + phi1)
        t += dt2
        _add_move(events, cx + tremor_x, cy + tremor_y, t)
        t_accum += phase2_ms / steps2 / 1000.0

    if random.random() < motor.overshoot_probability:
        ux = (tx - sx) / (distance + 1e-6)
        uy = (ty - sy) / (distance + 1e-6)
        overshoot = random.uniform(4.0, 16.0)
        ox, oy = tx + ux * overshoot, ty + uy * overshoot
        t += random.uniform(18.0, 50.0)
        t, (lx, ly) = fitts_move_to(events, (tx, ty), (ox, oy), t, motor, depth + 1)
        t += random.uniform(28.0, 80.0)
        t, _ = fitts_move_to(events, (lx, ly), (tx, ty), t, motor, depth + 1)

    return t, (tx, ty)


def fitts_click_at(
    events: List[Dict[str, Any]],
    x: float,
    y: float,
    t0: float,
    motor: FittsMotorConfig,
) -> float:
    t = t0 + random.uniform(*motor.hesitation_ms)
    ox = x + random.gauss(0.0, 1.4)
    oy = y + random.gauss(0.0, 1.4)
    events.append({"x": round(ox), "y": round(oy), "t": round(t, 2), "type": "mousedown"})
    dwell = random.uniform(*motor.dwell_ms)
    micro = random.randint(2, 5)
    for i in range(1, micro + 1):
        _add_move(
            events,
            ox + random.gauss(0.0, 0.8),
            oy + random.gauss(0.0, 0.8),
            t + dwell * (i / micro),
        )
    t_up = t + dwell
    events.append({"x": round(ox), "y": round(oy), "t": round(t_up, 2), "type": "mouseup"})
    return t_up
