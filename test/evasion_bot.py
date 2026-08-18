"""
Generalized evasion bot: Fitts-law + minimum-jerk motor, curved paths, dwell, wander.
No human replay — synthesizes telemetry from challenge geometry.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from test.fitts_motor import (
    FittsMotorConfig,
    default_motor,
    fitts_click_at,
    fitts_move_to,
    motor_from_dict,
    mutate_motor,
    wander,
)

MAX_DRAG_EVENTS = 6000


def default_cfg() -> FittsMotorConfig:
    return default_motor()


def cfg_from_dict(d: Dict[str, Any]) -> FittsMotorConfig:
    return motor_from_dict(d)


def mutate_cfg(cfg: FittsMotorConfig, rng: random.Random, scale: float = 1.0) -> FittsMotorConfig:
    return mutate_motor(cfg, rng, scale)


def _event_type_csv(raw: str) -> str:
    mapping = {
        "mousemove": "mouse_move",
        "mousedown": "mouse_click_down",
        "mouseup": "mouse_click_up",
    }
    return mapping.get(raw, raw)


def _drag_segment(
    events: List[Dict[str, Any]],
    start: Tuple[float, float],
    end: Tuple[float, float],
    t: float,
    motor: FittsMotorConfig,
    dragging: bool,
) -> Tuple[float, Tuple[float, float]]:
    t, pos = fitts_move_to(events, start, end, t, motor)
    if dragging and random.random() < 0.5:
        t = wander(events, pos, t, random.uniform(50, 160), motor.wander_strength * 0.35)
    return t, pos


def _drag_stroke(
    events: List[Dict[str, Any]],
    points: List[Tuple[float, float]],
    t: float,
    motor: FittsMotorConfig,
) -> Tuple[float, Tuple[float, float]]:
    if not points:
        return t, (0, 0)
    if len(points) > 80:
        stride = max(1, len(points) // 80)
        points = points[::stride]

    pos = points[0]
    events.append({"x": round(pos[0]), "y": round(pos[1]), "t": round(t, 2), "type": "mousedown"})
    t += random.uniform(40, 120)
    t = wander(events, pos, t, random.uniform(50, 140), motor.wander_strength * 0.4)

    for target in points[1:]:
        if len(events) >= MAX_DRAG_EVENTS:
            break
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = math.hypot(dx, dy)
        splits = min(3, max(1, int(dist / 220)))
        for step in range(1, splits + 1):
            if len(events) >= MAX_DRAG_EVENTS:
                break
            frac = step / splits
            sub = (pos[0] + dx * frac, pos[1] + dy * frac)
            t, pos = _drag_segment(events, pos, sub, t, motor, dragging=True)
            t += random.uniform(10, 40)
        t += random.uniform(20, 80)

    events.append({"x": round(pos[0]), "y": round(pos[1]), "t": round(t, 2), "type": "mouseup"})
    return t + random.uniform(30, 90), pos


def _click_targets(
    events: List[Dict[str, Any]],
    targets: List[Tuple[float, float]],
    t: float,
    pos: Tuple[float, float],
    motor: FittsMotorConfig,
) -> Tuple[float, Tuple[float, float]]:
    for tx, ty in targets:
        t, pos = fitts_move_to(events, pos, (tx, ty), t, motor)
        t += random.uniform(*motor.pre_click_delay_ms)
        t = wander(events, pos, t, random.uniform(40, 130), motor.wander_strength * 0.45)
        t = fitts_click_at(events, tx, ty, t, motor)
        pos = (tx, ty)
        t += random.uniform(*motor.pause_ms)
        t = wander(events, pos, t, random.uniform(60, 180), motor.wander_strength * 0.5)
    return t, pos


def generate_events(challenge: Dict[str, Any], motor: FittsMotorConfig) -> List[Dict[str, Any]]:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: List[Dict[str, Any]] = []
    t = random.uniform(150, 400)
    pos = (area["x"] + random.uniform(60, 200), area["y"] + random.uniform(60, 200))
    events.append({"x": round(pos[0]), "y": round(pos[1]), "t": round(t, 2), "type": "mousemove"})
    t = wander(events, pos, t, random.uniform(120, 300), motor.wander_strength)

    if ctype in ("shape_trace", "zigzag_trace", "spiral_trace"):
        pts = [(p[0], p[1]) for p in spec["guide_points"]]
        t, pos = _drag_stroke(events, pts, t, motor)
    elif ctype == "connect_dots":
        dots = {d["id"]: d for d in spec["dots"]}
        pts = [(dots[i]["x"], dots[i]["y"]) for i in spec["sequence"]]
        t, pos = _drag_stroke(events, pts, t, motor)
    elif ctype == "maze_navigate":
        pts = [(spec["start"][0], spec["start"][1])]
        if spec.get("waypoints"):
            pts.extend([(p[0], p[1]) for p in spec["waypoints"]])
        pts.append((spec["end"][0], spec["end"][1]))
        t, pos = _drag_stroke(events, pts, t, motor)
    elif ctype == "hazard_run":
        pts = [(p[0], p[1]) for p in spec.get("waypoints", [spec["start"], spec["end"]])]
        t, pos = _drag_stroke(events, pts, t, motor)
    elif ctype == "path_gates":
        pts = [(spec["start"][0], spec["start"][1])]
        for g in spec["gates"]:
            pts.append((g["x"], g["gap_y"]))
        pts.append((spec["end"][0], spec["end"][1]))
        t, pos = _drag_stroke(events, pts, t, motor)
    elif ctype == "button_sequence":
        buttons = {b["id"]: b for b in spec["buttons"]}
        targets = [(buttons[i]["x"], buttons[i]["y"]) for i in spec["sequence"]]
        t, pos = _click_targets(events, targets, t, pos, motor)
    elif ctype == "pin_entry":
        keys = {k["digit"]: k for k in spec["keys"]}
        targets = [(keys[d]["x"], keys[d]["y"]) for d in spec["code"]]
        t, pos = _click_targets(events, targets, t, pos, motor)
    elif ctype == "target_hunt":
        targets = [(t["x"], t["y"]) for t in spec["targets"]]
        t, pos = _click_targets(events, targets, t, pos, motor)
    else:
        t, pos = fitts_move_to(events, pos, (480, 320), t, motor)

    return events


def events_to_csv_rows(
    events: List[Dict[str, Any]],
    t_offset: float = 0.0,
    target_item: str = "evasion_bot",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for e in events:
        x, y = float(e["x"]), float(e["y"])
        rows.append({
            "rel_time_ms": f"{t_offset + float(e['t']):.3f}",
            "x_px": f"{x:.1f}",
            "y_px": f"{y:.1f}",
            "x_norm": f"{x / 1920:.5f}",
            "y_norm": f"{y / 1080:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": "0",
            "scroll_dy": "0",
            "event_type": _event_type_csv(e.get("type", "mousemove")),
            "target_item": target_item,
        })
    return rows


def motor_to_dict(motor: FittsMotorConfig) -> Dict[str, Any]:
    return asdict(motor)
