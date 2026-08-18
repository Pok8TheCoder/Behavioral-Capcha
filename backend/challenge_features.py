"""Challenge-specific human-capture features for later per-task model tuning."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _etype(event: Dict[str, Any]) -> str:
    raw = str(event.get("type", event.get("event_type", ""))).lower().replace("_", "")
    if raw in ("mousemove", "0"):
        return "move"
    if raw in ("mousedown", "mouseclickdown", "1"):
        return "down"
    if raw in ("mouseup", "mouseclickup", "2"):
        return "up"
    return raw


def _xy(event: Dict[str, Any]) -> Tuple[float, float]:
    return float(event.get("x", 0)), float(event.get("y", 0))


def _t(event: Dict[str, Any]) -> float:
    return float(event.get("t", 0))


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _clicks(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if _etype(e) == "down"]


def _click_dwells(events: List[Dict[str, Any]]) -> List[float]:
    dwells: List[float] = []
    down_t: Optional[float] = None
    for event in events:
        kind = _etype(event)
        if kind == "down":
            down_t = _t(event)
        elif kind == "up" and down_t is not None:
            dwells.append(max(0.0, _t(event) - down_t))
            down_t = None
    return dwells


def _inter_click_ms(clicks: List[Dict[str, Any]]) -> List[float]:
    if len(clicks) < 2:
        return []
    return [max(0.0, _t(clicks[i]) - _t(clicks[i - 1])) for i in range(1, len(clicks))]


def _stroke_moves(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stroke: List[Dict[str, Any]] = []
    in_stroke = False
    for event in events:
        kind = _etype(event)
        if kind == "down":
            in_stroke = True
            stroke = [event]
            continue
        if kind == "up":
            if in_stroke:
                stroke.append(event)
                break
            in_stroke = False
            continue
        if in_stroke and kind == "move":
            stroke.append(event)
    return stroke


def _path_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_dist(points[i][0], points[i][1], points[i - 1][0], points[i - 1][1]) for i in range(1, len(points)))


def _nearest_guide_dist(x: float, y: float, guide: List[List[float]]) -> float:
    if not guide:
        return -1.0
    return min(_dist(x, y, float(gx), float(gy)) for gx, gy in guide)


def _coverage_ratio(stroke: List[Dict[str, Any]], guide: List[List[float]], tol: float) -> float:
    if not guide or not stroke:
        return 0.0
    covered = 0
    for gx, gy in guide:
        if any(_dist(float(m["x"]), float(m["y"]), float(gx), float(gy)) <= tol for m in stroke):
            covered += 1
    return covered / len(guide)


def _mean(values: List[float]) -> Optional[float]:
    return round(mean(values), 4) if values else None


def _click_target_offsets(
    clicks: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    *,
    id_key: str = "id",
    radius_key: str = "radius",
    extra_tol: float = 0.0,
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for click in clicks:
        cx, cy = _xy(click)
        best = None
        best_d = 1e9
        for target in targets:
            d = _dist(cx, cy, float(target["x"]), float(target["y"]))
            if d < best_d:
                best_d = d
                best = target
        if best is None:
            continue
        radius = float(best.get(radius_key, 24))
        details.append({
            "t": round(_t(click), 2),
            "target_id": best.get(id_key),
            "offset_px": round(best_d, 2),
            "offset_over_radius": round(best_d / max(radius, 1.0), 4),
            "inside": best_d <= radius + extra_tol,
        })
    return details


def extract_challenge_features(
    challenge_type: str,
    spec: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    spec = spec or {}
    clicks = _clicks(events)
    dwells = _click_dwells(events)
    inter = _inter_click_ms(clicks)
    base = {
        "click_count": len(clicks),
        "avg_click_dwell_ms": _mean(dwells),
        "median_click_dwell_ms": round(sorted(dwells)[len(dwells) // 2], 2) if dwells else None,
        "inter_click_mean_ms": _mean(inter),
        "inter_click_min_ms": round(min(inter), 2) if inter else None,
        "inter_click_max_ms": round(max(inter), 2) if inter else None,
    }

    extractors = {
        "target_hunt": _target_hunt_features,
        "button_sequence": _button_sequence_features,
        "pin_entry": _pin_entry_features,
        "shape_trace": _guide_stroke_features,
        "zigzag_trace": _guide_stroke_features,
        "spiral_trace": _guide_stroke_features,
        "connect_dots": _connect_dots_features,
        "path_gates": _path_gates_features,
        "hazard_run": _hazard_run_features,
        "maze_navigate": _maze_features,
    }
    extra = extractors.get(challenge_type, lambda _s, _e: {})(spec, events)
    return {**base, **extra}


def _target_hunt_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    targets = spec.get("targets") or []
    tol = float(spec.get("click_tolerance_px", 30))
    details = _click_target_offsets(_clicks(events), targets, extra_tol=tol)
    offsets = [d["offset_px"] for d in details]
    inside = [d for d in details if d["inside"]]
    return {
        "target_count": len(targets),
        "hits": len({d["target_id"] for d in inside}),
        "click_offset_mean_px": _mean(offsets),
        "click_offset_min_px": round(min(offsets), 2) if offsets else None,
        "click_offset_max_px": round(max(offsets), 2) if offsets else None,
        "mean_offset_over_radius": _mean([d["offset_over_radius"] for d in details]),
        "clicks": details,
    }


def _button_sequence_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    buttons = spec.get("buttons") or []
    sequence = spec.get("sequence") or []
    tol = float(spec.get("click_tolerance_px", 34))
    details = _click_target_offsets(_clicks(events), buttons, extra_tol=tol)
    hit_order = [d["target_id"] for d in details if d["inside"]]
    offsets = [d["offset_px"] for d in details]
    return {
        "expected_sequence": sequence,
        "click_order": hit_order,
        "sequence_ok": hit_order == sequence,
        "click_offset_mean_px": _mean(offsets),
        "click_offset_min_px": round(min(offsets), 2) if offsets else None,
        "click_offset_max_px": round(max(offsets), 2) if offsets else None,
        "clicks": details,
    }


def _pin_entry_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = spec.get("keys") or []
    code = spec.get("code") or []
    tol = float(spec.get("click_tolerance_px", 30))
    mapped = []
    for key in keys:
        mapped.append({
            "id": key.get("digit"),
            "x": key.get("x"),
            "y": key.get("y"),
            "radius": min(float(key.get("w", 52)), float(key.get("h", 48))) / 2.0,
        })
    details = _click_target_offsets(_clicks(events), mapped, extra_tol=tol)
    entered = [d["target_id"] for d in details if d["inside"]]
    offsets = [d["offset_px"] for d in details]
    return {
        "expected_code": code,
        "entered": entered,
        "code_ok": entered == code,
        "click_offset_mean_px": _mean(offsets),
        "clicks": details,
    }


def _guide_stroke_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    guide = spec.get("guide_points") or []
    tol = float(spec.get("tolerance_px", 22))
    stroke = _stroke_moves(events)
    pts = [_xy(m) for m in stroke]
    errors = [_nearest_guide_dist(x, y, guide) for x, y in pts] if guide else []
    guide_len = _path_length([(float(g[0]), float(g[1])) for g in guide])
    stroke_len = _path_length(pts)
    duration = (_t(stroke[-1]) - _t(stroke[0])) if len(stroke) >= 2 else 0.0
    return {
        "shape": spec.get("shape"),
        "stroke_points": len(stroke),
        "stroke_length_px": round(stroke_len, 2),
        "guide_length_px": round(guide_len, 2),
        "length_ratio": round(stroke_len / guide_len, 4) if guide_len > 1 else None,
        "coverage_ratio": round(_coverage_ratio(stroke, guide, tol), 4),
        "mean_distance_to_guide_px": _mean(errors),
        "max_distance_to_guide_px": round(max(errors), 2) if errors else None,
        "stroke_duration_ms": round(duration, 2),
        "tolerance_px": round(tol, 2),
    }


def _connect_dots_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    dots = spec.get("dots") or []
    sequence = spec.get("sequence") or []
    tol = float(spec.get("dot_tolerance_px", 32))
    stroke = _stroke_moves(events)
    hit_order: List[int] = []
    remaining = list(sequence)
    closest: List[float] = []
    for move in stroke:
        mx, my = _xy(move)
        if remaining:
            nxt = next((d for d in dots if d.get("id") == remaining[0]), None)
            if nxt and _dist(mx, my, float(nxt["x"]), float(nxt["y"])) <= tol + float(nxt.get("radius", 20)):
                hit_order.append(remaining.pop(0))
        if dots:
            closest.append(min(_dist(mx, my, float(d["x"]), float(d["y"])) for d in dots))
    return {
        "expected_sequence": sequence,
        "dot_order": hit_order,
        "sequence_ok": hit_order == sequence,
        "mean_distance_to_nearest_dot_px": _mean(closest),
        "stroke_points": len(stroke),
    }


def _path_gates_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    gates = spec.get("gates") or []
    tol = float(spec.get("gate_tolerance_px", 16))
    start = spec.get("start") or [0, 0]
    start_tol = float(spec.get("start_tolerance_px", 32))
    origin = None
    in_stroke = False
    for event in events:
        kind = _etype(event)
        if kind == "down":
            origin = event
            in_stroke = True
        elif kind == "up" and in_stroke:
            break
    started_on_green = bool(
        origin and _dist(float(origin["x"]), float(origin["y"]), float(start[0]), float(start[1])) <= start_tol
    )
    stroke = _stroke_moves(events)
    moves = stroke if stroke else [e for e in events if _etype(e) == "move"]
    per_gate = []
    for gate in gates:
        gx = float(gate["x"])
        gap_y = float(gate["gap_y"])
        gap_half = float(gate["gap_half"])
        crossings = [e for e in moves if abs(float(e["x"]) - gx) <= 18]
        if not crossings:
            per_gate.append({"passed": False, "min_gap_error_px": None})
            continue
        errors = [abs(float(e["y"]) - gap_y) for e in crossings]
        min_err = min(errors)
        per_gate.append({
            "passed": min_err <= gap_half + tol,
            "min_gap_error_px": round(min_err, 2),
            "gap_half_px": gap_half,
        })
    errors = [g["min_gap_error_px"] for g in per_gate if g["min_gap_error_px"] is not None]
    finish_tol = float(spec.get("finish_tolerance_px", 36))
    end = spec.get("end") or [0, 0]
    finish = min((_dist(*_xy(m), float(end[0]), float(end[1])) for m in stroke), default=None)
    return {
        "gates_total": len(gates),
        "gates_passed": sum(1 for g in per_gate if g["passed"]),
        "mean_gap_error_px": _mean(errors),
        "gates": per_gate,
        "started_on_green": started_on_green,
        "stroke_points": len(stroke),
        "finish_distance_px": round(finish, 2) if finish is not None else None,
        "reached_end": finish is not None and finish <= finish_tol,
    }


def _hazard_run_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    hazards = spec.get("hazards") or []
    end = spec.get("end") or [0, 0]
    finish_tol = float(spec.get("finish_tolerance_px", 34))
    stroke = _stroke_moves(events)
    pts = [_xy(m) for m in stroke]
    min_hazard = None
    hit = False
    margin = float(spec.get("hazard_margin_px", 12))
    if hazards and pts:
        distances = []
        for x, y in pts:
            for h in hazards:
                d = _dist(x, y, float(h["x"]), float(h["y"])) - float(h["radius"])
                distances.append(d)
                if d < margin:
                    hit = True
        min_hazard = round(min(distances), 2)
    finish = min((_dist(x, y, float(end[0]), float(end[1])) for x, y in pts), default=None)
    return {
        "stroke_points": len(stroke),
        "min_clearance_to_hazard_px": min_hazard,
        "hit_hazard": hit,
        "finish_distance_px": round(finish, 2) if finish is not None else None,
        "reached_end": finish is not None and finish <= finish_tol,
        "stroke_length_px": round(_path_length(pts), 2),
    }


def _maze_features(spec: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    end = spec.get("end") or [0, 0]
    start = spec.get("start") or [0, 0]
    waypoints = spec.get("waypoints") or []
    finish_tol = float(spec.get("finish_tolerance_px", 32))
    stroke = _stroke_moves(events)
    pts = [_xy(m) for m in stroke]
    finish = min((_dist(x, y, float(end[0]), float(end[1])) for x, y in pts), default=None)
    ideal = _path_length([(float(w[0]), float(w[1])) for w in waypoints]) if waypoints else _dist(
        float(start[0]), float(start[1]), float(end[0]), float(end[1])
    )
    stroke_len = _path_length(pts)
    return {
        "stroke_points": len(stroke),
        "stroke_length_px": round(stroke_len, 2),
        "ideal_path_length_px": round(ideal, 2),
        "length_ratio": round(stroke_len / ideal, 4) if ideal > 1 else None,
        "finish_distance_px": round(finish, 2) if finish is not None else None,
        "reached_end": finish is not None and finish <= finish_tol,
    }
