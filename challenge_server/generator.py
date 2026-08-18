"""Algorithmically random behavioral challenge session generator."""

from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Tuple

from challenge_server.maze import generate_maze, maze_cell_centers, maze_path_pixels, solve_maze_path


CHALLENGE_TYPES = (
    "shape_trace",
    "button_sequence",
    "maze_navigate",
    "pin_entry",
    "path_gates",
    "zigzag_trace",
    "spiral_trace",
    "connect_dots",
    "target_hunt",
    "hazard_run",
)

CHALLENGE_CATALOG = {
    "shape_trace": {
        "title": "Shape trace",
        "interaction": "Click-drag-release",
        "summary": "Draw a random circle, square, triangle, or star along the glowing outline.",
    },
    "button_sequence": {
        "title": "Button sequence",
        "interaction": "Click in order",
        "summary": "Click numbered buttons from 1 to N in the correct order.",
    },
    "maze_navigate": {
        "title": "Maze navigation",
        "interaction": "Click-drag-release",
        "summary": "Drag from the green start through corridors to the red exit.",
    },
    "pin_entry": {
        "title": "PIN entry",
        "interaction": "Click keypad",
        "summary": "Enter the shown numeric code on the on-screen keypad.",
    },
    "path_gates": {
        "title": "Gate runner",
        "interaction": "Click-drag-release",
        "summary": "Click and hold the green start, drag through each gate, release on red.",
    },
    "zigzag_trace": {
        "title": "Zigzag trace",
        "interaction": "Click-drag-release",
        "summary": "Trace the glowing zigzag path in one continuous stroke.",
    },
    "spiral_trace": {
        "title": "Spiral trace",
        "interaction": "Click-drag-release",
        "summary": "Follow the spiral guide from center outward in one stroke.",
    },
    "connect_dots": {
        "title": "Connect the dots",
        "interaction": "Click-drag-release",
        "summary": "Drag through numbered dots in order without lifting the mouse.",
    },
    "target_hunt": {
        "title": "Target hunt",
        "interaction": "Click targets",
        "summary": "Click every glowing target circle on the arena.",
    },
    "hazard_run": {
        "title": "Hazard run",
        "interaction": "Click-drag-release",
        "summary": "Drag from green to red while avoiding the hazard zones.",
    },
}

AREA = {"x": 80, "y": 120, "width": 840, "height": 480}


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed if seed is not None else random.randrange(1, 2_000_000_000))


def _point_in_area(rng: random.Random, margin: float = 40) -> Tuple[float, float]:
    return (
        rng.uniform(AREA["x"] + margin, AREA["x"] + AREA["width"] - margin),
        rng.uniform(AREA["y"] + margin, AREA["y"] + AREA["height"] - margin),
    )


def _shape_points(shape: str, cx: float, cy: float, size: float, rng: random.Random) -> List[List[float]]:
    if shape == "circle":
        n = rng.randint(28, 40)
        pts = []
        for i in range(n + 1):
            theta = 2 * math.pi * i / n
            pts.append([cx + size * math.cos(theta), cy + size * math.sin(theta)])
        return pts

    if shape == "square":
        half = size * 0.75
        corners = [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
            [cx - half, cy - half],
        ]
        return corners

    if shape == "triangle":
        angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
        pts = [[cx + size * math.cos(a), cy + size * math.sin(a)] for a in angles]
        pts.append(pts[0])
        return pts

    # star
    pts: List[List[float]] = []
    for i in range(11):
        angle = math.pi / 2 + i * math.pi / 5
        radius = size if i % 2 == 0 else size * 0.45
        pts.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    return pts


def _guide_trace_challenge(
    rng: random.Random,
    session_id: str,
    challenge_type: str,
    title: str,
    instructions: str,
    guide_points: List[List[float]],
    **spec_extra: Any,
) -> Dict[str, Any]:
    spec = {
        "guide_points": guide_points,
        "tolerance_px": rng.uniform(18, 28),
        "min_coverage_ratio": spec_extra.pop("min_coverage_ratio", 0.55),
        "min_stroke_points": spec_extra.pop("min_stroke_points", 20),
    }
    spec.update(spec_extra)
    return {
        "session_id": session_id,
        "challenge_type": challenge_type,
        "title": title,
        "instructions": instructions,
        "area": AREA,
        "spec": spec,
    }


def _zigzag_trace(rng: random.Random, session_id: str) -> Dict[str, Any]:
    n_zigs = rng.randint(7, 11)
    x0 = AREA["x"] + 50
    x1 = AREA["x"] + AREA["width"] - 50
    base_y = rng.uniform(AREA["y"] + 100, AREA["y"] + AREA["height"] - 100)
    amp = rng.uniform(55, 95)
    guide: List[List[float]] = []
    for i in range(n_zigs):
        t = i / max(1, n_zigs - 1)
        x = x0 + t * (x1 - x0)
        y = base_y + (amp if i % 2 == 0 else -amp)
        guide.append([x, y])
    return _guide_trace_challenge(
        rng, session_id, "zigzag_trace", "Zigzag trace",
        "Click and hold, trace the zigzag guide in one stroke, then release.",
        guide, min_coverage_ratio=0.58,
    )


def _spiral_trace(rng: random.Random, session_id: str) -> Dict[str, Any]:
    cx, cy = _point_in_area(rng, 100)
    max_r = rng.uniform(70, 110)
    turns = rng.uniform(2.0, 3.5)
    n_pts = rng.randint(45, 65)
    guide: List[List[float]] = []
    for i in range(n_pts + 1):
        theta = 2 * math.pi * turns * (i / n_pts)
        r = max_r * (i / n_pts)
        guide.append([cx + r * math.cos(theta), cy + r * math.sin(theta)])
    return _guide_trace_challenge(
        rng, session_id, "spiral_trace", "Spiral trace",
        "Click and hold, spiral outward along the guide in one stroke, then release.",
        guide, min_coverage_ratio=0.52, min_stroke_points=25,
    )


def _connect_dots(rng: random.Random, session_id: str) -> Dict[str, Any]:
    count = rng.randint(5, 8)
    dots: List[Dict[str, Any]] = []
    used: List[Tuple[float, float]] = []
    for label in range(1, count + 1):
        for _ in range(50):
            x, y = _point_in_area(rng, 55)
            if all(math.hypot(x - ux, y - uy) > 75 for ux, uy in used):
                used.append((x, y))
                dots.append({"id": label, "x": round(x, 1), "y": round(y, 1), "radius": 24})
                break
        else:
            x, y = _point_in_area(rng, 55)
            dots.append({"id": label, "x": round(x, 1), "y": round(y, 1), "radius": 24})

    guide = [[d["x"], d["y"]] for d in dots]
    return {
        "session_id": session_id,
        "challenge_type": "connect_dots",
        "title": "Connect the dots",
        "instructions": f"Click and drag through dots 1 to {count} in order, then release.",
        "area": AREA,
        "spec": {
            "dots": dots,
            "sequence": list(range(1, count + 1)),
            "guide_points": guide,
            "dot_tolerance_px": 32,
            "min_stroke_points": 18,
        },
    }


def _target_hunt(rng: random.Random, session_id: str) -> Dict[str, Any]:
    count = rng.randint(5, 7)
    targets: List[Dict[str, Any]] = []
    used: List[Tuple[float, float]] = []
    colors = ["#22d3ee", "#a855f7", "#f59e0b", "#10b981", "#f43f5e"]
    for i in range(count):
        for _ in range(50):
            x, y = _point_in_area(rng, 45)
            if all(math.hypot(x - ux, y - uy) > 65 for ux, uy in used):
                used.append((x, y))
                targets.append({
                    "id": i + 1,
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "radius": rng.randint(22, 30),
                    "color": colors[i % len(colors)],
                })
                break
        else:
            x, y = _point_in_area(rng, 45)
            targets.append({
                "id": i + 1,
                "x": round(x, 1),
                "y": round(y, 1),
                "radius": 26,
                "color": colors[i % len(colors)],
            })

    return {
        "session_id": session_id,
        "challenge_type": "target_hunt",
        "title": "Target hunt",
        "instructions": f"Click all {count} target circles.",
        "area": AREA,
        "spec": {
            "targets": targets,
            "click_tolerance_px": 30,
        },
    }


def _point_in_hazard(x: float, y: float, hazards: List[Dict[str, Any]], margin: float) -> bool:
    return any(math.hypot(x - h["x"], y - h["y"]) < h["radius"] + margin for h in hazards)


def _segment_hits_hazard(
    x0: float, y0: float, x1: float, y1: float,
    hazards: List[Dict[str, Any]], margin: float, steps: int = 48,
) -> bool:
    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if _point_in_hazard(x, y, hazards, margin):
            return True
    return False


def _build_safe_hazard_waypoints(
    start: List[float],
    end: List[float],
    hazards: List[Dict[str, Any]],
    rng: random.Random,
    margin: float = 12,
) -> List[List[float]]:
    waypoints: List[List[float]] = [start]
    n_segments = rng.randint(6, 9)
    for i in range(1, n_segments):
        t = i / n_segments
        wx = start[0] + t * (end[0] - start[0])
        prev = waypoints[-1]
        wy: float | None = None
        for _ in range(50):
            candidate_y = start[1] + rng.uniform(-110, 110)
            if _point_in_hazard(wx, candidate_y, hazards, margin + 18):
                continue
            if not _segment_hits_hazard(prev[0], prev[1], wx, candidate_y, hazards, margin):
                wy = candidate_y
                break
        if wy is None:
            wy = start[1]
            if _segment_hits_hazard(prev[0], prev[1], wx, wy, hazards, margin):
                wy = AREA["y"] + 40
        waypoints.append([wx, wy])

    prev = waypoints[-1]
    if _segment_hits_hazard(prev[0], prev[1], end[0], end[1], hazards, margin):
        mid = [prev[0] + (end[0] - prev[0]) * 0.5, prev[1]]
        if not _segment_hits_hazard(prev[0], prev[1], mid[0], mid[1], hazards, margin):
            waypoints.append(mid)
    waypoints.append(end)
    return waypoints


def _hazard_run(rng: random.Random, session_id: str) -> Dict[str, Any]:
    start = [AREA["x"] + 40, AREA["y"] + AREA["height"] / 2]
    end = [AREA["x"] + AREA["width"] - 40, AREA["y"] + AREA["height"] / 2 + rng.uniform(-80, 80)]
    hazards: List[Dict[str, Any]] = []
    n_hazards = rng.randint(4, 6)
    for _ in range(n_hazards):
        hazards.append({
            "x": round(rng.uniform(AREA["x"] + 120, AREA["x"] + AREA["width"] - 120), 1),
            "y": round(rng.uniform(AREA["y"] + 60, AREA["y"] + AREA["height"] - 60), 1),
            "radius": round(rng.uniform(38, 58), 1),
        })

    margin = 12
    waypoints = _build_safe_hazard_waypoints(start, end, hazards, rng, margin)

    return {
        "session_id": session_id,
        "challenge_type": "hazard_run",
        "title": "Hazard run",
        "instructions": "Click and drag from green to red, avoiding red hazard zones, then release.",
        "area": AREA,
        "spec": {
            "start": start,
            "end": end,
            "hazards": hazards,
            "waypoints": waypoints,
            "hazard_margin_px": margin,
            "finish_tolerance_px": 34,
            "min_stroke_points": 18,
        },
    }


def _shape_trace(rng: random.Random, session_id: str) -> Dict[str, Any]:
    shape = rng.choice(["circle", "square", "triangle", "star"])
    cx, cy = _point_in_area(rng, 80)
    size = rng.uniform(55, 95)
    guide_points = _shape_points(shape, cx, cy, size, rng)
    return {
        "session_id": session_id,
        "challenge_type": "shape_trace",
        "title": f"Trace the {shape}",
        "instructions": "Click and hold, drag along the glowing outline in one stroke, then release to submit.",
        "area": AREA,
        "spec": {
            "shape": shape,
            "guide_points": guide_points,
            "tolerance_px": rng.uniform(18, 28),
            "min_coverage_ratio": 0.55,
            "min_stroke_points": 20,
        },
    }


def _button_sequence(rng: random.Random, session_id: str) -> Dict[str, Any]:
    count = rng.randint(4, 6)
    labels = list(range(1, count + 1))
    rng.shuffle(labels)
    buttons: List[Dict[str, Any]] = []
    used: List[Tuple[float, float]] = []

    for label in labels:
        for _ in range(40):
            x, y = _point_in_area(rng, 50)
            if all(math.hypot(x - ux, y - uy) > 70 for ux, uy in used):
                used.append((x, y))
                buttons.append({"id": label, "x": round(x, 1), "y": round(y, 1), "radius": 28})
                break
        else:
            x, y = _point_in_area(rng, 50)
            buttons.append({"id": label, "x": round(x, 1), "y": round(y, 1), "radius": 28})

    sequence = list(range(1, count + 1))
    return {
        "session_id": session_id,
        "challenge_type": "button_sequence",
        "title": "Button sequence",
        "instructions": f"Click the numbered buttons in order from 1 to {count}.",
        "area": AREA,
        "spec": {
            "buttons": buttons,
            "sequence": sequence,
            "click_tolerance_px": 34,
        },
    }


def _maze_navigate(rng: random.Random, session_id: str) -> Dict[str, Any]:
    cols = rng.randint(8, 11)
    rows = rng.randint(8, 11)
    maze = generate_maze(cols, rows, rng)
    cell_w = (AREA["width"] - 20) / cols
    cell_h = (AREA["height"] - 20) / rows
    origin_x = AREA["x"] + 10
    origin_y = AREA["y"] + 10
    start, end = maze_cell_centers(maze, origin_x, origin_y, cell_w, cell_h)
    waypoints = maze_path_pixels(maze, origin_x, origin_y, cell_w, cell_h)

    return {
        "session_id": session_id,
        "challenge_type": "maze_navigate",
        "title": "Maze navigation",
        "instructions": "Click and drag from the green start through the corridors to the red exit, then release.",
        "area": AREA,
        "spec": {
            "maze": maze,
            "origin": [origin_x, origin_y],
            "cell_width": cell_w,
            "cell_height": cell_h,
            "start": start,
            "end": end,
            "waypoints": waypoints,
            "wall_tolerance_px": 6,
            "finish_tolerance_px": 32,
        },
    }


def _pin_entry(rng: random.Random, session_id: str) -> Dict[str, Any]:
    length = rng.randint(4, 5)
    code = [rng.randint(0, 9) for _ in range(length)]
    keypad_origin_x = AREA["x"] + AREA["width"] * 0.55
    keypad_origin_y = AREA["y"] + 60
    key_w, key_h = 52, 48
    gap = 10
    keys: List[Dict[str, Any]] = []

    for digit in range(10):
        row, col = divmod(digit, 3)
        if digit == 0:
            row, col = 3, 1
        x = keypad_origin_x + col * (key_w + gap)
        y = keypad_origin_y + row * (key_h + gap)
        keys.append({"digit": digit, "x": round(x + key_w / 2, 1), "y": round(y + key_h / 2, 1), "w": key_w, "h": key_h})

    return {
        "session_id": session_id,
        "challenge_type": "pin_entry",
        "title": "PIN entry",
        "instructions": f"Enter the {length}-digit code using the on-screen keypad.",
        "area": AREA,
        "spec": {
            "code": code,
            "keys": keys,
            "click_tolerance_px": 30,
            "display_code": "".join(str(d) for d in code),
        },
    }


def _path_gates(rng: random.Random, session_id: str) -> Dict[str, Any]:
    gate_count = rng.randint(5, 7)
    gates: List[Dict[str, Any]] = []
    x_step = AREA["width"] / (gate_count + 1)
    last_gap_y = AREA["y"] + AREA["height"] / 2

    for i in range(gate_count):
        x = AREA["x"] + x_step * (i + 1)
        gap_y = rng.uniform(AREA["y"] + 80, AREA["y"] + AREA["height"] - 80)
        # Keep gaps somewhat continuous for humans
        if i > 0:
            gap_y = last_gap_y + rng.uniform(-60, 60)
            gap_y = max(AREA["y"] + 70, min(AREA["y"] + AREA["height"] - 70, gap_y))
        last_gap_y = gap_y
        gap_half = rng.uniform(42, 58)
        gates.append({
            "x": round(x, 1),
            "gap_y": round(gap_y, 1),
            "gap_half": round(gap_half, 1),
            "thickness": 8,
        })

    start = [AREA["x"] + 30, AREA["y"] + AREA["height"] / 2]
    end = [AREA["x"] + AREA["width"] - 30, last_gap_y]
    return {
        "session_id": session_id,
        "challenge_type": "path_gates",
        "title": "Gate runner",
        "instructions": "Click and hold the green start, drag through each gate opening, then release on the red finish.",
        "area": AREA,
        "spec": {
            "gates": gates,
            "start": start,
            "end": end,
            "start_tolerance_px": 32,
            "gate_tolerance_px": 26,
            "finish_tolerance_px": 36,
        },
    }


def generate_challenge(seed: int | None = None, challenge_type: str | None = None) -> Dict[str, Any]:
    rng = _rng(seed)
    session_id = str(uuid.uuid4())
    chosen = challenge_type or rng.choice(CHALLENGE_TYPES)
    if chosen not in CHALLENGE_TYPES:
        chosen = rng.choice(CHALLENGE_TYPES)

    builders = {
        "shape_trace": _shape_trace,
        "button_sequence": _button_sequence,
        "maze_navigate": _maze_navigate,
        "pin_entry": _pin_entry,
        "path_gates": _path_gates,
        "zigzag_trace": _zigzag_trace,
        "spiral_trace": _spiral_trace,
        "connect_dots": _connect_dots,
        "target_hunt": _target_hunt,
        "hazard_run": _hazard_run,
    }
    challenge = builders[chosen](rng, session_id)
    challenge["seed"] = seed if seed is not None else rng.randint(1, 2_000_000_000)
    return challenge


def _stroke_moves(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stroke: List[Dict[str, Any]] = []
    in_stroke = False
    for event in events:
        et = str(event.get("type", event.get("event_type", ""))).lower()
        if "down" in et or et == "1":
            in_stroke = True
            stroke = []
            continue
        if "up" in et or et == "2":
            if in_stroke:
                break
            in_stroke = False
            continue
        if in_stroke and et in ("mousemove", "mouse_move", "0"):
            stroke.append(event)
    return stroke


def _stroke_origin(events: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    origin: Dict[str, Any] | None = None
    in_stroke = False
    for event in events:
        et = str(event.get("type", event.get("event_type", ""))).lower()
        if "down" in et or et == "1":
            origin = event
            in_stroke = True
            continue
        if "up" in et or et == "2":
            if in_stroke:
                return origin
            in_stroke = False
    return origin if in_stroke else None


def _validate_guide_stroke(events: List[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    guide = spec["guide_points"]
    tol = spec.get("tolerance_px", 22)
    stroke_moves = _stroke_moves(events)
    min_points = spec.get("min_stroke_points", 20)
    if len(stroke_moves) < min_points:
        return {"passed": False, "reason": f"Stroke too short ({len(stroke_moves)}/{min_points} points)."}
    covered = sum(
        1 for gx, gy in guide
        if any(math.hypot(m["x"] - gx, m["y"] - gy) <= tol for m in stroke_moves)
    )
    ratio = covered / max(1, len(guide))
    passed = ratio >= spec.get("min_coverage_ratio", 0.55)
    return {"passed": passed, "reason": f"Path coverage {ratio:.0%}", "coverage_ratio": ratio}


def validate_completion(challenge: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    moves = [e for e in events if str(e.get("type", e.get("event_type", ""))).lower() in ("mousemove", "mouse_move", "0")]
    clicks = [e for e in events if "down" in str(e.get("type", e.get("event_type", "")).lower()) or str(e.get("type", "")) == "1"]

    if ctype == "shape_trace":
        return _validate_guide_stroke(events, spec)

    if ctype in ("zigzag_trace", "spiral_trace"):
        return _validate_guide_stroke(events, spec)

    if ctype == "connect_dots":
        stroke_moves = _stroke_moves(events)
        if len(stroke_moves) < spec.get("min_stroke_points", 18):
            return {"passed": False, "reason": "Drag through all numbered dots in one stroke."}
        tol = spec["dot_tolerance_px"]
        dots = {d["id"]: d for d in spec["dots"]}
        hit_order: List[int] = []
        remaining = list(spec["sequence"])
        for m in stroke_moves:
            if not remaining:
                break
            next_id = remaining[0]
            dot = dots[next_id]
            if math.hypot(m["x"] - dot["x"], m["y"] - dot["y"]) <= tol + dot.get("radius", 20):
                hit_order.append(next_id)
                remaining.pop(0)
        passed = hit_order == spec["sequence"]
        return {"passed": passed, "reason": f"Dot order {hit_order}, expected {spec['sequence']}", "dot_order": hit_order}

    if ctype == "target_hunt":
        tol = spec["click_tolerance_px"]
        hit_ids: List[int] = []
        for click in clicks:
            for target in spec["targets"]:
                if math.hypot(click["x"] - target["x"], click["y"] - target["y"]) <= tol + target.get("radius", 24):
                    if target["id"] not in hit_ids:
                        hit_ids.append(target["id"])
                    break
        expected = [t["id"] for t in spec["targets"]]
        passed = sorted(hit_ids) == sorted(expected)
        return {"passed": passed, "reason": f"Hit {len(hit_ids)}/{len(expected)} targets", "hit_targets": hit_ids}

    if ctype == "hazard_run":
        stroke_moves = _stroke_moves(events)
        if len(stroke_moves) < spec.get("min_stroke_points", 18):
            return {"passed": False, "reason": "Drag from start to end in one stroke."}
        margin = spec.get("hazard_margin_px", 12)
        for h in spec["hazards"]:
            for m in stroke_moves:
                if math.hypot(m["x"] - h["x"], m["y"] - h["y"]) < h["radius"] + margin:
                    return {"passed": False, "reason": "Path entered a hazard zone."}
        end = spec["end"]
        finish_tol = spec["finish_tolerance_px"]
        reached = any(math.hypot(m["x"] - end[0], m["y"] - end[1]) <= finish_tol for m in stroke_moves)
        return {"passed": reached, "reason": "Reached safe exit." if reached else "Release on the red exit."}

    if ctype == "button_sequence":
        sequence = spec["sequence"]
        buttons = {b["id"]: b for b in spec["buttons"]}
        tol = spec["click_tolerance_px"]
        hit_order: List[int] = []
        for click in clicks:
            for btn_id, btn in buttons.items():
                if math.hypot(click["x"] - btn["x"], click["y"] - btn["y"]) <= tol + btn.get("radius", 24):
                    hit_order.append(btn_id)
                    break
        passed = hit_order == sequence
        return {"passed": passed, "reason": f"Click order {hit_order}, expected {sequence}", "click_order": hit_order}

    if ctype == "pin_entry":
        code = spec["code"]
        keys = {k["digit"]: k for k in spec["keys"]}
        tol = spec["click_tolerance_px"]
        digits: List[int] = []
        for click in clicks:
            for digit, key in keys.items():
                if math.hypot(click["x"] - key["x"], click["y"] - key["y"]) <= tol:
                    digits.append(digit)
                    break
        passed = digits == code
        return {"passed": passed, "reason": f"Entered {digits}, expected {code}", "entered": digits}

    if ctype == "maze_navigate":
        stroke_moves = _stroke_moves(events)
        if len(stroke_moves) < 15:
            return {"passed": False, "reason": "Drag through the maze in one stroke (too few points)."}
        end = spec["end"]
        finish_tol = spec["finish_tolerance_px"]
        reached_end = any(math.hypot(m["x"] - end[0], m["y"] - end[1]) <= finish_tol for m in stroke_moves)
        return {"passed": reached_end, "reason": "Reached maze exit." if reached_end else "Release the drag on the red exit."}

    if ctype == "path_gates":
        gates = spec["gates"]
        tol = spec["gate_tolerance_px"]
        start = spec["start"]
        start_tol = spec.get("start_tolerance_px", 32)
        origin = _stroke_origin(events)
        stroke_moves = _stroke_moves(events)
        if origin is None or math.hypot(origin["x"] - start[0], origin["y"] - start[1]) > start_tol:
            return {"passed": False, "reason": "Click and hold from the green start."}
        if len(stroke_moves) < 15:
            return {"passed": False, "reason": "Drag through the gates in one stroke (too few points)."}
        passed_gates = 0
        for gate in gates:
            gx, gap_y, gap_half = gate["x"], gate["gap_y"], gate["gap_half"]
            for m in stroke_moves:
                if abs(m["x"] - gx) <= 18 and abs(m["y"] - gap_y) <= gap_half + tol:
                    passed_gates += 1
                    break
        end = spec["end"]
        reached_end = any(
            math.hypot(m["x"] - end[0], m["y"] - end[1]) <= spec["finish_tolerance_px"]
            for m in stroke_moves
        )
        passed = passed_gates >= len(gates) and reached_end
        return {
            "passed": passed,
            "reason": (
                f"Passed {passed_gates}/{len(gates)} gates"
                if passed
                else (
                    "Release on the red finish."
                    if passed_gates >= len(gates)
                    else f"Passed {passed_gates}/{len(gates)} gates"
                )
            ),
            "gates_passed": passed_gates,
        }

    return {"passed": False, "reason": "Unknown challenge type."}
