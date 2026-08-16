"""Algorithmically random behavioral challenge session generator."""

from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Tuple

from challenge_server.maze import generate_maze, maze_cell_centers


CHALLENGE_TYPES = (
    "shape_trace",
    "button_sequence",
    "maze_navigate",
    "pin_entry",
    "path_gates",
)

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


def _shape_trace(rng: random.Random, session_id: str) -> Dict[str, Any]:
    shape = rng.choice(["circle", "square", "triangle", "star"])
    cx, cy = _point_in_area(rng, 80)
    size = rng.uniform(55, 95)
    guide_points = _shape_points(shape, cx, cy, size, rng)
    return {
        "session_id": session_id,
        "challenge_type": "shape_trace",
        "title": f"Trace the {shape}",
        "instructions": f"Draw the {shape} outline by moving your cursor along the glowing guide.",
        "area": AREA,
        "spec": {
            "shape": shape,
            "guide_points": guide_points,
            "tolerance_px": rng.uniform(18, 28),
            "min_coverage_ratio": 0.62,
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

    return {
        "session_id": session_id,
        "challenge_type": "maze_navigate",
        "title": "Maze navigation",
        "instructions": "Move your cursor from the green start through the maze to the red exit without touching walls.",
        "area": AREA,
        "spec": {
            "maze": maze,
            "origin": [origin_x, origin_y],
            "cell_width": cell_w,
            "cell_height": cell_h,
            "start": start,
            "end": end,
            "wall_tolerance_px": 6,
            "finish_tolerance_px": 28,
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
        "instructions": "Move through each gate opening from left to right without hitting the bars.",
        "area": AREA,
        "spec": {
            "gates": gates,
            "start": start,
            "end": end,
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
    }
    challenge = builders[chosen](rng, session_id)
    challenge["seed"] = seed if seed is not None else rng.randint(1, 2_000_000_000)
    return challenge


def validate_completion(challenge: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    moves = [e for e in events if str(e.get("type", e.get("event_type", ""))).lower() in ("mousemove", "mouse_move", "0")]
    clicks = [e for e in events if "down" in str(e.get("type", e.get("event_type", "")).lower()) or str(e.get("type", "")) == "1"]

    if ctype == "shape_trace":
        guide = spec["guide_points"]
        tol = spec["tolerance_px"]
        if len(moves) < 12:
            return {"passed": False, "reason": "Too few movement samples for shape trace."}
        covered = 0
        for gx, gy in guide:
            if any(math.hypot(m["x"] - gx, m["y"] - gy) <= tol for m in moves):
                covered += 1
        ratio = covered / max(1, len(guide))
        passed = ratio >= spec["min_coverage_ratio"]
        return {"passed": passed, "reason": f"Shape coverage {ratio:.0%}", "coverage_ratio": ratio}

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
        if len(moves) < 20:
            return {"passed": False, "reason": "Too few movement samples for maze."}
        end = spec["end"]
        finish_tol = spec["finish_tolerance_px"]
        reached_end = any(math.hypot(m["x"] - end[0], m["y"] - end[1]) <= finish_tol for m in moves)
        return {"passed": reached_end, "reason": "Reached maze exit." if reached_end else "Did not reach maze exit."}

    if ctype == "path_gates":
        gates = spec["gates"]
        tol = spec["gate_tolerance_px"]
        if len(moves) < 15:
            return {"passed": False, "reason": "Too few movement samples for gate path."}
        passed_gates = 0
        for gate in gates:
            gx, gap_y, gap_half = gate["x"], gate["gap_y"], gate["gap_half"]
            for m in moves:
                if abs(m["x"] - gx) <= 18 and abs(m["y"] - gap_y) <= gap_half + tol:
                    passed_gates += 1
                    break
        end = spec["end"]
        reached_end = any(math.hypot(m["x"] - end[0], m["y"] - end[1]) <= spec["finish_tolerance_px"] for m in moves)
        passed = passed_gates >= len(gates) and reached_end
        return {
            "passed": passed,
            "reason": f"Passed {passed_gates}/{len(gates)} gates",
            "gates_passed": passed_gates,
        }

    return {"passed": False, "reason": "Unknown challenge type."}
