"""Bot solvers for randomized behavioral challenge tasks."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from bots.base_bot import BaseBot
from challenge_server.maze import solve_maze_path


def _interp_points(points: List[Tuple[float, float]], steps_per_segment: int) -> List[Tuple[float, float]]:
    if not points:
        return []
    out: List[Tuple[float, float]] = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        for step in range(steps_per_segment):
            alpha = step / max(1, steps_per_segment - 1)
            out.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)))
    return out


def _move_path(bot: BaseBot, points: List[Tuple[float, float]], step_delay: float, steps_per_segment: int = 12) -> None:
    for x, y in _interp_points(points, steps_per_segment):
        bot.record_raw(x, y, event_type=0, evt_name="mouse_move")
        time.sleep(step_delay)


def solve_shape_trace_linear(bot: BaseBot, spec: Dict[str, Any], step_delay: float = 0.006) -> None:
    points = [(p[0], p[1]) for p in spec["guide_points"]]
    _move_path(bot, points, step_delay, steps_per_segment=8)


def solve_shape_trace_humanoid(bot: BaseBot, spec: Dict[str, Any]) -> None:
    points = spec["guide_points"]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        bot.move_to(x0, y0, x1, y1)


def solve_button_sequence(bot: BaseBot, spec: Dict[str, Any], dwell: float = 0.02) -> None:
    buttons = {b["id"]: b for b in spec["buttons"]}
    last_x, last_y = spec["buttons"][0]["x"], spec["buttons"][0]["y"]
    for btn_id in spec["sequence"]:
        btn = buttons[btn_id]
        bot.move_to(last_x, last_y, btn["x"], btn["y"])
        bot.click_target(btn["x"], btn["y"])
        last_x, last_y = btn["x"], btn["y"]
        time.sleep(dwell)


def solve_pin_entry(bot: BaseBot, spec: Dict[str, Any], dwell: float = 0.02) -> None:
    keys = {k["digit"]: k for k in spec["keys"]}
    first = spec["keys"][0]
    last_x, last_y = first["x"], first["y"]
    for digit in spec["code"]:
        key = keys[digit]
        bot.move_to(last_x, last_y, key["x"], key["y"])
        bot.click_target(key["x"], key["y"])
        last_x, last_y = key["x"], key["y"]
        time.sleep(dwell)


def solve_maze_linear(bot: BaseBot, spec: Dict[str, Any], step_delay: float = 0.005) -> None:
    maze = spec["maze"]
    ox, oy = spec["origin"]
    cw, ch = spec["cell_width"], spec["cell_height"]
    path = solve_maze_path(maze)
    points = [(ox + (cx + 0.5) * cw, oy + (cy + 0.5) * ch) for cx, cy in path]
    if points:
        points.append((spec["end"][0], spec["end"][1]))
    _move_path(bot, points, step_delay, steps_per_segment=10)
    bot.record_raw(spec["end"][0], spec["end"][1], event_type=0, evt_name="mouse_move")


def solve_maze_humanoid(bot: BaseBot, spec: Dict[str, Any]) -> None:
    maze = spec["maze"]
    ox, oy = spec["origin"]
    cw, ch = spec["cell_width"], spec["cell_height"]
    path = solve_maze_path(maze)
    if not path:
        return
    sx = ox + (path[0][0] + 0.5) * cw
    sy = oy + (path[0][1] + 0.5) * ch
    for cx, cy in path[1:]:
        tx = ox + (cx + 0.5) * cw
        ty = oy + (cy + 0.5) * ch
        bot.move_to(sx, sy, tx, ty)
        sx, sy = tx, ty


def solve_path_gates_linear(bot: BaseBot, spec: Dict[str, Any], step_delay: float = 0.006) -> None:
    points: List[Tuple[float, float]] = [(spec["start"][0], spec["start"][1])]
    for gate in spec["gates"]:
        points.append((gate["x"], gate["gap_y"]))
    points.append((spec["end"][0], spec["end"][1]))
    _move_path(bot, points, step_delay, steps_per_segment=14)


def solve_path_gates_humanoid(bot: BaseBot, spec: Dict[str, Any]) -> None:
    points: List[Tuple[float, float]] = [(spec["start"][0], spec["start"][1])]
    for gate in spec["gates"]:
        points.append((gate["x"], gate["gap_y"]))
    points.append((spec["end"][0], spec["end"][1]))
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        bot.move_to(x0, y0, x1, y1)


def solve_challenge(bot: BaseBot, challenge: Dict[str, Any], style: str = "linear") -> None:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]

    if ctype == "shape_trace":
        if style == "humanoid":
            solve_shape_trace_humanoid(bot, spec)
        else:
            solve_shape_trace_linear(bot, spec)
    elif ctype == "button_sequence":
        solve_button_sequence(bot, spec, dwell=0.01 if style == "linear" else 0.02)
    elif ctype == "pin_entry":
        solve_pin_entry(bot, spec, dwell=0.01 if style == "linear" else 0.02)
    elif ctype == "maze_navigate":
        if style == "humanoid":
            solve_maze_humanoid(bot, spec)
        else:
            solve_maze_linear(bot, spec)
    elif ctype == "path_gates":
        if style == "humanoid":
            solve_path_gates_humanoid(bot, spec)
        else:
            solve_path_gates_linear(bot, spec)
