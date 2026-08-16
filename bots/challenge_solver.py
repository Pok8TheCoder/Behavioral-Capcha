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


def _move_path(
    bot: BaseBot,
    points: List[Tuple[float, float]],
    step_delay: float,
    steps_per_segment: int = 12,
    fast: bool = False,
) -> None:
    for x, y in _interp_points(points, steps_per_segment):
        bot.record_raw(x, y, event_type=0, evt_name="mouse_move")
        if not fast and step_delay > 0:
            time.sleep(step_delay)


def _move_segment(bot: BaseBot, x0: float, y0: float, x1: float, y1: float, fast: bool, style: str) -> None:
    if fast or style != "humanoid":
        _move_path(bot, [(x0, y0), (x1, y1)], 0.0, steps_per_segment=14, fast=True)
        return
    bot.move_to(x0, y0, x1, y1)


def _click_at(bot: BaseBot, x: float, y: float, dwell: float, fast: bool) -> None:
    bot.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
    if not fast and dwell > 0:
        time.sleep(dwell)
    bot.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")


def solve_shape_trace(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    points = [(p[0], p[1]) for p in spec["guide_points"]]
    if style == "humanoid" and not fast:
        for i in range(len(points) - 1):
            bot.move_to(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    else:
        _move_path(bot, points, 0.006, steps_per_segment=8, fast=fast)


def solve_button_sequence(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    buttons = {b["id"]: b for b in spec["buttons"]}
    dwell = 0.01 if style == "linear" or fast else 0.02
    last_x, last_y = spec["buttons"][0]["x"], spec["buttons"][0]["y"]
    for btn_id in spec["sequence"]:
        btn = buttons[btn_id]
        _move_segment(bot, last_x, last_y, btn["x"], btn["y"], fast, style)
        _click_at(bot, btn["x"], btn["y"], dwell, fast)
        last_x, last_y = btn["x"], btn["y"]
        if not fast:
            time.sleep(dwell)


def solve_pin_entry(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    keys = {k["digit"]: k for k in spec["keys"]}
    dwell = 0.01 if style == "linear" or fast else 0.02
    first = spec["keys"][0]
    last_x, last_y = first["x"], first["y"]
    for digit in spec["code"]:
        key = keys[digit]
        _move_segment(bot, last_x, last_y, key["x"], key["y"], fast, style)
        _click_at(bot, key["x"], key["y"], dwell, fast)
        last_x, last_y = key["x"], key["y"]
        if not fast:
            time.sleep(dwell)


def solve_maze(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    maze = spec["maze"]
    ox, oy = spec["origin"]
    cw, ch = spec["cell_width"], spec["cell_height"]
    path = solve_maze_path(maze)
    points = [(ox + (cx + 0.5) * cw, oy + (cy + 0.5) * ch) for cx, cy in path]
    if points:
        points.append((spec["end"][0], spec["end"][1]))
    if style == "humanoid" and not fast:
        if not path:
            return
        sx = ox + (path[0][0] + 0.5) * cw
        sy = oy + (path[0][1] + 0.5) * ch
        for cx, cy in path[1:]:
            tx = ox + (cx + 0.5) * cw
            ty = oy + (cy + 0.5) * ch
            bot.move_to(sx, sy, tx, ty)
            sx, sy = tx, ty
        bot.move_to(sx, sy, spec["end"][0], spec["end"][1])
    else:
        _move_path(bot, points, 0.005, steps_per_segment=10, fast=fast)
        bot.record_raw(spec["end"][0], spec["end"][1], event_type=0, evt_name="mouse_move")


def solve_path_gates(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    points: List[Tuple[float, float]] = [(spec["start"][0], spec["start"][1])]
    for gate in spec["gates"]:
        points.append((gate["x"], gate["gap_y"]))
    points.append((spec["end"][0], spec["end"][1]))
    if style == "humanoid" and not fast:
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            bot.move_to(x0, y0, x1, y1)
    else:
        _move_path(bot, points, 0.006, steps_per_segment=14, fast=fast)


def solve_challenge(bot: BaseBot, challenge: Dict[str, Any], style: str = "linear", fast: bool = False) -> None:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    solvers = {
        "shape_trace": solve_shape_trace,
        "button_sequence": solve_button_sequence,
        "pin_entry": solve_pin_entry,
        "maze_navigate": solve_maze,
        "path_gates": solve_path_gates,
    }
    solver = solvers.get(ctype)
    if solver:
        solver(bot, spec, style, fast)
