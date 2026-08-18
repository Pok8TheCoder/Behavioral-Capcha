"""Bot solvers for randomized behavioral challenge tasks."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

from bots.base_bot import BaseBot


def _uses_simulated_time(bot: BaseBot) -> bool:
    return bool(getattr(bot, "simulated", False))


def _maybe_sleep(bot: BaseBot, seconds: float, fast: bool) -> None:
    if _uses_simulated_time(bot):
        if not fast and seconds > 0:
            base_ms = max(20.0, seconds * 1000.0)
            bot.advance(random.uniform(base_ms * 0.75, base_ms * 1.35 + 55.0))
        return
    if not fast and seconds > 0:
        time.sleep(seconds)


def _inter_action_pause(bot: BaseBot, fast: bool) -> None:
    if _uses_simulated_time(bot) and not fast:
        bot.advance(random.uniform(90.0, 280.0))
    elif not fast:
        time.sleep(random.uniform(0.09, 0.24))


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
        _maybe_sleep(bot, step_delay, fast)


def _move_segment(bot: BaseBot, x0: float, y0: float, x1: float, y1: float, fast: bool, style: str) -> None:
    if fast or style != "humanoid":
        _move_path(bot, [(x0, y0), (x1, y1)], 0.0, steps_per_segment=16, fast=True)
        return
    bot.move_to(x0, y0, x1, y1)


def _click_at(bot: BaseBot, x: float, y: float, dwell: float, fast: bool) -> None:
    bot.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
    if _uses_simulated_time(bot):
        bot.advance(max(42.0, random.gauss(88.0, 18.0)))
    else:
        _maybe_sleep(bot, dwell, fast)
    bot.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")


def _drag_along_guide(bot: BaseBot, guide_points: List[List[float]], style: str, fast: bool) -> None:
    points = [(p[0], p[1]) for p in guide_points]
    if not points:
        return
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    bot.record_raw(start_x, start_y, event_type=1, target_id=1, evt_name="mouse_click_down")
    if style == "humanoid" and not fast:
        for i in range(len(points) - 1):
            bot.move_to(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    else:
        _move_path(bot, points, 0.006, steps_per_segment=10, fast=fast)
    bot.record_raw(end_x, end_y, event_type=2, target_id=1, evt_name="mouse_click_up")


def solve_shape_trace(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    _drag_along_guide(bot, spec["guide_points"], style, fast)


def solve_zigzag_trace(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    _drag_along_guide(bot, spec["guide_points"], style, fast)


def solve_spiral_trace(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    _drag_along_guide(bot, spec["guide_points"], style, fast)


def solve_connect_dots(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    dots = {d["id"]: d for d in spec["dots"]}
    points = [(dots[i]["x"], dots[i]["y"]) for i in spec["sequence"]]
    if not points:
        return
    start_x, start_y = points[0]
    bot.record_raw(start_x, start_y, event_type=1, target_id=1, evt_name="mouse_click_down")
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if style == "humanoid" and not fast:
            bot.move_to(x0, y0, x1, y1)
        else:
            _move_path(bot, [(x0, y0), (x1, y1)], 0.006, steps_per_segment=22, fast=fast)
        bot.record_raw(x1, y1, event_type=0, evt_name="mouse_move")
    end_x, end_y = points[-1]
    bot.record_raw(end_x, end_y, event_type=2, target_id=1, evt_name="mouse_click_up")


def solve_target_hunt(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    dwell = 0.01 if style == "linear" or fast else 0.02
    targets = spec["targets"]
    last_x, last_y = targets[0]["x"], targets[0]["y"]
    for target in targets:
        _move_segment(bot, last_x, last_y, target["x"], target["y"], fast, style)
        _click_at(bot, target["x"], target["y"], dwell, fast)
        last_x, last_y = target["x"], target["y"]
        _inter_action_pause(bot, fast)


def solve_button_sequence(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    buttons = {b["id"]: b for b in spec["buttons"]}
    dwell = 0.01 if style == "linear" or fast else 0.02
    last_x, last_y = spec["buttons"][0]["x"], spec["buttons"][0]["y"]
    for btn_id in spec["sequence"]:
        btn = buttons[btn_id]
        _move_segment(bot, last_x, last_y, btn["x"], btn["y"], fast, style)
        _click_at(bot, btn["x"], btn["y"], dwell, fast)
        last_x, last_y = btn["x"], btn["y"]
        _inter_action_pause(bot, fast)


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
        _inter_action_pause(bot, fast)


def _maze_waypoints(spec: Dict[str, Any]) -> List[Tuple[float, float]]:
    if spec.get("waypoints"):
        return [(p[0], p[1]) for p in spec["waypoints"]]
    from challenge_server.maze import solve_maze_path

    maze = spec["maze"]
    ox, oy = spec["origin"]
    cw, ch = spec["cell_width"], spec["cell_height"]
    path = solve_maze_path(maze)
    return [(ox + (cx + 0.5) * cw, oy + (cy + 0.5) * ch) for cx, cy in path]


def solve_maze(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    waypoints = _maze_waypoints(spec)
    if len(waypoints) < 2:
        return
    end_x, end_y = spec["end"][0], spec["end"][1]
    if (waypoints[-1][0], waypoints[-1][1]) != (end_x, end_y):
        waypoints = list(waypoints) + [(end_x, end_y)]
    _drag_along_waypoints(bot, waypoints, end_x, end_y, style, fast)


def solve_hazard_run(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    waypoints = [(p[0], p[1]) for p in spec["waypoints"]]
    end_x, end_y = spec["end"][0], spec["end"][1]
    _drag_along_waypoints(bot, waypoints, end_x, end_y, style, fast, steps_per_segment=36)


def _drag_along_waypoints(
    bot: BaseBot,
    waypoints: List[Tuple[float, float]],
    end_x: float,
    end_y: float,
    style: str,
    fast: bool,
    steps_per_segment: int = 18,
) -> None:
    if not waypoints:
        return
    start_x, start_y = waypoints[0]
    bot.record_raw(start_x, start_y, event_type=1, target_id=1, evt_name="mouse_click_down")
    if style == "humanoid" and not fast:
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            bot.move_to(x0, y0, x1, y1)
    else:
        _move_path(bot, waypoints, 0.005, steps_per_segment=steps_per_segment, fast=fast)
    bot.record_raw(end_x, end_y, event_type=0, evt_name="mouse_move")
    bot.record_raw(end_x, end_y, event_type=2, target_id=1, evt_name="mouse_click_up")


def solve_path_gates(bot: BaseBot, spec: Dict[str, Any], style: str, fast: bool) -> None:
    points: List[Tuple[float, float]] = [(spec["start"][0], spec["start"][1])]
    for gate in spec["gates"]:
        points.append((gate["x"], gate["gap_y"]))
    points.append((spec["end"][0], spec["end"][1]))
    end_x, end_y = spec["end"][0], spec["end"][1]
    _drag_along_waypoints(bot, points, end_x, end_y, style, fast, steps_per_segment=14)


def solve_challenge(bot: BaseBot, challenge: Dict[str, Any], style: str = "linear", fast: bool = False) -> None:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    solvers = {
        "shape_trace": solve_shape_trace,
        "zigzag_trace": solve_zigzag_trace,
        "spiral_trace": solve_spiral_trace,
        "connect_dots": solve_connect_dots,
        "target_hunt": solve_target_hunt,
        "button_sequence": solve_button_sequence,
        "pin_entry": solve_pin_entry,
        "maze_navigate": solve_maze,
        "hazard_run": solve_hazard_run,
        "path_gates": solve_path_gates,
    }
    solver = solvers.get(ctype)
    if solver:
        solver(bot, spec, style, fast)
