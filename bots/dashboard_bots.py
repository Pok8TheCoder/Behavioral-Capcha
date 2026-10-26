"""Latest dashboard challenge bots: wide-arc hops, min-jerk, simulated humanoid."""

from __future__ import annotations

import os
import random
import secrets
import sys
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.challenge_bot_factory import CHALLENGE_BOT_STYLES, prelude_bot
from bots.challenge_solver import _maze_waypoints, solve_challenge
from bots.telemetry_export import ensure_export_timestamps
from test.evasion_bot import default_motor, generate_events as fitts_generate
from test.fast_direct_bot import FAMILY_PRESETS as DART_PRESETS
from test.fast_direct_bot import click_at as dart_click
from test.fast_direct_bot import travel_along as dart_travel
from test.hi_jit_bot import FAMILY_PRESETS as HI_JIT_PRESETS
from test.idle_correct_bot import FAMILY_PRESETS as ARC_PRESETS
from test.minjerk_bot import FAMILY_PRESETS as MINJERK_PRESETS
from test.minjerk_bot import click_at as minjerk_click
from test.minjerk_bot import travel_along as minjerk_travel
from test.playwright_dense_family_bot import cfg_from_preset as dense_cfg
from test.playwright_dense_family_bot import click_at as dense_click
from test.playwright_dense_family_bot import travel_along as dense_travel
from test.playwright_hop_bot import click_at as hop_click
from test.playwright_hop_bot import default_hop_cfg
from test.playwright_hop_bot import travel_along as hop_travel
from test.sparse_irregular_bot import FAMILY_PRESETS as SPARSE_PRESETS
from test.sparse_irregular_bot import click_at as sparse_click
from test.sparse_irregular_bot import travel_along as sparse_travel
from test.wobble_mid_bot import FAMILY_PRESETS as WOBBLE_PRESETS

Point = Tuple[float, float]
TravelFn = Callable[..., Tuple[float, Point]]
ClickFn = Callable[..., float]


def _start_pos(challenge: Dict[str, Any], rng: random.Random | None = None) -> Point:
    roll = rng or random
    area = challenge.get("area") or {"x": 80, "y": 120, "width": 840, "height": 480}
    return (
        float(area["x"]) + roll.uniform(40.0, 160.0),
        float(area["y"]) + roll.uniform(40.0, 160.0),
    )


def _motion_plan(challenge: Dict[str, Any]) -> Tuple[str, List[Point]]:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]
    if ctype in ("shape_trace", "zigzag_trace", "spiral_trace"):
        return "drag", [(float(p[0]), float(p[1])) for p in spec["guide_points"]]
    if ctype == "connect_dots":
        dots = {d["id"]: d for d in spec["dots"]}
        return "drag", [(float(dots[i]["x"]), float(dots[i]["y"])) for i in spec["sequence"]]
    if ctype == "maze_navigate":
        pts = list(_maze_waypoints(spec))
        end = (float(spec["end"][0]), float(spec["end"][1]))
        if not pts or pts[-1] != end:
            pts.append(end)
        return "drag", pts
    if ctype == "hazard_run":
        return "drag", [(float(p[0]), float(p[1])) for p in spec["waypoints"]]
    if ctype == "path_gates":
        pts = [(float(spec["start"][0]), float(spec["start"][1]))]
        for gate in spec["gates"]:
            pts.append((float(gate["x"]), float(gate["gap_y"])))
        pts.append((float(spec["end"][0]), float(spec["end"][1])))
        return "drag", pts
    if ctype == "button_sequence":
        buttons = {b["id"]: b for b in spec["buttons"]}
        return "click", [(float(buttons[i]["x"]), float(buttons[i]["y"])) for i in spec["sequence"]]
    if ctype == "pin_entry":
        keys = {k["digit"]: k for k in spec["keys"]}
        return "click", [(float(keys[d]["x"]), float(keys[d]["y"])) for d in spec["code"]]
    if ctype == "target_hunt":
        return "click", [(float(t["x"]), float(t["y"])) for t in spec["targets"]]
    raise ValueError(f"Unsupported challenge type: {ctype}")


def _append(events: List[Dict[str, Any]], x: float, y: float, t: float, etype: str) -> float:
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


def generate_family_events(
    challenge: Dict[str, Any],
    travel_along: TravelFn,
    click_at: ClickFn,
    cfg: Any,
) -> List[Dict[str, Any]]:
    mode, targets = _motion_plan(challenge)
    rng = random.Random(int(challenge.get("seed") or 0) ^ secrets.randbits(32))
    events: List[Dict[str, Any]] = []
    t = rng.uniform(40.0, 90.0)
    pos = _start_pos(challenge, rng)
    t = _append(events, pos[0], pos[1], t, "mousemove")
    pause = getattr(cfg, "pause_ms", (80.0, 180.0))

    if mode == "drag":
        if not targets:
            return events
        t, pos = travel_along(events, pos, targets[0], t, cfg)
        t = _append(events, pos[0], pos[1], t, "mousedown")
        for target in targets[1:]:
            t, pos = travel_along(events, pos, target, t, cfg)
        t = _append(events, pos[0], pos[1], t + rng.uniform(20.0, 60.0), "mouseup")
        return events

    if mode == "fly":
        for target in targets:
            t, pos = travel_along(events, pos, target, t, cfg)
            t += rng.uniform(*pause)
        return events

    for target in targets:
        t, pos = travel_along(events, pos, target, t, cfg)
        t = click_at(events, target[0], target[1], t, cfg)
        pos = target
        t += rng.uniform(*pause)
    return events


def _run_style_bot(challenge: Dict[str, Any], style_id: str) -> List[Dict[str, Any]]:
    rng = random.Random(int(challenge.get("seed") or 0) ^ secrets.randbits(32))
    bot, solve_style, fast = CHALLENGE_BOT_STYLES[style_id](rng)
    prelude_bot(bot, rng, 1920, 1080)
    solve_challenge(bot, challenge, style=solve_style, fast=fast)
    return ensure_export_timestamps(list(bot.events), rng)


def _family(travel, click, cfg):
    def _run(challenge: Dict[str, Any]) -> List[Dict[str, Any]]:
        return generate_family_events(challenge, travel, click, cfg)
    return _run


DASHBOARD_BOTS: List[Dict[str, Any]] = [
    {"id": "wide_arc", "label": "Wide Arc", "group": "Latest families",
     "run": _family(sparse_travel, sparse_click, ARC_PRESETS["wide_arc"])},
    {"id": "idle_correct", "label": "Idle Correct", "group": "Latest families",
     "run": _family(sparse_travel, sparse_click, ARC_PRESETS["idle_correct"])},
    {"id": "wobble_mid", "label": "Wobble Mid", "group": "Latest families",
     "run": _family(sparse_travel, sparse_click, WOBBLE_PRESETS["wobble_mid"])},
    {"id": "hi_jit", "label": "Hi Jitter", "group": "Latest families",
     "run": _family(sparse_travel, sparse_click, HI_JIT_PRESETS["hi_jit"])},
    {"id": "minjerk", "label": "Min-jerk", "group": "Latest families",
     "run": _family(minjerk_travel, minjerk_click, MINJERK_PRESETS["minjerk_direct"])},
    {"id": "dense_smooth", "label": "Dense Smooth", "group": "Latest families",
     "run": _family(minjerk_travel, minjerk_click, MINJERK_PRESETS["dense_smooth"])},
    {"id": "sparse_irregular", "label": "Sparse Irregular", "group": "Hop / dart",
     "run": _family(sparse_travel, sparse_click, SPARSE_PRESETS["sparse_irregular"])},
    {"id": "hop", "label": "Playwright Hop", "group": "Hop / dart",
     "run": lambda challenge: generate_family_events(
         challenge, hop_travel, hop_click, default_hop_cfg(),
     )},
    {"id": "fast_direct", "label": "Dart Hop", "group": "Hop / dart",
     "run": _family(dart_travel, dart_click, DART_PRESETS["fast_direct"])},
    {"id": "mid_dense", "label": "Mid Dense", "group": "Hop / dart",
     "run": _family(dense_travel, dense_click, dense_cfg("mid_dense"))},
    {"id": "fitts", "label": "Fitts Motor", "group": "Fitts / humanoid",
     "run": lambda challenge: fitts_generate(challenge, default_motor())},
    {"id": "simulated", "label": "Simulated Humanoid", "group": "Fitts / humanoid",
     "run": lambda challenge: _run_style_bot(challenge, "humanoid_simulated")},
    {"id": "humanoid_fast", "label": "Humanoid Fitts", "group": "Fitts / humanoid",
     "run": lambda challenge: _run_style_bot(challenge, "humanoid_fast")},
    {"id": "linear_fast", "label": "Linear", "group": "Baseline",
     "run": lambda challenge: _run_style_bot(challenge, "linear_fast")},
    {"id": "bezier_fast", "label": "Bezier", "group": "Baseline",
     "run": lambda challenge: _run_style_bot(challenge, "bezier_fast")},
    {"id": "naive", "label": "Naive", "group": "Baseline",
     "run": lambda challenge: _run_style_bot(challenge, "naive")},
]

_BOT_BY_ID = {entry["id"]: entry for entry in DASHBOARD_BOTS}


def list_dashboard_bots() -> List[Dict[str, str]]:
    return [{"id": b["id"], "label": b["label"], "group": b["group"]} for b in DASHBOARD_BOTS]


def run_dashboard_bot(challenge: Dict[str, Any], bot_name: str) -> List[Dict[str, Any]]:
    name = bot_name.lower().strip()
    entry = _BOT_BY_ID.get(name)
    if not entry:
        raise ValueError(f"Unknown dashboard bot: {bot_name}")
    return entry["run"](challenge)
