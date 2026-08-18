"""Test a fixed bot config against the active API model."""

from __future__ import annotations

import json
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Any

API_BASE = "http://127.0.0.1:8010"

BEST_CFG = {
    "max_speed_px_ms": 0.7316875895872461,
    "dwell_ms": [114.62991296279344, 255.6147550655669],
    "pause_ms": [279.6033811448404, 531.9694939812301],
    "move_interval_ms": [18.471673210841097, 30.745156930123535],
    "jitter_px": 4.468538585944212,
    "overshoot_px": 22.266995999175176,
    "pre_click_delay_ms": [57.820268261401765, 197.16327163697687],
    "wander_strength": 4.602561447658741,
}


@dataclass
class BotConfig:
    max_speed_px_ms: float
    dwell_ms: tuple[float, float]
    pause_ms: tuple[float, float]
    move_interval_ms: tuple[float, float]
    jitter_px: float
    overshoot_px: float
    pre_click_delay_ms: tuple[float, float]
    wander_strength: float


def cfg_from_dict(d: dict) -> BotConfig:
    return BotConfig(
        max_speed_px_ms=d["max_speed_px_ms"],
        dwell_ms=tuple(d["dwell_ms"]),
        pause_ms=tuple(d["pause_ms"]),
        move_interval_ms=tuple(d["move_interval_ms"]),
        jitter_px=d["jitter_px"],
        overshoot_px=d["overshoot_px"],
        pre_click_delay_ms=tuple(d["pre_click_delay_ms"]),
        wander_strength=d["wander_strength"],
    )


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def bezier(p0, p1, p2, t):
    u = 1 - t
    return (
        u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
    )


def add_move(events, x, y, t):
    events.append(
        {
            "x": round(x + random.uniform(-0.9, 0.9)),
            "y": round(y + random.uniform(-0.9, 0.9)),
            "t": round(t, 2),
            "type": "mousemove",
        }
    )


def wander(events, pos, t0, ms, strength):
    steps = max(2, int(ms / random.uniform(18, 30)))
    for i in range(1, steps + 1):
        add_move(
            events,
            pos[0] + random.uniform(-strength, strength),
            pos[1] + random.uniform(-strength, strength),
            t0 + ms * (i / steps),
        )
    return t0 + ms


def move_to(events, start, end, t0, cfg):
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return t0, end
    nx, ny = (-dy / dist, dx / dist)
    offset = random.uniform(-cfg.overshoot_px, cfg.overshoot_px)
    mid = ((start[0] + end[0]) / 2 + nx * offset, (start[1] + end[1]) / 2 + ny * offset)
    duration = max(dist / cfg.max_speed_px_ms, random.uniform(350, 1100) * (dist / 450))
    steps = max(6, int(duration / random.uniform(*cfg.move_interval_ms)))
    for i in range(1, steps + 1):
        raw = i / steps
        x, y = bezier(start, mid, end, ease(raw))
        add_move(events, x, y, t0 + duration * raw)
    return t0 + duration, end


def click_at(events, x, y, t0, cfg):
    ox = x + random.uniform(-cfg.jitter_px, cfg.jitter_px)
    oy = y + random.uniform(-cfg.jitter_px, cfg.jitter_px)
    events.append({"x": round(ox), "y": round(oy), "t": round(t0, 2), "type": "mousedown"})
    dwell = random.uniform(*cfg.dwell_ms)
    micro = random.randint(2, 6)
    for i in range(1, micro + 1):
        add_move(events, ox + random.uniform(-1.5, 1.5), oy + random.uniform(-1.5, 1.5), t0 + dwell * (i / micro))
    t_up = t0 + dwell
    events.append({"x": round(ox), "y": round(oy), "t": round(t_up, 2), "type": "mouseup"})
    return t_up


def generate_events(challenge, cfg: BotConfig):
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events = []
    t = random.uniform(150, 350)
    pos = (area["x"] + random.uniform(60, 180), area["y"] + random.uniform(60, 180))
    add_move(events, pos[0], pos[1], t)
    t = wander(events, pos, t, random.uniform(120, 280), cfg.wander_strength)
    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        t, pos = move_to(events, pos, target, t, cfg)
        t += random.uniform(*cfg.pre_click_delay_ms)
        t = wander(events, pos, t, random.uniform(40, 120), cfg.wander_strength * 0.4)
        t = click_at(events, target[0], target[1], t, cfg)
        pos = (target[0] + random.uniform(-2, 2), target[1] + random.uniform(-2, 2))
        t += random.uniform(*cfg.pause_ms)
        t = wander(events, pos, t, random.uniform(80, 220), cfg.wander_strength * 0.6)
    return events


def main():
    cfg = cfg_from_dict(BEST_CFG)
    scores = []
    for i in range(25):
        challenge = api_post("/api/challenge/new", {"challenge_type": "button_sequence"})
        events = generate_events(challenge, cfg)
        result = api_post(
            "/api/challenge/verify",
            {
                "session_id": challenge["session_id"],
                "events": events,
                "participant": "bot_custom_humanlike",
            },
        )
        hc = float(result.get("human_confidence") or 0)
        model = (result.get("evaluation") or {}).get("model_type")
        scores.append(hc)
        print(f"{i+1:02d} model={model} hc={hc:.4f} bot_prob={(result.get('evaluation') or {}).get('bot_probability')}")
    print("max", max(scores))
    print("avg", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
