"""Human-like button sequence bot for Behavioral CAPTCHA API testing."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

API_BASE = "http://127.0.0.1:8010"


@dataclass
class BotConfig:
    move_interval_ms: tuple[float, float] = (12.0, 22.0)
    dwell_ms: tuple[float, float] = (95.0, 180.0)
    pause_ms: tuple[float, float] = (180.0, 420.0)
    max_speed_px_ms: float = 1.8
    jitter_px: float = 2.5
    overshoot_px: float = 6.0


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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)


def bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], t: float) -> tuple[float, float]:
    u = 1 - t
    x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
    y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
    return x, y


def add_move(events: list[dict[str, Any]], x: float, y: float, t: float) -> None:
    events.append(
        {
            "x": round(x + random.uniform(-0.8, 0.8)),
            "y": round(y + random.uniform(-0.8, 0.8)),
            "t": round(t, 2),
            "type": "mousemove",
        }
    )


def move_to(
    events: list[dict[str, Any]],
    start: tuple[float, float],
    end: tuple[float, float],
    t0: float,
    cfg: BotConfig,
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return t0

    # Control point offset for a natural arc.
    nx, ny = (-dy / dist, dx / dist)
    offset = random.uniform(-cfg.overshoot_px, cfg.overshoot_px)
    mid = ((start[0] + end[0]) / 2 + nx * offset, (start[1] + end[1]) / 2 + ny * offset)

    # Choose duration from max speed cap.
    min_duration = dist / cfg.max_speed_px_ms
    duration = max(min_duration, random.uniform(280, 900) * (dist / 500))
    steps = max(8, int(duration / random.uniform(*cfg.move_interval_ms)))

    for i in range(1, steps + 1):
        raw = i / steps
        eased = ease_in_out(raw)
        x, y = bezier(start, mid, end, eased)
        t = t0 + duration * raw
        add_move(events, x, y, t)

    return t0 + duration


def click_at(events: list[dict[str, Any]], x: float, y: float, t0: float, cfg: BotConfig) -> float:
    ox = x + random.uniform(-cfg.jitter_px, cfg.jitter_px)
    oy = y + random.uniform(-cfg.jitter_px, cfg.jitter_px)
    events.append({"x": round(ox), "y": round(oy), "t": round(t0, 2), "type": "mousedown"})

    dwell = random.uniform(*cfg.dwell_ms)
    micro_steps = random.randint(2, 5)
    for i in range(1, micro_steps + 1):
        frac = i / micro_steps
        mt = t0 + dwell * frac
        add_move(
            events,
            ox + random.uniform(-1.2, 1.2),
            oy + random.uniform(-1.2, 1.2),
            mt,
        )

    t_up = t0 + dwell
    events.append({"x": round(ox), "y": round(oy), "t": round(t_up, 2), "type": "mouseup"})
    return t_up


def generate_events(challenge: dict[str, Any], cfg: BotConfig) -> list[dict[str, Any]]:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    sequence = spec["sequence"]

    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    start_x = area["x"] + random.uniform(40, area["width"] - 40)
    start_y = area["y"] + random.uniform(40, area["height"] - 40)

    events: list[dict[str, Any]] = []
    t = random.uniform(120.0, 260.0)
    pos = (start_x, start_y)
    add_move(events, pos[0], pos[1], t)

    for btn_id in sequence:
        btn = buttons[btn_id]
        target = (btn["x"], btn["y"])
        t = move_to(events, pos, target, t, cfg)
        pos = target
        t = click_at(events, target[0], target[1], t + random.uniform(20, 60), cfg)
        t += random.uniform(*cfg.pause_ms)
        add_move(events, pos[0], pos[1], t)

    return events


def run_attempt(cfg: BotConfig | None = None, seed: int | None = None) -> dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    challenge = api_post("/api/challenge/new", {"challenge_type": "button_sequence"})
    cfg = cfg or BotConfig()
    events = generate_events(challenge, cfg)
    result = api_post(
        "/api/challenge/verify",
        {
            "session_id": challenge["session_id"],
            "events": events,
            "participant": "bot_custom_humanlike",
        },
    )
    result["_seed"] = challenge.get("seed")
    result["_event_count"] = len(events)
    return result


def sweep_configs(attempts: int = 30) -> None:
    best = {"human_confidence": -1.0}
    configs = [
        BotConfig(max_speed_px_ms=1.2, dwell_ms=(120, 220), pause_ms=(250, 500)),
        BotConfig(max_speed_px_ms=1.5, dwell_ms=(100, 180), pause_ms=(200, 450)),
        BotConfig(max_speed_px_ms=1.8, dwell_ms=(95, 170), pause_ms=(180, 400)),
        BotConfig(max_speed_px_ms=2.0, dwell_ms=(90, 160), pause_ms=(160, 360)),
        BotConfig(max_speed_px_ms=1.0, dwell_ms=(140, 240), pause_ms=(300, 550)),
    ]

    for i in range(attempts):
        cfg = random.choice(configs)
        # jitter config slightly
        cfg = BotConfig(
            move_interval_ms=(cfg.move_interval_ms[0] + random.uniform(-2, 2), cfg.move_interval_ms[1] + random.uniform(-2, 2)),
            dwell_ms=(cfg.dwell_ms[0] + random.uniform(-10, 10), cfg.dwell_ms[1] + random.uniform(-10, 10)),
            pause_ms=(cfg.pause_ms[0] + random.uniform(-20, 20), cfg.pause_ms[1] + random.uniform(-20, 20)),
            max_speed_px_ms=cfg.max_speed_px_ms + random.uniform(-0.15, 0.15),
            jitter_px=cfg.jitter_px + random.uniform(-0.5, 0.5),
            overshoot_px=cfg.overshoot_px + random.uniform(-2, 2),
        )
        try:
            result = run_attempt(cfg)
        except Exception as exc:
            print(f"attempt {i+1} failed: {exc}")
            continue

        hc = float(result.get("human_confidence") or 0)
        flags = (result.get("evaluation") or {}).get("flags") or []
        print(
            f"attempt {i+1:02d}: human_confidence={hc:.4f} "
            f"bot_prob={(result.get('evaluation') or {}).get('bot_probability')} "
            f"events={result['_event_count']} flags={flags}"
        )
        if hc > best.get("human_confidence", -1):
            best = {**result, "human_confidence": hc, "cfg": cfg.__dict__}
        if hc >= 0.80:
            print("TARGET REACHED")
            print(json.dumps(best, indent=2, default=str))
            return

    print("BEST RESULT:")
    print(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    sweep_configs(40)
