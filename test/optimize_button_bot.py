"""Aggressive optimizer for button-sequence human confidence."""

from __future__ import annotations

import json
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Any

API_BASE = "http://127.0.0.1:8010"
TARGET = 0.80
MAX_ATTEMPTS = 800
REQUIRED_MODEL = "random_forest_challenge"


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
        frac = i / steps
        add_move(
            events,
            pos[0] + random.uniform(-strength, strength),
            pos[1] + random.uniform(-strength, strength),
            t0 + ms * frac,
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


def generate_events(challenge, cfg):
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


def random_cfg() -> BotConfig:
    return BotConfig(
        max_speed_px_ms=random.uniform(0.55, 1.35),
        dwell_ms=(random.uniform(110, 150), random.uniform(170, 260)),
        pause_ms=(random.uniform(220, 380), random.uniform(420, 750)),
        move_interval_ms=(random.uniform(14, 20), random.uniform(22, 34)),
        jitter_px=random.uniform(2.0, 5.5),
        overshoot_px=random.uniform(8, 28),
        pre_click_delay_ms=(random.uniform(30, 90), random.uniform(100, 220)),
        wander_strength=random.uniform(2, 10),
    )


def mutate_cfg(base: BotConfig, scale: float = 0.15) -> BotConfig:
    def n(v, lo, hi):
        return max(lo, min(hi, v * (1 + random.uniform(-scale, scale))))

    return BotConfig(
        max_speed_px_ms=n(base.max_speed_px_ms, 0.35, 1.8),
        dwell_ms=(n(base.dwell_ms[0], 80, 300), n(base.dwell_ms[1], 100, 350)),
        pause_ms=(n(base.pause_ms[0], 120, 900), n(base.pause_ms[1], 200, 1200)),
        move_interval_ms=base.move_interval_ms,
        jitter_px=n(base.jitter_px, 1, 8),
        overshoot_px=n(base.overshoot_px, 4, 40),
        pre_click_delay_ms=base.pre_click_delay_ms,
        wander_strength=n(base.wander_strength, 1, 16),
    )


def run_once(cfg: BotConfig) -> dict[str, Any]:
    challenge = api_post("/api/challenge/new", {"challenge_type": "button_sequence"})
    events = generate_events(challenge, cfg)
    result = api_post(
        "/api/challenge/verify",
        {
            "session_id": challenge["session_id"],
            "events": events,
            "participant": "bot_optimizer",
        },
    )
    result["_cfg"] = cfg.__dict__
    return result


def ensure_model() -> None:
    health = api_post("/api/models/activate", {"model_name": REQUIRED_MODEL})
    active = health.get("active_model")
    if active != REQUIRED_MODEL:
        raise RuntimeError(f"Expected model {REQUIRED_MODEL}, got {active}")


def main():
    ensure_model()
    best_hc = -1.0
    best = None
    elite: list[tuple[float, BotConfig]] = []

    for i in range(1, MAX_ATTEMPTS + 1):
        if elite and random.random() < 0.55:
            _, parent = random.choice(elite[:5])
            cfg = mutate_cfg(parent, scale=random.uniform(0.08, 0.22))
        else:
            cfg = random_cfg()

        try:
            result = run_once(cfg)
        except Exception as exc:
            print(f"{i:03d} error: {exc}")
            continue

        hc = float(result.get("human_confidence") or 0)
        model = (result.get("evaluation") or {}).get("model_type")
        if model != REQUIRED_MODEL:
            print(f"{i:03d} skipped wrong model {model}")
            continue
        feats = result.get("features") or {}
        flags = (result.get("evaluation") or {}).get("flags") or []
        print(
            f"{i:03d} hc={hc:.4f} events={feats.get('event_count')} "
            f"dur={feats.get('duration_ms')} straight={feats.get('straightness_index')} "
            f"vel={feats.get('velocity_mean')} idle={feats.get('idle_ratio')} flags={flags}"
        )

        elite.append((hc, cfg))
        elite.sort(key=lambda x: x[0], reverse=True)
        elite = elite[:12]

        if hc > best_hc:
            best_hc = hc
            best = result
            with open(r"d:\Cursor\Behavioral Captcha\test\best_result.json", "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)

        if hc >= TARGET:
            print("TARGET_REACHED", hc)
            break

    print("MAX_HUMAN_CONFIDENCE", best_hc)
    if best:
        print(json.dumps({"human_confidence": best_hc, "features": best.get("features"), "cfg": best.get("_cfg")}, indent=2))


if __name__ == "__main__":
    main()
