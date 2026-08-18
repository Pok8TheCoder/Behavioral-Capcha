"""Qmodel v0.2 evasion bot — avoids patched procedural-mimic + dynamic rules."""

from __future__ import annotations

import json
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Any

API_BASE = "http://127.0.0.1:8010"
TARGET = 0.70
MAX_ATTEMPTS = 600
REQUIRED_MODEL = "qmodel_v0.2"


@dataclass
class EvasionConfig:
    max_speed_px_ms: float
    min_speed_px_ms: float
    dwell_ms: tuple[float, float]
    pause_ms: tuple[float, float]
    move_interval_ms: tuple[float, float]
    aim_jitter_px: float
    arc_strength: float
    directness: float  # 0=curvy, 1=near straight
    tremor_px: float
    burst_speed_mult: float


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


def ensure_model() -> None:
    api_post("/api/models/activate", {"model_name": REQUIRED_MODEL})


def ease_out_in(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def move_along(events: list[dict], start, end, t0: float, cfg: EvasionConfig) -> tuple[float, tuple[float, float]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return t0, end

    steps = max(8, int(dist / random.uniform(5.5, 9.0)))
    duration = dist / random.uniform(cfg.min_speed_px_ms, cfg.max_speed_px_ms)
    duration = max(duration, steps * random.uniform(*cfg.move_interval_ms))

    # Perpendicular wobble scaled down by directness (higher straightness).
    nx, ny = (-dy / dist, dx / dist)
    wobble = cfg.arc_strength * (1.0 - cfg.directness)

    for i in range(1, steps + 1):
        raw = i / steps
        eased = ease_out_in(raw)
        base_x = start[0] + dx * eased
        base_y = start[1] + dy * eased
        wave = math.sin(raw * math.pi) * wobble * random.uniform(0.6, 1.2)
        tremor = cfg.tremor_px * random.uniform(-1, 1)
        speed_phase = 1.0 + cfg.burst_speed_mult * math.sin(raw * math.pi * 2.2) * 0.15
        t = t0 + (duration * raw) / max(0.5, speed_phase)
        events.append(
            {
                "x": round(base_x + nx * wave + tremor),
                "y": round(base_y + ny * wave + tremor * 0.7),
                "t": round(t, 2),
                "type": "mousemove",
            }
        )

    return t0 + duration, end


def click_at(events: list[dict], x: float, y: float, t0: float, cfg: EvasionConfig) -> float:
    ox = x + random.uniform(-cfg.aim_jitter_px, cfg.aim_jitter_px)
    oy = y + random.uniform(-cfg.aim_jitter_px, cfg.aim_jitter_px)
    events.append({"x": round(ox), "y": round(oy), "t": round(t0, 2), "type": "mousedown"})
    dwell = random.uniform(*cfg.dwell_ms)
    micro = random.randint(1, 3)
    for i in range(1, micro + 1):
        events.append(
            {
                "x": round(ox + random.uniform(-0.6, 0.6)),
                "y": round(oy + random.uniform(-0.6, 0.6)),
                "t": round(t0 + dwell * (i / micro), 2),
                "type": "mousemove",
            }
        )
    t_up = t0 + dwell
    events.append({"x": round(ox), "y": round(oy), "t": round(t_up, 2), "type": "mouseup"})
    return t_up


def generate_events(challenge: dict[str, Any], cfg: EvasionConfig) -> list[dict]:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    events: list[dict] = []
    t = random.uniform(80, 180)
    pos = (area["x"] + random.uniform(30, 90), area["y"] + random.uniform(30, 90))
    events.append({"x": round(pos[0]), "y": round(pos[1]), "t": round(t, 2), "type": "mousemove"})

    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        t, pos = move_along(events, pos, target, t, cfg)
        t += random.uniform(25, 90)
        t = click_at(events, target[0], target[1], t, cfg)
        pos = (target[0], target[1])
        t += random.uniform(*cfg.pause_ms)

    return events


def evasion_cfg_from_dict(d: dict) -> EvasionConfig:
    return EvasionConfig(
        max_speed_px_ms=d["max_speed_px_ms"],
        min_speed_px_ms=d["min_speed_px_ms"],
        dwell_ms=tuple(d["dwell_ms"]),
        pause_ms=tuple(d["pause_ms"]),
        move_interval_ms=tuple(d["move_interval_ms"]),
        aim_jitter_px=d["aim_jitter_px"],
        arc_strength=d["arc_strength"],
        directness=d["directness"],
        tremor_px=d["tremor_px"],
        burst_speed_mult=d["burst_speed_mult"],
    )


def random_cfg() -> EvasionConfig:
    return EvasionConfig(
        max_speed_px_ms=random.uniform(1.0, 2.4),
        min_speed_px_ms=random.uniform(0.45, 0.95),
        dwell_ms=(random.uniform(55, 85), random.uniform(88, 130)),
        pause_ms=(random.uniform(90, 220), random.uniform(180, 420)),
        move_interval_ms=(random.uniform(10, 16), random.uniform(16, 26)),
        aim_jitter_px=random.uniform(1.5, 5.0),
        arc_strength=random.uniform(4, 28),
        directness=random.uniform(0.55, 0.92),
        tremor_px=random.uniform(0.2, 1.2),
        burst_speed_mult=random.uniform(0.0, 1.5),
    )


def mutate_cfg(base: EvasionConfig) -> EvasionConfig:
    def n(v, lo, hi, scale=0.18):
        return max(lo, min(hi, v * (1 + random.uniform(-scale, scale))))

    return EvasionConfig(
        max_speed_px_ms=n(base.max_speed_px_ms, 0.7, 3.5),
        min_speed_px_ms=n(base.min_speed_px_ms, 0.2, 1.5),
        dwell_ms=(n(base.dwell_ms[0], 40, 120), n(base.dwell_ms[1], 70, 160)),
        pause_ms=(n(base.pause_ms[0], 50, 400), n(base.pause_ms[1], 100, 700)),
        move_interval_ms=base.move_interval_ms,
        aim_jitter_px=n(base.aim_jitter_px, 0.5, 8),
        arc_strength=n(base.arc_strength, 0, 40),
        directness=n(base.directness, 0.35, 0.98),
        tremor_px=n(base.tremor_px, 0, 2.5),
        burst_speed_mult=n(base.burst_speed_mult, 0, 2.5),
    )


def run_once(cfg: EvasionConfig) -> dict[str, Any]:
    challenge = api_post("/api/challenge/new", {"challenge_type": "button_sequence"})
    events = generate_events(challenge, cfg)
    result = api_post(
        "/api/challenge/verify",
        {
            "session_id": challenge["session_id"],
            "events": events,
            "participant": "bot_qmodel_evasion",
        },
    )
    result["_cfg"] = cfg.__dict__
    return result


def main() -> None:
    ensure_model()
    best_hc = -1.0
    best: dict[str, Any] | None = None
    elite: list[tuple[float, EvasionConfig]] = []

    for i in range(1, MAX_ATTEMPTS + 1):
        cfg = mutate_cfg(random.choice(elite)[1]) if elite and random.random() < 0.6 else random_cfg()
        try:
            result = run_once(cfg)
        except Exception as exc:
            print(f"{i:03d} error: {exc}")
            continue

        model = (result.get("evaluation") or {}).get("model_type")
        if model != REQUIRED_MODEL:
            continue

        hc = float(result.get("human_confidence") or 0)
        feats = result.get("features") or {}
        flags = (result.get("evaluation") or {}).get("flags") or []
        heuristic = (result.get("evaluation") or {}).get("heuristic_score") or 0
        print(
            f"{i:03d} hc={hc:.4f} heur={heuristic:.2f} "
            f"straight={feats.get('straightness_index')} jitter={feats.get('jitter_ratio')} "
            f"vel={feats.get('velocity_mean')} dur={feats.get('duration_ms')} "
            f"events={feats.get('event_count')} flags={flags}"
        )

        elite.append((hc, cfg))
        elite.sort(key=lambda x: x[0], reverse=True)
        elite = elite[:15]

        if hc > best_hc:
            best_hc = hc
            best = result
            with open("qmodel_best_result.json", "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)

        if hc >= TARGET:
            print(f"TARGET_REACHED {hc:.4f}")
            break

    print("MAX_HUMAN_CONFIDENCE", best_hc)
    if best:
        print(json.dumps({"human_confidence": best_hc, "features": best.get("features"), "flags": best.get("evaluation", {}).get("flags")}, indent=2))


if __name__ == "__main__":
    main()
