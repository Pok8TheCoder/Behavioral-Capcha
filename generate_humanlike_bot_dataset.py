"""
Generate adversarial human-like bot telemetry without replaying human data.

The output uses the same CSV shape as the recorder and is intended for the
next supervised Random Forest training pass. Each generated session is a
short active movement episode separated by an AFK-sized gap, which makes it
align with the 5s AFK split and 3s window preprocessing used by
train_isolation_forest.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


ROOT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT_DIR, "datasets", "bot_humanlike")
IFOREST_METADATA = os.path.join(ROOT_DIR, "backend", "behavioral_iforest_metadata.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "humanlike_bot_summary.json")

SCREEN_W = 1920
SCREEN_H = 1080
WINDOW_MS = 3_000.0
EPISODE_MS_RANGE = (3_050.0, 3_450.0)
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
RANDOM_STATE = 20260812


@dataclass
class BotProfile:
    name: str
    curvature_range: Tuple[float, float]
    tremor_px_range: Tuple[float, float]
    pause_probability: float
    overshoot_probability: float
    click_probability: float
    noise_px_range: Tuple[float, float]


PROFILES = [
    BotProfile("minimum_jerk_precise", (0.03, 0.10), (0.12, 0.45), 0.16, 0.18, 0.62, (0.05, 0.22)),
    BotProfile("curved_submovement", (0.08, 0.20), (0.25, 0.75), 0.24, 0.38, 0.72, (0.12, 0.42)),
    BotProfile("hesitant_trackpad", (0.04, 0.16), (0.40, 1.10), 0.38, 0.24, 0.54, (0.18, 0.62)),
    BotProfile("fast_confident_mouse", (0.02, 0.08), (0.08, 0.35), 0.08, 0.14, 0.66, (0.04, 0.18)),
    BotProfile("imperfect_targeting", (0.06, 0.18), (0.20, 0.85), 0.28, 0.55, 0.78, (0.10, 0.50)),
]


def minimum_jerk(tau: float) -> float:
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_target_episode_count() -> int:
    if not os.path.exists(IFOREST_METADATA):
        return 11_500
    with open(IFOREST_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return int(metadata.get("human_train_windows", 0) + metadata.get("human_test_windows", 0)) or 11_500


def choose_points() -> Tuple[float, float, float, float]:
    start_x = random.uniform(80.0, SCREEN_W - 80.0)
    start_y = random.uniform(80.0, SCREEN_H - 80.0)

    angle = random.uniform(0.0, math.tau)
    distance = random.triangular(180.0, 980.0, 430.0)
    target_x = clamp(start_x + math.cos(angle) * distance, 50.0, SCREEN_W - 50.0)
    target_y = clamp(start_y + math.sin(angle) * distance, 50.0, SCREEN_H - 50.0)

    if math.hypot(target_x - start_x, target_y - start_y) < 120.0:
        return choose_points()
    return start_x, start_y, target_x, target_y


def add_event(rows: List[Dict[str, object]], t: float, x: float, y: float, event_type: str, target_item: str = "none") -> None:
    x = clamp(x, 0.0, float(SCREEN_W))
    y = clamp(y, 0.0, float(SCREEN_H))
    rows.append(
        {
            "rel_time_ms": round(t, 3),
            "x_px": round(x, 1),
            "y_px": round(y, 1),
            "x_norm": f"{x / SCREEN_W:.5f}",
            "y_norm": f"{y / SCREEN_H:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": 0,
            "scroll_dy": 0,
            "event_type": event_type,
            "target_item": target_item,
        }
    )


def generate_episode(t0: float, profile: BotProfile) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sx, sy, tx, ty = choose_points()
    distance = math.hypot(tx - sx, ty - sy)
    duration_ms = random.uniform(*EPISODE_MS_RANGE)

    target_width = random.uniform(28.0, 80.0)
    fitts_id = math.log2(1.0 + distance / target_width)
    primary_ratio = random.uniform(0.78, 0.88)
    correction_ratio = 1.0 - primary_ratio

    sample_rate = random.triangular(58.0, 165.0, 96.0)
    dt_base = 1000.0 / sample_rate
    steps = max(80, int(duration_ms / dt_base))

    dx = tx - sx
    dy = ty - sy
    ux = dx / (distance + 1e-9)
    uy = dy / (distance + 1e-9)
    px = -uy
    py = ux

    curve_sign = random.choice([-1.0, 1.0])
    curvature = random.uniform(*profile.curvature_range) * distance * curve_sign
    tremor_amp = random.uniform(*profile.tremor_px_range)
    noise_px = random.uniform(*profile.noise_px_range)
    tremor_f1 = random.uniform(7.0, 10.5)
    tremor_f2 = random.uniform(10.5, 13.0)
    phase1 = random.uniform(0.0, math.tau)
    phase2 = random.uniform(0.0, math.tau)
    curve_style = random.choice(["arc", "s_curve", "late_hook"])

    endpoint_x = tx
    endpoint_y = ty
    overshot = random.random() < profile.overshoot_probability
    if overshot:
        overshoot_px = random.uniform(5.0, 28.0)
        endpoint_x = tx + ux * overshoot_px + px * random.uniform(-4.0, 4.0)
        endpoint_y = ty + uy * overshoot_px + py * random.uniform(-4.0, 4.0)

    pause_start = random.uniform(0.35, 0.78) if random.random() < profile.pause_probability else None
    pause_ms = random.uniform(35.0, 160.0) if pause_start else 0.0

    last_x = sx
    last_y = sy
    for i in range(steps):
        raw_tau = i / max(1, steps - 1)
        effective_tau = raw_tau
        time_offset = raw_tau * duration_ms

        if pause_start and pause_start <= raw_tau <= min(0.96, pause_start + pause_ms / duration_ms):
            effective_tau = pause_start

        if effective_tau < primary_ratio:
            local_tau = effective_tau / primary_ratio
            move = minimum_jerk(local_tau)
            base_x = sx + (endpoint_x - sx) * move
            base_y = sy + (endpoint_y - sy) * move
        else:
            local_tau = (effective_tau - primary_ratio) / max(0.001, correction_ratio)
            move = minimum_jerk(local_tau)
            base_x = endpoint_x + (tx - endpoint_x) * move
            base_y = endpoint_y + (ty - endpoint_y) * move

        if curve_style == "arc":
            curve_offset = math.sin(math.pi * effective_tau) * curvature
        elif curve_style == "s_curve":
            curve_offset = math.sin(2.0 * math.pi * effective_tau) * curvature * 0.55
        else:
            curve_offset = math.sin(math.pi * effective_tau) ** 1.8 * curvature
            if effective_tau > 0.72:
                curve_offset *= 1.0 - (effective_tau - 0.72) / 0.28

        seconds = time_offset / 1000.0
        tremor_x = tremor_amp * math.sin(math.tau * tremor_f1 * seconds + phase1)
        tremor_y = tremor_amp * math.cos(math.tau * tremor_f2 * seconds + phase2)
        random_walk_x = random.gauss(0.0, noise_px)
        random_walk_y = random.gauss(0.0, noise_px)

        x = base_x + px * curve_offset + tremor_x + random_walk_x
        y = base_y + py * curve_offset + tremor_y + random_walk_y

        if i > 0 and random.random() < 0.015:
            x = last_x + random.uniform(-0.3, 0.3)
            y = last_y + random.uniform(-0.3, 0.3)

        timestamp = t0 + time_offset + random.gauss(0.0, 1.8)
        add_event(rows, timestamp, x, y, "mouse_move")
        last_x, last_y = x, y

    if random.random() < profile.click_probability:
        down_t = t0 + duration_ms + random.uniform(40.0, 220.0)
        dwell_ms = clamp(random.gauss(92.0, 28.0), 42.0, 210.0)
        click_x = tx + random.gauss(0.0, 1.8)
        click_y = ty + random.gauss(0.0, 1.8)
        add_event(rows, down_t, click_x, click_y, "mouse_click_down", "synthetic_target")
        add_event(rows, down_t + dwell_ms, click_x + random.gauss(0.0, 0.3), click_y + random.gauss(0.0, 0.3), "mouse_click_up", "synthetic_target")

    rows.sort(key=lambda row: float(row["rel_time_ms"]))
    summary = {
        "profile": profile.name,
        "distance_px": round(distance, 2),
        "duration_ms": round(duration_ms, 2),
        "events": len(rows),
        "fitts_id": round(fitts_id, 3),
        "overshot": overshot,
    }
    return rows, summary


def write_csv(path: str, rows: Iterable[Dict[str, object]]) -> int:
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({SCREEN_W}x{SCREEN_H})\n")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rel_time_ms",
                "x_px",
                "y_px",
                "x_norm",
                "y_norm",
                "scroll_x",
                "scroll_y",
                "scroll_dx",
                "scroll_dy",
                "event_type",
                "target_item",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    random.seed(RANDOM_STATE)
    os.makedirs(OUT_DIR, exist_ok=True)

    target_episodes = load_target_episode_count()
    files = 6
    episodes_per_file = math.ceil(target_episodes / files)
    generated_episodes = 0
    generated_rows = 0
    profile_counts = {profile.name: 0 for profile in PROFILES}
    summaries: List[Dict[str, object]] = []

    print("=" * 72)
    print("HUMAN-LIKE BOT DATASET GENERATOR")
    print("=" * 72)
    print(f"[Plan] Target episodes/windows: {target_episodes:,}")
    print(f"[Plan] Output directory: {OUT_DIR}")

    for file_idx in range(files):
        rows: List[Dict[str, object]] = []
        t = 0.0
        file_episode_count = min(episodes_per_file, target_episodes - generated_episodes)
        if file_episode_count <= 0:
            break

        for _ in range(file_episode_count):
            profile = random.choice(PROFILES)
            episode_rows, summary = generate_episode(t, profile)
            rows.extend(episode_rows)
            summaries.append(summary)
            profile_counts[profile.name] += 1
            generated_episodes += 1
            t = max(float(row["rel_time_ms"]) for row in episode_rows) + random.uniform(*AFK_GAP_MS_RANGE)

        out_path = os.path.join(OUT_DIR, f"humanlike_bot_{file_idx + 1:02d}.csv")
        row_count = write_csv(out_path, rows)
        generated_rows += row_count
        print(f"[Write] {os.path.basename(out_path)}: {file_episode_count:,} episodes, {row_count:,} rows")

    summary_doc = {
        "target_episodes": target_episodes,
        "generated_episodes": generated_episodes,
        "generated_rows": generated_rows,
        "expected_valid_3s_windows": generated_episodes,
        "window_ms": WINDOW_MS,
        "afk_gap_ms_range": AFK_GAP_MS_RANGE,
        "screen": {"width": SCREEN_W, "height": SCREEN_H},
        "profile_counts": profile_counts,
        "profiles": [profile.__dict__ for profile in PROFILES],
        "sample_episode_summaries": summaries[:25],
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    print(f"[Done] Generated {generated_episodes:,} human-like bot episodes.")
    print(f"[Done] Expected valid 3s windows: {generated_episodes:,}")
    print(f"[Done] Rows written: {generated_rows:,}")
    print(f"[Done] Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
