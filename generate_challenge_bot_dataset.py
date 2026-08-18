"""
Generate human-like bot telemetry by solving randomized behavioral challenges.

Output CSVs match the human recorder format and are sized to ~2x the cleaned
human 3s window count (see analyze_human_dataset.py). Bots use simulated
Fitts/minimum-jerk motor control — not human replay.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT_DIR, "datasets", "bot_challenge")
SUMMARY_PATH = os.path.join(OUT_DIR, "challenge_bot_summary.json")
HUMAN_SUMMARY = os.path.join(ROOT_DIR, "datasets", "human", "merged_analysis_summary.json")

SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
EPISODES_PER_FILE = 350
TARGET_MULTIPLIER = 2.0
RANDOM_STATE = 20260816

sys.path.insert(0, ROOT_DIR)

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from bots.challenge_solver import solve_challenge
from bots.simulated_humanoid import SimulatedHumanoidBot
from train_isolation_forest import extract_active_windows_from_csv


def load_target_windows() -> int:
    if os.path.exists(HUMAN_SUMMARY):
        with open(HUMAN_SUMMARY, "r", encoding="utf-8") as f:
            doc = json.load(f)
        human_windows = int(doc.get("merged_virtual_windows", 0))
        if human_windows > 0:
            return int(human_windows * TARGET_MULTIPLIER)
    return 41_080


def challenge_entry_point(challenge: Dict[str, Any]) -> Tuple[float, float]:
    ctype = challenge["challenge_type"]
    spec = challenge["spec"]

    if ctype in ("shape_trace", "zigzag_trace", "spiral_trace"):
        return float(spec["guide_points"][0][0]), float(spec["guide_points"][0][1])

    if ctype == "connect_dots":
        dots = {d["id"]: d for d in spec["dots"]}
        first = dots[spec["sequence"][0]]
        return float(first["x"]), float(first["y"])

    if ctype in ("maze_navigate", "hazard_run", "path_gates"):
        return float(spec["start"][0]), float(spec["start"][1])

    if ctype == "button_sequence":
        buttons = {b["id"]: b for b in spec["buttons"]}
        btn = buttons[spec["sequence"][0]]
        return float(btn["x"]), float(btn["y"])

    if ctype == "target_hunt":
        return float(spec["targets"][0]["x"]), float(spec["targets"][0]["y"])

    if ctype == "pin_entry":
        return float(spec["keys"][0]["x"]), float(spec["keys"][0]["y"])

    return 480.0, 320.0


def generate_episode(rng: random.Random) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bot = SimulatedHumanoidBot.random(rng)
    start_x = rng.uniform(100.0, SCREEN_W - 100.0)
    start_y = rng.uniform(80.0, SCREEN_H - 120.0)
    bot.record_raw(start_x, start_y)
    bot.idle_jitter(rng.uniform(150.0, 520.0), start_x, start_y)

    challenge_type = rng.choice(CHALLENGE_TYPES)
    seed = rng.randint(1, 2_000_000_000)
    challenge = generate_challenge(seed=seed, challenge_type=challenge_type)

    entry_x, entry_y = challenge_entry_point(challenge)
    last_x = bot._last_x or start_x
    last_y = bot._last_y or start_y
    bot.move_to(last_x, last_y, entry_x, entry_y)
    bot.advance(rng.uniform(40.0, 180.0))

    solve_challenge(bot, challenge, style="humanoid", fast=False)
    bot.advance(rng.uniform(120.0, 420.0))

    meta = {
        "challenge_type": challenge_type,
        "seed": seed,
        "profile": bot.profile.name,
        "events": len(bot.events),
        "duration_ms": round(bot._t_ms, 2),
    }
    return bot.events, meta


def events_to_csv_rows(events: List[Dict[str, Any]], t_offset_ms: float) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for event in events:
        x = float(event["x"])
        y = float(event["y"])
        rows.append({
            "rel_time_ms": f"{t_offset_ms + float(event['t']):.3f}",
            "x_px": f"{x:.1f}",
            "y_px": f"{y:.1f}",
            "x_norm": f"{x / SCREEN_W:.5f}",
            "y_norm": f"{y / SCREEN_H:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": "0",
            "scroll_dy": "0",
            "event_type": str(event.get("event_type", "mouse_move")),
            "target_item": "challenge_bot",
        })
    return rows


def write_csv(path: str, rows: List[Dict[str, str]]) -> int:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({SCREEN_W}x{SCREEN_H})\n")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
                "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
                "event_type", "target_item",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def count_windows(csv_path: str) -> int:
    samples, _ = extract_active_windows_from_csv(csv_path, max_windows=10_000_000)
    return len(samples)


def main() -> None:
    rng = random.Random(RANDOM_STATE)
    os.makedirs(OUT_DIR, exist_ok=True)

    target_windows = load_target_windows()
    total_windows = 0
    total_episodes = 0
    total_rows = 0
    file_idx = 0
    type_counts: Dict[str, int] = {t: 0 for t in CHALLENGE_TYPES}
    profile_counts: Dict[str, int] = {}
    episode_samples: List[Dict[str, Any]] = []
    t_cursor = 0.0
    started = time.time()

    print("=" * 72)
    print("CHALLENGE BOT DATASET GENERATOR (human-like motor simulation)")
    print("=" * 72)
    print(f"[Plan] Target 3s windows: {target_windows:,} ({TARGET_MULTIPLIER}x human)")
    print(f"[Plan] Challenge types: {len(CHALLENGE_TYPES)}")
    print(f"[Plan] Output: {OUT_DIR}")

    while total_windows < target_windows:
        file_idx += 1
        batch_rows: List[Dict[str, str]] = []
        batch_meta: List[Dict[str, Any]] = []

        for _ in range(EPISODES_PER_FILE):
            events, meta = generate_episode(rng)
            batch_rows.extend(events_to_csv_rows(events, t_cursor))
            batch_meta.append(meta)
            total_episodes += 1
            type_counts[meta["challenge_type"]] += 1
            profile_counts[meta["profile"]] = profile_counts.get(meta["profile"], 0) + 1
            if len(episode_samples) < 30:
                episode_samples.append(meta)

            last_t = max(float(e["t"]) for e in events)
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)

            if total_windows >= target_windows:
                break

        out_path = os.path.join(OUT_DIR, f"challenge_bot_{file_idx:03d}.csv")
        row_count = write_csv(out_path, batch_rows)
        total_rows += row_count
        file_windows = count_windows(out_path)
        total_windows += file_windows

        elapsed = time.time() - started
        print(
            f"[Write] {os.path.basename(out_path)}: "
            f"{len(batch_meta):,} episodes, {row_count:,} rows, "
            f"{file_windows:,} windows | total {total_windows:,}/{target_windows:,} "
            f"({elapsed:.1f}s)"
        )

        if file_idx > 500:
            print("[Warn] File limit reached; stopping early.")
            break

    summary = {
        "target_windows": target_windows,
        "generated_windows": total_windows,
        "target_multiplier": TARGET_MULTIPLIER,
        "generated_episodes": total_episodes,
        "generated_rows": total_rows,
        "output_files": file_idx,
        "challenge_type_counts": type_counts,
        "profile_counts": profile_counts,
        "screen": {"width": SCREEN_W, "height": SCREEN_H},
        "afk_gap_ms_range": AFK_GAP_MS_RANGE,
        "sample_episodes": episode_samples,
        "generation_seconds": round(time.time() - started, 2),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(f"[Done] Episodes: {total_episodes:,}")
    print(f"[Done] Rows: {total_rows:,}")
    print(f"[Done] Valid 3s windows: {total_windows:,}")
    print(f"[Done] Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
