"""
Generate challenge bot telemetry from ALL bot tiers (linear, bezier, naive,
fast humanoid, simulated humanoid) on randomized challenges.

Output: datasets/bot_challenge_tiers/*.csv
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT_DIR, "datasets", "bot_challenge_tiers")
SUMMARY_PATH = os.path.join(OUT_DIR, "multi_tier_bot_summary.json")
HUMAN_SUMMARY = os.path.join(ROOT_DIR, "datasets", "human", "merged_analysis_summary.json")

SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
EPISODES_PER_STYLE = 200
EPISODES_SIMULATED = 80
MAX_EVENTS_PER_EPISODE = 8_000

# Stamp profiles for fast bots (spread timestamps without wall-clock sleep)
STAMP_PROFILES: Dict[str, Tuple[float, float]] = {
    "linear_fast": (0.8, 3.5),
    "bezier_fast": (1.0, 4.0),
    "humanoid_fast": (1.2, 5.0),
    "naive": (2.0, 8.0),
    "linear_timed": (8.0, 18.0),
    "bezier_timed": (7.0, 16.0),
    "humanoid_simulated": (1.0, 4.0),
}
RANDOM_STATE = 20260816

sys.path.insert(0, ROOT_DIR)

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from bots.challenge_bot_factory import CHALLENGE_BOT_STYLES, prelude_bot
from bots.challenge_solver import solve_challenge
from bots.telemetry_export import ensure_export_timestamps
from generate_challenge_bot_dataset import challenge_entry_point, write_csv
from train_isolation_forest import extract_active_windows_from_csv


def generate_episode_for_style(
    rng: random.Random,
    style_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    factory = CHALLENGE_BOT_STYLES[style_id]
    for _ in range(8):
        bot, solve_style, fast = factory(rng)
        prelude_bot(bot, rng, SCREEN_W, SCREEN_H)

        challenge_type = rng.choice(CHALLENGE_TYPES)
        seed = rng.randint(1, 2_000_000_000)
        challenge = generate_challenge(seed=seed, challenge_type=challenge_type)

        entry_x, entry_y = challenge_entry_point(challenge)
        last_x = (
            bot._last_x if hasattr(bot, "_last_x") and bot._last_x is not None
            else float(bot.events[-1]["x"])
        )
        last_y = (
            bot._last_y if hasattr(bot, "_last_y") and bot._last_y is not None
            else float(bot.events[-1]["y"])
        )

        if getattr(bot, "simulated", False):
            bot.move_to(last_x, last_y, entry_x, entry_y)
            bot.advance(rng.uniform(40.0, 180.0))
        elif not fast:
            bot.move_to(last_x, last_y, entry_x, entry_y)

        solve_challenge(bot, challenge, style=solve_style, fast=fast)
        step_lo, step_hi = STAMP_PROFILES.get(style_id, (1.0, 6.0))
        events = ensure_export_timestamps(
            list(bot.events), rng, min_step_ms=step_lo, max_step_ms=step_hi,
        )
        if len(events) <= MAX_EVENTS_PER_EPISODE:
            break
    else:
        events = ensure_export_timestamps(
            list(bot.events), rng,
            min_step_ms=STAMP_PROFILES.get(style_id, (1.0, 6.0))[0],
            max_step_ms=STAMP_PROFILES.get(style_id, (1.0, 6.0))[1],
        )

    duration_ms = max(float(e["t"]) for e in events) if events else 0.0
    meta = {
        "bot_style": style_id,
        "challenge_type": challenge_type,
        "seed": seed,
        "fast": fast,
        "events": len(events),
        "duration_ms": round(duration_ms, 2),
    }
    return events, meta


def generate_episode(rng: random.Random) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    style_id = rng.choice(list(CHALLENGE_BOT_STYLES.keys()))
    return generate_episode_for_style(rng, style_id)


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
            "target_item": "challenge_bot_tier",
        })
    return rows


def count_windows(csv_path: str) -> int:
    samples, _ = extract_active_windows_from_csv(csv_path, max_windows=10_000_000)
    return len(samples)


def main() -> None:
    rng = random.Random(RANDOM_STATE)
    os.makedirs(OUT_DIR, exist_ok=True)

    total_windows = 0
    total_episodes = 0
    total_rows = 0
    style_counts: Dict[str, int] = {k: 0 for k in CHALLENGE_BOT_STYLES}
    challenge_counts: Dict[str, int] = {t: 0 for t in CHALLENGE_TYPES}
    windows_per_style: Dict[str, int] = {}
    episode_samples: List[Dict[str, Any]] = []
    started = time.time()

    print("=" * 72)
    print("MULTI-TIER CHALLENGE BOT DATASET GENERATOR")
    print("=" * 72)
    print(f"[Plan] Bot styles: {list(CHALLENGE_BOT_STYLES.keys())}")
    print(f"[Plan] Episodes per style: {EPISODES_PER_STYLE}")
    print(f"[Plan] Output: {OUT_DIR}")

    for style_id in CHALLENGE_BOT_STYLES:
        episode_target = EPISODES_SIMULATED if style_id == "humanoid_simulated" else EPISODES_PER_STYLE
        batch_rows: List[Dict[str, str]] = []
        t_cursor = 0.0

        for _ in range(episode_target):
            events, meta = generate_episode_for_style(rng, style_id)
            batch_rows.extend(events_to_csv_rows(events, t_cursor))
            total_episodes += 1
            style_counts[style_id] += 1
            challenge_counts[meta["challenge_type"]] += 1
            if len(episode_samples) < 40:
                episode_samples.append(meta)
            last_t = max(float(e["t"]) for e in events)
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)

        out_path = os.path.join(OUT_DIR, f"tier_{style_id}.csv")
        row_count = write_csv(out_path, batch_rows)
        total_rows += row_count
        file_windows = count_windows(out_path)
        windows_per_style[style_id] = file_windows
        total_windows += file_windows
        print(
            f"[Write] {os.path.basename(out_path)}: "
            f"{episode_target} episodes, {file_windows:,} windows"
        )

    summary = {
        "episodes_per_style": EPISODES_PER_STYLE,
        "episodes_simulated": EPISODES_SIMULATED,
        "generated_windows": total_windows,
        "generated_episodes": total_episodes,
        "generated_rows": total_rows,
        "windows_per_style": windows_per_style,
        "bot_style_counts": style_counts,
        "challenge_type_counts": challenge_counts,
        "bot_styles": list(CHALLENGE_BOT_STYLES.keys()),
        "output_dir": OUT_DIR,
        "generation_seconds": round(time.time() - started, 2),
        "sample_episodes": episode_samples,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(f"[Done] Episodes: {total_episodes:,}, windows: {total_windows:,}")
    print(f"[Done] Style counts: {style_counts}")


if __name__ == "__main__":
    main()
