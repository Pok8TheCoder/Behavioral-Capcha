"""Sweep Playwright-like hop families into datasets/bot_families for Ymodel training."""

from __future__ import annotations

import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from challenge_server.generator import generate_challenge
from test.evasion_bot import events_to_csv_rows
from test.sparse_irregular_bot import SparseConfig, generate_events

OUT_DIR = os.path.join(ROOT, "datasets", "bot_families")
SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)

NAMED_FAMILIES: dict[str, SparseConfig] = {
    "sparse_irregular": SparseConfig(name="sparse_irregular"),
    "mid_count": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=16, max_points=42,
        arc_px=(8.0, 22.0), tremor_px=1.2, aim_sigma=2.0, name="mid_count",
    ),
    "high_str_fast": SparseConfig(
        spacing_px=(14.0, 24.0), min_points=8, max_points=20,
        arc_px=(2.0, 8.0), tremor_px=0.7, aim_sigma=1.6,
        s_curve=False, irregular=False, name="high_str_fast",
    ),
    "idle_correct": SparseConfig(
        spacing_px=(8.0, 15.0), min_points=12, max_points=32,
        arc_px=(10.0, 26.0), tremor_px=1.25, aim_sigma=2.4,
        dwell_ms=(80.0, 160.0), pause_ms=(120.0, 280.0), name="idle_correct",
    ),
    "hi_jit": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=14, max_points=38,
        arc_px=(14.0, 32.0), tremor_px=2.8, aim_sigma=2.6, name="hi_jit",
    ),
    "smooth_low_jit": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=16, max_points=40,
        arc_px=(6.0, 14.0), tremor_px=0.4, aim_sigma=1.5,
        s_curve=True, irregular=False, name="smooth_low_jit",
    ),
    "hi_jit_long": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=20, max_points=50,
        arc_px=(16.0, 36.0), tremor_px=2.6, aim_sigma=2.5, name="hi_jit_long",
    ),
    "wobble_mid": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=14, max_points=38,
        arc_px=(18.0, 38.0), tremor_px=1.8, aim_sigma=2.3,
        pause_ms=(60.0, 140.0), name="wobble_mid",
    ),
    "slow_arc": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=16, max_points=40,
        arc_px=(14.0, 30.0), tremor_px=1.4, hop_dt_ms=(12.0, 36.0), name="slow_arc",
    ),
    "dense_slow": SparseConfig(
        spacing_px=(5.5, 10.0), min_points=24, max_points=64,
        arc_px=(8.0, 20.0), tremor_px=0.85, hop_dt_ms=(10.0, 28.0),
        irregular=False, name="dense_slow",
    ),
    "burst_hi_vel": SparseConfig(
        spacing_px=(11.0, 18.0), min_points=10, max_points=24,
        arc_px=(8.0, 18.0), tremor_px=1.0, hop_dt_ms=(3.0, 12.0),
        s_curve=False, name="burst_hi_vel",
    ),
    "low_dyn": SparseConfig(
        spacing_px=(9.0, 16.0), min_points=14, max_points=36,
        arc_px=(8.0, 18.0), tremor_px=0.9, hop_dt_ms=(8.0, 22.0),
        s_curve=False, name="low_dyn",
    ),
}


def _pair(rng: random.Random, lo: float, hi: float) -> tuple[float, float]:
    a = rng.uniform(lo, hi)
    b = rng.uniform(lo, hi)
    return (min(a, b), max(a, b))


def random_family(rng: random.Random, idx: int) -> SparseConfig:
    min_pts = rng.randint(8, 28)
    max_pts = rng.randint(max(min_pts + 4, 18), 72)
    return SparseConfig(
        spacing_px=_pair(rng, 5.0, 22.0),
        min_points=min_pts,
        max_points=max_pts,
        arc_px=_pair(rng, 2.0, 42.0),
        tremor_px=rng.uniform(0.25, 3.4),
        aim_sigma=rng.uniform(1.2, 2.8),
        dwell_ms=_pair(rng, 30.0, 180.0),
        pause_ms=_pair(rng, 30.0, 280.0),
        hop_dt_ms=_pair(rng, 3.0, 40.0),
        s_curve=rng.random() < 0.65,
        irregular=rng.random() < 0.75,
        name=f"sweep_{idx:03d}",
    )


def export_families(
    *,
    named_episodes: int = 14,
    random_families: int = 90,
    random_episodes: int = 3,
    seed: int = 1501,
) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(seed)
    out_path = os.path.join(OUT_DIR, "ymodel_v01_families.csv")
    cfg_path = os.path.join(OUT_DIR, "ymodel_v01_families_cfg.json")
    all_rows: list[dict[str, str]] = []
    t_cursor = 0.0
    episodes = 0
    catalog: list[str] = []

    jobs: list[SparseConfig] = []
    for cfg in NAMED_FAMILIES.values():
        jobs.extend([cfg] * named_episodes)
        catalog.append(cfg.name)
    for i in range(random_families):
        cfg = random_family(rng, i)
        jobs.extend([cfg] * random_episodes)
        catalog.append(cfg.name)

    rng.shuffle(jobs)
    for cfg in jobs:
        challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
        events = generate_events(challenge, cfg)
        all_rows.extend(events_to_csv_rows(events, t_cursor))
        last_t = max(float(e["t"]) for e in events)
        t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)
        episodes += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
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
        for row in all_rows:
            writer.writerow(row)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "named_families": list(NAMED_FAMILIES),
                "named_episodes": named_episodes,
                "random_families": random_families,
                "random_episodes": random_episodes,
                "episodes": episodes,
                "rows": len(all_rows),
                "catalog": catalog,
            },
            f,
            indent=2,
        )
    return {"csv_path": out_path, "episodes": episodes, "rows": len(all_rows)}


def main() -> None:
    result = export_families()
    print(json.dumps(result, indent=2), flush=True)
    print("FAMILY_EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
