"""Ymodel V0.2 family dataset: live hop families, kinematic grids, Fitts, all challenge types."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from test.evasion_bot import events_to_csv_rows, generate_events as fitts_generate
from test.fast_direct_bot import FAMILY_PRESETS as DART_PRESETS
from test.fast_direct_bot import generate_events as dart_generate
from test.fitts_motor import FittsMotorConfig, default_motor, mutate_motor
from test.playwright_dense_family_bot import FAMILY_NAMES as DENSE_NAMES
from test.playwright_dense_family_bot import cfg_from_preset as dense_cfg
from test.playwright_dense_family_bot import generate_events as dense_generate
from test.playwright_hop_bot import HopConfig, generate_events as hop_generate
from test.sparse_irregular_bot import SparseConfig, generate_events as hop_sparse_generate

OUT_DIR = os.path.join(ROOT, "datasets", "bot_families")
SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
CSV_FIELDS = [
    "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
    "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
    "event_type", "target_item",
]

NAMED_HOP_FAMILIES: dict[str, SparseConfig] = {
    "pause_heavy": SparseConfig(
        spacing_px=(8.0, 15.0), min_points=12, max_points=30,
        arc_px=(8.0, 20.0), tremor_px=1.2, aim_sigma=2.0,
        dwell_ms=(90.0, 160.0), pause_ms=(120.0, 260.0), name="pause_heavy",
    ),
    "wobble_mid": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=14, max_points=38,
        arc_px=(18.0, 38.0), tremor_px=1.8, aim_sigma=2.3,
        pause_ms=(60.0, 140.0), name="wobble_mid",
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
    "hi_jit_long": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=20, max_points=50,
        arc_px=(16.0, 36.0), tremor_px=2.6, aim_sigma=2.5, name="hi_jit_long",
    ),
    "smooth_low_jit": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=16, max_points=40,
        arc_px=(6.0, 14.0), tremor_px=0.4, aim_sigma=1.5,
        irregular=False, name="smooth_low_jit",
    ),
    "high_str_fast": SparseConfig(
        spacing_px=(14.0, 24.0), min_points=8, max_points=20,
        arc_px=(2.0, 8.0), tremor_px=0.7, aim_sigma=1.6,
        s_curve=False, irregular=False, name="high_str_fast",
    ),
    "mid_count": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=16, max_points=42,
        arc_px=(8.0, 22.0), tremor_px=1.2, aim_sigma=2.0, name="mid_count",
    ),
    "sparse_irregular": SparseConfig(name="sparse_irregular"),
    "tight_arc": SparseConfig(
        spacing_px=(9.0, 16.0), min_points=12, max_points=32,
        arc_px=(3.0, 10.0), tremor_px=0.7, aim_sigma=1.6,
        s_curve=False, name="tight_arc",
    ),
    "micro_tremor": SparseConfig(
        spacing_px=(7.0, 12.0), min_points=18, max_points=44,
        arc_px=(10.0, 22.0), tremor_px=0.35, aim_sigma=1.5,
        irregular=False, name="micro_tremor",
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
    "wide_arc": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=16, max_points=42,
        arc_px=(18.0, 40.0), tremor_px=1.4, aim_sigma=2.2, name="wide_arc",
    ),
}


FITTS_PRESETS: dict[str, FittsMotorConfig] = {
    "fitts_default": default_motor(),
    "fitts_fast": FittsMotorConfig(
        fitts_a=0.07, fitts_b=0.11, sample_hz=(90.0, 130.0),
        pause_ms=(80.0, 180.0), dwell_ms=(40.0, 90.0), wander_strength=2.4,
        pause_probability=0.08, overshoot_probability=0.12,
    ),
    "fitts_hesitant": FittsMotorConfig(
        fitts_a=0.16, fitts_b=0.22, sample_hz=(55.0, 80.0),
        pause_ms=(280.0, 620.0), dwell_ms=(120.0, 240.0), wander_strength=6.0,
        pause_probability=0.38, hesitation_ms=(40.0, 140.0),
    ),
    "fitts_tremor": FittsMotorConfig(
        tremor_scale=1.35, wander_strength=7.5, curvature_scale=0.22,
        sample_hz=(70.0, 110.0),
    ),
    "fitts_curved": FittsMotorConfig(
        curvature_scale=0.28, tremor_scale=0.55, overshoot_probability=0.42,
    ),
    "fitts_pause_heavy": FittsMotorConfig(
        dwell_ms=(90.0, 180.0), pause_ms=(140.0, 320.0),
        pre_click_delay_ms=(80.0, 200.0), pause_probability=0.32,
        wander_strength=3.5,
    ),
    "fitts_low_wander": FittsMotorConfig(
        wander_strength=1.6, tremor_scale=0.35, curvature_scale=0.07,
        pause_probability=0.06,
    ),
    "fitts_twitchy": FittsMotorConfig(
        sample_hz=(100.0, 150.0), tremor_scale=1.1, wander_strength=5.0,
        fitts_a=0.08, fitts_b=0.13, dwell_ms=(35.0, 80.0),
    ),
}


def _append_episode(all_rows: list, t_cursor: float, events: list[dict], label: str) -> float:
    if not events:
        return t_cursor
    all_rows.extend(events_to_csv_rows(events, t_cursor, target_item=label))
    last_t = max(float(e["t"]) for e in events)
    return t_cursor + last_t + random.uniform(*AFK_GAP_MS_RANGE)


def named_hop_jobs(episodes: int) -> list[tuple[str, SparseConfig]]:
    jobs = []
    for name, cfg in NAMED_HOP_FAMILIES.items():
        jobs.extend([(name, cfg)] * episodes)
    return jobs


def grid_jobs(episodes: int) -> list[tuple[str, SparseConfig]]:
    jobs: list[tuple[str, SparseConfig]] = []
    for dwell in ((80.0, 110.0), (110.0, 150.0), (150.0, 190.0)):
        for pause in ((80.0, 140.0), (140.0, 220.0), (220.0, 300.0)):
            for hop in ((4.0, 14.0), (10.0, 24.0), (16.0, 40.0)):
                name = f"grid_pause_d{int(dwell[0])}_p{int(pause[0])}_h{int(hop[0])}"
                cfg = SparseConfig(
                    spacing_px=(8.0, 15.0), min_points=12, max_points=34,
                    arc_px=(8.0, 22.0), tremor_px=1.2, aim_sigma=2.0,
                    dwell_ms=dwell, pause_ms=pause, hop_dt_ms=hop, name=name,
                )
                jobs.extend([(name, cfg)] * episodes)
    for arc in ((6.0, 14.0), (14.0, 26.0), (26.0, 42.0)):
        for tremor in (1.4, 1.9, 2.4):
            for pts in ((12, 28), (16, 40), (22, 52)):
                name = f"grid_wobble_a{int(arc[0])}_t{tremor:.1f}_n{pts[1]}"
                cfg = SparseConfig(
                    spacing_px=(8.0, 14.0), min_points=pts[0], max_points=pts[1],
                    arc_px=arc, tremor_px=tremor, aim_sigma=2.2, name=name,
                )
                jobs.extend([(name, cfg)] * episodes)
    for tremor in (0.35, 1.1, 2.0, 2.9):
        for pts in ((10, 26), (16, 42), (24, 60)):
            name = f"grid_jit_t{tremor:.2f}_n{pts[1]}"
            cfg = SparseConfig(
                spacing_px=(7.0, 14.0), min_points=pts[0], max_points=pts[1],
                arc_px=(10.0, 28.0), tremor_px=tremor, aim_sigma=2.0, name=name,
            )
            jobs.extend([(name, cfg)] * episodes)
    for hop in ((3.0, 10.0), (8.0, 20.0), (16.0, 36.0)):
        for pts in ((8, 20), (14, 36), (22, 55), (30, 70)):
            name = f"grid_speed_h{int(hop[0])}_n{pts[1]}"
            cfg = SparseConfig(
                spacing_px=(6.0, 16.0), min_points=pts[0], max_points=pts[1],
                arc_px=(8.0, 24.0), tremor_px=1.1, hop_dt_ms=hop, name=name,
            )
            jobs.extend([(name, cfg)] * episodes)
    return jobs


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({SCREEN_W}x{SCREEN_H})\n")
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_jobs(
    rng: random.Random,
    jobs: list[tuple[str, object]],
    emit: Callable,
    challenge_type: str = "button_sequence",
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    t_cursor = 0.0
    episodes = 0
    rng.shuffle(jobs)
    for label, cfg in jobs:
        challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type=challenge_type)
        events = emit(challenge, cfg)
        t_cursor = _append_episode(rows, t_cursor, events, label)
        episodes += 1
        if episodes % 200 == 0:
            print(f"  ... {episodes} episodes, {len(rows):,} rows", flush=True)
    return rows, episodes


def export_v02(seed: int = 202) -> dict:
    rng = random.Random(seed)
    summary: dict = {"seed": seed, "files": []}

    print("[V0.2] Named hop families...", flush=True)
    named_rows, n_ep = run_jobs(rng, named_hop_jobs(50), hop_sparse_generate)
    path = os.path.join(OUT_DIR, "ymodel_v02_named_hops.csv")
    write_csv(path, named_rows)
    summary["files"].append({"path": path, "episodes": n_ep, "rows": len(named_rows)})
    print(f"  named hops: {n_ep} episodes / {len(named_rows):,} rows", flush=True)

    print("[V0.2] Kinematic grids...", flush=True)
    grid_rows, n_ep = run_jobs(rng, grid_jobs(4), hop_sparse_generate)
    path = os.path.join(OUT_DIR, "ymodel_v02_grids.csv")
    write_csv(path, grid_rows)
    summary["files"].append({"path": path, "episodes": n_ep, "rows": len(grid_rows)})
    print(f"  grids: {n_ep} episodes / {len(grid_rows):,} rows", flush=True)

    print("[V0.2] Dense / dart / playwright hop...", flush=True)
    extra_rows: list[dict[str, str]] = []
    t_cursor = 0.0
    extra_ep = 0
    for name in DENSE_NAMES:
        cfg = dense_cfg(name)
        for _ in range(16):
            challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
            t_cursor = _append_episode(extra_rows, t_cursor, dense_generate(challenge, cfg), f"dense_{name}")
            extra_ep += 1
    for name, cfg in DART_PRESETS.items():
        for _ in range(16):
            challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
            t_cursor = _append_episode(extra_rows, t_cursor, dart_generate(challenge, cfg), f"dart_{name}")
            extra_ep += 1
    for _ in range(24):
        challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
        t_cursor = _append_episode(extra_rows, t_cursor, hop_generate(challenge, HopConfig()), "playwright_hop")
        extra_ep += 1
    path = os.path.join(OUT_DIR, "ymodel_v02_legacy_live_families.csv")
    write_csv(path, extra_rows)
    summary["files"].append({"path": path, "episodes": extra_ep, "rows": len(extra_rows)})
    print(f"  legacy live: {extra_ep} episodes / {len(extra_rows):,} rows", flush=True)

    print("[V0.2] Fitts on all challenge types...", flush=True)
    fitts_rows: list[dict[str, str]] = []
    t_cursor = 0.0
    fitts_ep = 0
    for ctype in CHALLENGE_TYPES:
        for motor_name, motor in FITTS_PRESETS.items():
            for _ in range(6):
                challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type=ctype)
                label = f"{motor_name}::{ctype}"
                t_cursor = _append_episode(fitts_rows, t_cursor, fitts_generate(challenge, motor), label)
                fitts_ep += 1
        for i in range(5):
            motor = mutate_motor(default_motor(), rng, scale=1.4)
            challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type=ctype)
            label = f"fitts_mut_{i}::{ctype}"
            t_cursor = _append_episode(fitts_rows, t_cursor, fitts_generate(challenge, motor), label)
            fitts_ep += 1
        print(f"  ... after {ctype}: {fitts_ep} Fitts episodes, {len(fitts_rows):,} rows", flush=True)
    path = os.path.join(OUT_DIR, "ymodel_v02_fitts_all_challenges.csv")
    write_csv(path, fitts_rows)
    summary["files"].append({"path": path, "episodes": fitts_ep, "rows": len(fitts_rows)})

    summary["episodes"] = sum(f["episodes"] for f in summary["files"])
    summary["rows"] = sum(f["rows"] for f in summary["files"])
    summary["challenge_types"] = list(CHALLENGE_TYPES)
    summary["named_hop_families"] = list(NAMED_HOP_FAMILIES)
    summary["fitts_presets"] = list(FITTS_PRESETS)
    cfg_path = os.path.join(OUT_DIR, "ymodel_v02_families_cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["cfg_path"] = cfg_path
    return summary


def main() -> None:
    result = export_v02()
    print(json.dumps({k: v for k, v in result.items() if k != "files"} | {"files": result["files"]}, indent=2), flush=True)
    print("FAMILY_EXPORT_COMPLETE ymodel_v0.2", flush=True)


if __name__ == "__main__":
    main()
