"""
Regenerate all bot training CSVs at native 60 Hz for Xmodel V0.2S.

Mirrors Ymodel V0.3M bot mix (~101k windows) but every episode is exported
at native 60 Hz (no legacy downsample-at-load).

Outputs to datasets/bot_xmodel_60hz/:
  challenge/  — simulated humanoid challenge episodes (~40k windows)
  tiers/      — all CHALLENGE_BOT_STYLES (~55k windows)
  evasion/    — regenerated from evasion_cycle_*_cfg.json (~1k windows)
  families/   — hop/Fitts/playwright-sim families (~4.6k windows)
  legacy/     — tier legacy bots (linear/bezier/naive)
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.hz60 import downsample_events_to_hz
from bots.telemetry_export import ensure_60hz_export_events
from challenge_server.generator import CHALLENGE_TYPES, generate_challenge as generate_challenge_spec
from generate_challenge_bot_dataset import (
    generate_episode as generate_challenge_episode,
    write_csv,
)
from generate_multi_tier_bot_dataset import CHALLENGE_BOT_STYLES, generate_episode_for_style
from test.evasion_bot import cfg_from_dict as fitts_cfg_from_dict, generate_events as fitts_generate
from test.generate_bot_families import NAMED_FAMILIES, export_families, random_family
from test.generate_ymodel_v02_families import export_v02
from test.generate_ymodel_v03_playwright import FAMILIES as PLAYWRIGHT_FAMILIES
from test.qmodel_evasion_bot import evasion_cfg_from_dict, generate_events as evasion_generate
from test.sparse_irregular_bot import SparseConfig, generate_events as hop_generate
from train_isolation_forest import extract_active_windows_from_csv

OUT_ROOT = os.path.join(ROOT, "datasets", "bot_xmodel_60hz")
LEGACY_BOT_DIR = os.path.join(ROOT, "datasets", "bot")
EVASION_SRC = os.path.join(ROOT, "datasets", "bot_evasion")
HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
SCREEN_W = 1920
SCREEN_H = 1080
TARGET_HZ = 60.0
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
RANDOM_STATE = 20260822

# Ymodel V0.3M window targets (downsampled-at-load counts)
YM_CHALLENGE = int(os.getenv("XMODEL_BOT_CHALLENGE_WINDOWS", "40335"))
YM_TIERS = int(os.getenv("XMODEL_BOT_TIERS_WINDOWS", "55186"))
YM_EVASION = int(os.getenv("XMODEL_BOT_EVASION_WINDOWS", "1055"))
YM_FAMILIES = int(os.getenv("XMODEL_BOT_FAMILIES_WINDOWS", "4652"))

TIER_STYLE_RATIOS = {
    "humanoid_simulated": 464 / 1221,
    "linear_timed": 230 / 1221,
    "bezier_timed": 203 / 1221,
    "naive": 120 / 1221,
    "linear_fast": 73 / 1221,
    "bezier_fast": 67 / 1221,
    "humanoid_fast": 64 / 1221,
}

CSV_FIELDS = [
    "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
    "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
    "event_type", "target_item",
]


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    import io
    with open(path, encoding="utf-8") as sf:
        lines = [ln for ln in sf if not ln.startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(io.StringIO("".join(lines))))


def count_windows(path: str) -> int:
    samples, _ = extract_active_windows_from_csv(path, max_windows=10_000_000, target_hz=TARGET_HZ)
    return len(samples)


def events_to_csv_rows(events: List[Dict[str, Any]], t_offset_ms: float, target_item: str) -> List[Dict[str, str]]:
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
            "event_type": str(event.get("event_type", event.get("type", "mouse_move"))),
            "target_item": target_item,
        })
    return rows


def normalize_bot_events(events: List[Dict[str, Any]], rng: random.Random, stamp: bool = False) -> List[Dict[str, Any]]:
    normalized = [
        {
            "x": float(e["x"]),
            "y": float(e["y"]),
            "t": float(e["t"]),
            "type": str(e.get("type", e.get("event_type", "mouse_move"))),
            "event_type": str(e.get("event_type", e.get("type", "mouse_move"))),
        }
        for e in events
    ]
    if stamp:
        return ensure_60hz_export_events(normalized, rng)
    return downsample_events_to_hz(normalized, TARGET_HZ)


def write_family_csv(path: str, rows: List[Dict[str, str]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({SCREEN_W}x{SCREEN_H})\n")
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def generate_challenge_bots(rng: random.Random, target_windows: int) -> Dict[str, Any]:
    out_dir = os.path.join(OUT_ROOT, "challenge")
    os.makedirs(out_dir, exist_ok=True)
    total_windows = 0
    total_episodes = 0
    total_rows = 0
    file_idx = 0
    t_cursor = 0.0
    started = time.time()

    while total_windows < target_windows:
        file_idx += 1
        batch_rows: List[Dict[str, str]] = []
        for _ in range(350):
            events, _meta = generate_challenge_episode(rng)
            events = normalize_bot_events(events, rng, stamp=False)
            batch_rows.extend(events_to_csv_rows(events, t_cursor, "xmodel_challenge_bot"))
            total_episodes += 1
            last_t = max(float(e["t"]) for e in events) if events else 0.0
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)
        out_path = os.path.join(out_dir, f"challenge_bot_{file_idx:03d}_60hz.csv")
        row_count = write_csv(out_path, batch_rows)
        file_windows = count_windows(out_path)
        total_rows += row_count
        total_windows += file_windows
        print(f"[challenge] {os.path.basename(out_path)}: {file_windows:,} windows ({total_windows:,}/{target_windows:,})")
        if file_idx > 500:
            print("[challenge] file limit reached")
            break

    return {
        "dir": out_dir,
        "files": file_idx,
        "episodes": total_episodes,
        "rows": total_rows,
        "windows": total_windows,
        "target": target_windows,
        "seconds": round(time.time() - started, 2),
    }


def generate_tiers(rng: random.Random, target_windows: int) -> Dict[str, Any]:
    out_dir = os.path.join(OUT_ROOT, "tiers")
    os.makedirs(out_dir, exist_ok=True)
    total_windows = 0
    total_episodes = 0
    total_rows = 0
    windows_per_style: Dict[str, int] = {}
    started = time.time()

    style_targets = {
        style: max(200, int(target_windows * TIER_STYLE_RATIOS.get(style, 1 / len(CHALLENGE_BOT_STYLES))))
        for style in CHALLENGE_BOT_STYLES
    }

    for style_id in CHALLENGE_BOT_STYLES:
        style_target = style_targets[style_id]
        style_windows = 0
        file_idx = 0
        print(f"[tiers] style={style_id} target={style_target:,} windows")

        while style_windows < style_target:
            file_idx += 1
            batch_rows: List[Dict[str, str]] = []
            t_cursor = 0.0
            for _ in range(200):
                events, _meta = generate_episode_for_style(rng, style_id)
                events = normalize_bot_events(events, rng, stamp=True)
                batch_rows.extend(events_to_csv_rows(events, t_cursor, "xmodel_tier_bot"))
                total_episodes += 1
                last_t = max(float(e["t"]) for e in events) if events else 0.0
                t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)
            out_path = os.path.join(out_dir, f"tier_{style_id}_{file_idx:03d}_60hz.csv")
            row_count = write_csv(out_path, batch_rows)
            file_windows = count_windows(out_path)
            total_rows += row_count
            style_windows += file_windows
            total_windows += file_windows
            print(f"  {os.path.basename(out_path)}: +{file_windows:,} ({style_windows:,}/{style_target:,})")
            if file_idx > 300:
                print(f"  [warn] file cap for {style_id}")
                break

        windows_per_style[style_id] = style_windows

    return {
        "dir": out_dir,
        "files": sum(1 for _ in glob.glob(os.path.join(out_dir, "*.csv"))),
        "episodes": total_episodes,
        "rows": total_rows,
        "windows": total_windows,
        "target": target_windows,
        "windows_per_style": windows_per_style,
        "seconds": round(time.time() - started, 2),
    }


def regenerate_evasion(rng: random.Random, target_windows: int) -> Dict[str, Any]:
    """Port legacy evasion cycles to native 60 Hz (mixed cfg types)."""
    out_dir = os.path.join(OUT_ROOT, "evasion")
    os.makedirs(out_dir, exist_ok=True)
    total_windows = 0
    total_episodes = 0
    total_rows = 0
    started = time.time()
    legacy_csvs = sorted(
        p for p in glob.glob(os.path.join(EVASION_SRC, "evasion_cycle_*.csv"))
        if "_cfg" not in os.path.basename(p)
    )

    for src_path in legacy_csvs:
        cycle_id = os.path.basename(src_path).replace("evasion_cycle_", "").replace(".csv", "")
        dst_path = os.path.join(out_dir, f"evasion_cycle_{cycle_id}_60hz.csv")
        rows_in = _read_csv_rows(src_path)
        if not rows_in:
            continue
        t0 = float(rows_in[0]["rel_time_ms"])
        events = [
            {
                "x": float(row["x_px"]),
                "y": float(row["y_px"]),
                "t": float(row["rel_time_ms"]) - t0,
                "type": row.get("event_type", "mouse_move"),
                "event_type": row.get("event_type", "mouse_move"),
            }
            for row in rows_in
        ]
        events = downsample_events_to_hz(events, TARGET_HZ)
        out_rows = events_to_csv_rows(events, t0, f"xmodel_evasion_{cycle_id}")
        row_count = write_family_csv(dst_path, out_rows)
        file_windows = count_windows(dst_path)
        total_rows += row_count
        total_windows += file_windows
        total_episodes += 1
        print(f"[evasion] cycle {cycle_id}: {file_windows:,} windows (ported from legacy)", flush=True)

    return {
        "dir": out_dir,
        "files": len(legacy_csvs),
        "episodes": total_episodes,
        "rows": total_rows,
        "windows": total_windows,
        "target": target_windows,
        "seconds": round(time.time() - started, 2),
    }


def playwright_family_to_sparse(cfg: Dict[str, Any]) -> SparseConfig:
    return SparseConfig(
        spacing_px=tuple(cfg["spacing"]),
        min_points=int(cfg["min_points"]),
        max_points=int(cfg["max_points"]),
        arc_px=tuple(cfg["arc_px"]),
        tremor_px=float(cfg["tremor_px"]),
        aim_sigma=float(cfg["aim_sigma"]),
        dwell_ms=(float(cfg["dwell_s"][0]) * 1000, float(cfg["dwell_s"][1]) * 1000),
        pause_ms=(float(cfg["pause_s"][0]) * 1000, float(cfg["pause_s"][1]) * 1000),
        name=str(cfg["name"]),
    )


def generate_families(rng: random.Random, target_windows: int) -> Dict[str, Any]:
    out_dir = os.path.join(OUT_ROOT, "families")
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    file_summaries: List[Dict[str, Any]] = []
    total_windows = 0
    total_episodes = 0
    total_rows = 0

    # V0.1-style named + random sweep
    print("[families] v01 named/random sweep...")
    import test.generate_bot_families as gf1
    orig1 = gf1.OUT_DIR
    gf1.OUT_DIR = out_dir
    v01 = export_families(seed=RANDOM_STATE + 1)
    gf1.OUT_DIR = orig1
    v01_path = v01["csv_path"]
    _resample_csv_to_60hz(v01_path, rng)
    w = count_windows(v01_path)
    os.rename(v01_path, os.path.join(out_dir, "xmodel_v01_families_60hz.csv"))
    v01_path = os.path.join(out_dir, "xmodel_v01_families_60hz.csv")
    total_windows += w
    file_summaries.append({"file": os.path.basename(v01_path), "windows": w, "source": "v01_families"})

    # V0.2 Fitts/hop/grid families
    print("[families] v02 Fitts/hop/grid export...")
    import test.generate_ymodel_v02_families as gf2
    orig2 = gf2.OUT_DIR
    gf2.OUT_DIR = out_dir
    v02 = gf2.export_v02(seed=RANDOM_STATE + 2)
    gf2.OUT_DIR = orig2
    for finfo in v02.get("files", []):
        path = finfo.get("path", "") if isinstance(finfo, dict) else str(finfo)
        if not path or not os.path.exists(path):
            continue
        _resample_csv_to_60hz(path, rng)
        base = os.path.basename(path).replace(".csv", "_60hz.csv")
        new_path = os.path.join(out_dir, f"xmodel_{base}")
        if path != new_path:
            os.replace(path, new_path)
            path = new_path
        w = count_windows(path)
        total_windows += w
        file_summaries.append({"file": os.path.basename(path), "windows": w, "source": "v02_families"})

    # Playwright-family simulation (no browser)
    print("[families] playwright-sim families...")
    pw_rows: List[Dict[str, str]] = []
    t_cursor = 0.0
    pw_episodes = 0
    for fam in PLAYWRIGHT_FAMILIES:
        cfg = playwright_family_to_sparse(fam)
        for _ in range(12):
            challenge = generate_challenge_spec(
                seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence"
            )
            events = hop_generate(challenge, cfg)
            events = normalize_bot_events(events, rng, stamp=True)
            pw_rows.extend(events_to_csv_rows(events, t_cursor, fam["name"]))
            pw_episodes += 1
            total_episodes += 1
            last_t = max(float(e["t"]) for e in events) if events else 0.0
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)
    pw_path = os.path.join(out_dir, "xmodel_playwright_sim_60hz.csv")
    total_rows += write_family_csv(pw_path, pw_rows)
    w = count_windows(pw_path)
    total_windows += w
    file_summaries.append({"file": os.path.basename(pw_path), "windows": w, "episodes": pw_episodes, "source": "playwright_sim"})

    return {
        "dir": out_dir,
        "files": len(file_summaries),
        "episodes": total_episodes + v01.get("episodes", 0) + v02.get("episodes", 0),
        "rows": total_rows,
        "windows": total_windows,
        "target": target_windows,
        "file_summaries": file_summaries,
        "seconds": round(time.time() - started, 2),
    }


def _resample_csv_to_60hz(path: str, rng: random.Random) -> None:
    """Rewrite a bot CSV so movement events are native 60 Hz."""
    header_line = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_line = line
                break
    raw_rows = _read_csv_rows(path)
    if not raw_rows:
        return

    events: List[Dict[str, Any]] = []
    t0 = float(raw_rows[0]["rel_time_ms"])
    for row in raw_rows:
        events.append({
            "x": float(row["x_px"]),
            "y": float(row["y_px"]),
            "t": float(row["rel_time_ms"]) - t0,
            "type": row.get("event_type", "mouse_move"),
            "event_type": row.get("event_type", "mouse_move"),
        })
    events = downsample_events_to_hz(events, TARGET_HZ)
    out_rows = events_to_csv_rows(events, t0, raw_rows[0].get("target_item", "xmodel_bot"))
    with open(path, "w", newline="", encoding="utf-8") as f:
        if header_line:
            f.write(header_line)
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def port_legacy_tiers(rng: random.Random) -> Dict[str, Any]:
    """Regenerate legacy tier bots (linear/bezier/naive) at 60 Hz."""
    out_dir = os.path.join(OUT_ROOT, "legacy")
    os.makedirs(out_dir, exist_ok=True)
    total_windows = 0
    started = time.time()
    for src_path in sorted(glob.glob(os.path.join(LEGACY_BOT_DIR, "bot_*.csv"))):
        base = os.path.basename(src_path).replace(".csv", "_60hz.csv")
        dst = os.path.join(out_dir, base)
        rows_in = _read_csv_rows(src_path)
        if not rows_in:
            continue
        t0 = float(rows_in[0]["rel_time_ms"])
        events = [
            {
                "x": float(row["x_px"]),
                "y": float(row["y_px"]),
                "t": float(row["rel_time_ms"]) - t0,
                "type": row.get("event_type", "mouse_move"),
                "event_type": row.get("event_type", "mouse_move"),
            }
            for row in rows_in
        ]
        events = downsample_events_to_hz(events, TARGET_HZ)
        out_rows = events_to_csv_rows(events, t0, rows_in[0].get("target_item", "xmodel_legacy_bot"))
        write_family_csv(dst, out_rows)
        w = count_windows(dst)
        total_windows += w
        print(f"[legacy] {base}: {w:,} windows")
    return {
        "dir": out_dir,
        "files": len(glob.glob(os.path.join(out_dir, "*.csv"))),
        "windows": total_windows,
        "seconds": round(time.time() - started, 2),
    }


def _count_dir_windows(subdir: str) -> int:
    d = os.path.join(OUT_ROOT, subdir)
    if not os.path.isdir(d):
        return 0
    total = 0
    for path in glob.glob(os.path.join(d, "*.csv")):
        total += count_windows(path)
    return total


def main() -> None:
    rng = random.Random(RANDOM_STATE)
    os.makedirs(OUT_ROOT, exist_ok=True)
    print("=" * 72, flush=True)
    print("XMODEL 60 Hz BOT DATASET GENERATOR — Ymodel-scale native export", flush=True)
    print("=" * 72, flush=True)
    print(f"[Plan] challenge target: {YM_CHALLENGE:,}", flush=True)
    print(f"[Plan] tiers target:     {YM_TIERS:,}", flush=True)
    print(f"[Plan] evasion target:   {YM_EVASION:,}", flush=True)
    print(f"[Plan] families target:  {YM_FAMILIES:,}", flush=True)
    print(f"[Plan] Output root: {OUT_ROOT}", flush=True)

    summary: Dict[str, Any] = {
        "target_hz": TARGET_HZ,
        "ymodel_targets": {
            "challenge": YM_CHALLENGE,
            "tiers": YM_TIERS,
            "evasion": YM_EVASION,
            "families": YM_FAMILIES,
        },
        "random_state": RANDOM_STATE,
        "output_root": OUT_ROOT,
    }

    existing_challenge = _count_dir_windows("challenge")
    if existing_challenge >= int(YM_CHALLENGE * 0.95):
        print(f"[skip] challenge already has {existing_challenge:,} windows", flush=True)
        summary["challenge"] = {"windows": existing_challenge, "skipped": True}
    else:
        summary["challenge"] = generate_challenge_bots(rng, YM_CHALLENGE)

    existing_tiers = _count_dir_windows("tiers")
    if existing_tiers >= int(YM_TIERS * 0.95):
        print(f"[skip] tiers already has {existing_tiers:,} windows", flush=True)
        summary["tiers"] = {"windows": existing_tiers, "skipped": True}
    else:
        summary["tiers"] = generate_tiers(rng, YM_TIERS)

    existing_evasion = _count_dir_windows("evasion")
    if existing_evasion >= int(YM_EVASION * 0.5):
        print(f"[skip] evasion already has {existing_evasion:,} windows", flush=True)
        summary["evasion"] = {"windows": existing_evasion, "skipped": True}
    else:
        summary["evasion"] = regenerate_evasion(rng, YM_EVASION)

    existing_families = _count_dir_windows("families")
    if existing_families >= int(YM_FAMILIES * 0.5):
        print(f"[skip] families already has {existing_families:,} windows", flush=True)
        summary["families"] = {"windows": existing_families, "skipped": True}
    else:
        summary["families"] = generate_families(rng, YM_FAMILIES)
    summary["legacy"] = port_legacy_tiers(rng)
    summary["total_bot_windows"] = (
        summary["challenge"]["windows"]
        + summary["tiers"]["windows"]
        + summary["evasion"]["windows"]
        + summary["families"]["windows"]
        + summary["legacy"]["windows"]
    )
    summary_path = os.path.join(OUT_ROOT, "xmodel_60hz_bot_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-" * 72)
    print(f"[Done] Total bot windows: {summary['total_bot_windows']:,}")
    print(f"[Done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
