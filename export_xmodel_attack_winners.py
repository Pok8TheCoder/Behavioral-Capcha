"""Export dashboard-bot telemetry from Xmodel attack winners for retraining."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.hz60 import downsample_events_to_hz
from bots.dashboard_bots import run_dashboard_bot
from bots.event_utils import normalize_bot_events
from challenge_server.generator import generate_challenge
from train_isolation_forest import extract_active_windows_from_csv

ATTACK_RESULT = os.path.join(
    ROOT, "agent_evasion", "xmodel_attack_xmodel_v0_2S_result.json"
)
OUT_DIR = os.path.join(ROOT, "datasets", "bot_xmodel_60hz", "attack_winners")
OUT_CFG = os.path.join(OUT_DIR, "xmodel_v02s_attack_winners_cfg.json")
SCREEN_W = 1920
SCREEN_H = 1080
TARGET_HZ = 60.0
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
EPISODES_PER_FAMILY = int(os.getenv("XMODEL_ATTACK_EXPORT_EPISODES", "40"))


def events_to_rows(events: List[Dict[str, Any]], t_offset: float, label: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for event in events:
        x = float(event["x"])
        y = float(event["y"])
        etype = str(event.get("event_type", event.get("type", "mouse_move")))
        rows.append({
            "rel_time_ms": f"{t_offset + float(event['t']):.3f}",
            "x_px": f"{x:.1f}",
            "y_px": f"{y:.1f}",
            "x_norm": f"{x / SCREEN_W:.5f}",
            "y_norm": f"{y / SCREEN_H:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": "0",
            "scroll_dy": "0",
            "event_type": etype,
            "target_item": label,
        })
    return rows


def run_bot_episode(bot_id: str, challenge_type: str, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    challenge = generate_challenge(seed=seed, challenge_type=challenge_type)
    stamped = run_dashboard_bot(challenge, bot_id)
    events = normalize_bot_events(stamped)
    return downsample_events_to_hz(events, TARGET_HZ)


def export_family(
    bot_id: str,
    winning: Dict[str, Any],
    episodes: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    t_cursor = 0.0
    episode_meta: List[Dict[str, Any]] = []

    seeds = [int(winning["seed"])]
    rng = random.Random(int(winning["seed"]) ^ hash(bot_id))
    while len(seeds) < episodes:
        candidate = rng.randint(1, 2_000_000_000)
        if candidate not in seeds:
            seeds.append(candidate)

    for i, seed in enumerate(seeds[:episodes]):
        ctype = winning["challenge_type"] if i == 0 else None
        if i > 0:
            ctype = rng.choice([
                "button_sequence", "pin_entry", "shape_trace", "maze_navigate",
                "path_gates", "spiral_trace", "connect_dots", "target_hunt",
            ])
        events = run_bot_episode(bot_id, ctype or winning["challenge_type"], seed)
        rows.extend(events_to_rows(events, t_cursor, f"xmodel_attack_{bot_id}"))
        last_t = max(float(e["t"]) for e in events) if events else 0.0
        episode_meta.append({
            "bot": bot_id,
            "seed": seed,
            "challenge_type": ctype or winning["challenge_type"],
            "events": len(events),
            "is_winning_replay": i == 0,
        })
        t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)

    out_path = os.path.join(OUT_DIR, f"attack_winner_{bot_id}_60hz.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
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
        for row in rows:
            writer.writerow(row)

    windows = len(extract_active_windows_from_csv(out_path, max_windows=10_000_000, target_hz=TARGET_HZ)[0])
    return {
        "bot_id": bot_id,
        "csv": out_path,
        "episodes": len(episode_meta),
        "rows": len(rows),
        "windows": windows,
        "winning_hc": winning.get("human_confidence"),
        "episodes_meta": episode_meta,
    }


def main() -> None:
    if not os.path.exists(ATTACK_RESULT):
        raise SystemExit(f"Missing attack result: {ATTACK_RESULT}")

    with open(ATTACK_RESULT, encoding="utf-8") as f:
        attack = json.load(f)

    beaten = attack.get("beaten_families") or {}
    if not beaten:
        raise SystemExit("No beaten families in attack result.")

    print(f"[Export] {len(beaten)} beaten families -> {OUT_DIR}")
    files: List[Dict[str, Any]] = []
    total_windows = 0
    for bot_id, info in beaten.items():
        winning = info.get("winning_attempt") or {}
        if not winning.get("seed"):
            print(f"[skip] {bot_id}: no winning seed")
            continue
        summary = export_family(bot_id, winning, EPISODES_PER_FAMILY)
        files.append(summary)
        total_windows += int(summary["windows"])
        print(
            f"  {bot_id}: {summary['windows']:,} windows "
            f"({summary['episodes']} episodes, winner hc={summary['winning_hc']:.1%})"
        )

    cfg = {
        "source_attack": ATTACK_RESULT,
        "target_model": attack.get("target_model"),
        "target_hz": TARGET_HZ,
        "episodes_per_family": EPISODES_PER_FAMILY,
        "beaten_families": list(beaten.keys()),
        "files": files,
        "total_windows": total_windows,
    }
    with open(OUT_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Done] {total_windows:,} attack-winner windows exported")
    print(f"[Done] Config: {OUT_CFG}")


if __name__ == "__main__":
    main()
