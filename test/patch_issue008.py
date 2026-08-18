"""Blue-team patch for Issue 008 — mid_count ML evasion vs Qmodel V0.9."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVASION_DIR = os.path.join(ROOT, "datasets", "bot_evasion")
SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
CYCLE_ID = 9
VERSION_ID = "qmodel_v0.10"

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.evasion_heuristics import append_rule_from_features, load_rules
from backend.qmodel_registry import QMODEL_DIR
from challenge_server.generator import generate_challenge
from test.evasion_bot import events_to_csv_rows
from test.mid_count_bot import FAMILY_NAMES, FAMILY_PRESETS, generate_events, winning_feature_sample

PREV_RULES = os.path.join(QMODEL_DIR, "qmodel_v0.9_evasion_rules.json")


def export_dataset(cycle_id: int, episodes_per_family: int = 16, seed: int = 910) -> dict:
    os.makedirs(EVASION_DIR, exist_ok=True)
    rng = random.Random(seed)
    out_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}.csv")
    cfg_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}_cfg.json")
    all_rows: list[dict[str, str]] = []
    t_cursor = 0.0
    episodes = 0
    for family in FAMILY_NAMES:
        for _ in range(episodes_per_family):
            challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
            events = generate_events(challenge, FAMILY_PRESETS[family])
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
        json.dump({"families": FAMILY_NAMES, "episodes_per_family": episodes_per_family}, f, indent=2)
    return {"csv_path": out_path, "episodes": episodes, "rows": len(all_rows)}


def build_rules(feature_sample: dict) -> str:
    rules_filename = f"{VERSION_ID}_evasion_rules.json"
    rules_path = os.path.join(QMODEL_DIR, rules_filename)
    if os.path.exists(PREV_RULES):
        shutil.copy2(PREV_RULES, rules_path)
    else:
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    append_rule_from_features(CYCLE_ID, feature_sample, rules_path)
    return rules_filename


def main() -> None:
    print("Exporting mid_count dataset (cycle 009)...")
    print(json.dumps(export_dataset(CYCLE_ID), indent=2), flush=True)
    rules_filename = build_rules(winning_feature_sample())
    rules_path = os.path.join(QMODEL_DIR, rules_filename)
    print(f"Rules: {rules_path} ({len(load_rules(rules_path))} entries)", flush=True)
    env = os.environ.copy()
    env["QMODEL_OUTPUT_ID"] = VERSION_ID
    env["QMODEL_LABEL"] = "Qmodel V0.10"
    env["QMODEL_VERSION"] = "0.10"
    env["QMODEL_PATCH_CYCLE"] = str(CYCLE_ID)
    env["PYTHONUNBUFFERED"] = "1"
    print("Retraining GPU XGBoost as qmodel_v0.10...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", os.path.join(ROOT, "train_qmodel_v081g.py")],
        cwd=ROOT,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit("Training failed")
    print("PATCH_COMPLETE qmodel_v0.10", flush=True)


if __name__ == "__main__":
    main()
