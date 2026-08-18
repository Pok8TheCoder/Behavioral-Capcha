"""Blue-team patch for Issue 005 — fast_direct dart-hop evasion vs Qmodel V0.6."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVASION_DIR = os.path.join(ROOT, "datasets", "bot_evasion")
SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)
CYCLE_ID = 6

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.evasion_heuristics import append_rule_from_features, load_rules
from backend.qmodel_registry import QMODEL_DIR, register_new_version
from challenge_server.generator import generate_challenge
from test.evasion_bot import events_to_csv_rows
from test.fast_direct_bot import FAMILY_NAMES, FAMILY_PRESETS, generate_events, winning_feature_sample
from test.patch_detector import retrain_detector

V06_RULES = os.path.join(QMODEL_DIR, "qmodel_v0.6_evasion_rules.json")


def export_dart_dataset(cycle_id: int, episodes_per_family: int = 12, seed: int = 606) -> dict:
    os.makedirs(EVASION_DIR, exist_ok=True)
    rng = random.Random(seed)
    out_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}.csv")
    cfg_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}_cfg.json")

    all_rows: list[dict[str, str]] = []
    meta: list[dict] = []
    t_cursor = 0.0

    for family in FAMILY_NAMES:
        for _ in range(episodes_per_family):
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type="button_sequence",
            )
            events = generate_events(challenge, FAMILY_PRESETS[family])
            all_rows.extend(events_to_csv_rows(events, t_cursor))
            last_t = max(float(e["t"]) for e in events)
            meta.append({
                "family": family,
                "seed": challenge.get("seed"),
                "events": len(events),
                "duration_ms": last_t,
            })
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)

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

    return {
        "csv_path": out_path,
        "cfg_path": cfg_path,
        "episodes": len(meta),
        "rows": len(all_rows),
        "families": FAMILY_NAMES,
    }


def build_v07_rules(feature_sample: dict) -> str:
    version_id = "qmodel_v0.7"
    rules_filename = f"{version_id}_evasion_rules.json"
    rules_path = os.path.join(QMODEL_DIR, rules_filename)

    if os.path.exists(V06_RULES):
        shutil.copy2(V06_RULES, rules_path)
    else:
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    append_rule_from_features(CYCLE_ID, feature_sample, rules_path)
    return rules_filename


def main() -> None:
    features = winning_feature_sample()

    print("Exporting dart-hop family evasion dataset (cycle 006)...")
    export_info = export_dart_dataset(CYCLE_ID)
    print(json.dumps(export_info, indent=2))

    print("Building Qmodel V0.7 evasion rules...")
    rules_filename = build_v07_rules(features)
    rules_path = os.path.join(QMODEL_DIR, rules_filename)
    print(f"Rules: {rules_path} ({len(load_rules(rules_path))} entries)")

    print("Retraining random forest as qmodel_v0.7...")
    train_info = retrain_detector(qmodel_id="qmodel_v0.7")
    print(json.dumps(train_info, indent=2))
    if train_info.get("exit_code") != 0:
        raise SystemExit("Training failed")

    registry_entry = register_new_version(
        version_id="qmodel_v0.7",
        label="Qmodel V0.7",
        version="0.7",
        description="After Issue 005 fast_direct dart-hop live Playwright evasion",
        patch_cycle=CYCLE_ID,
        evasion_rules_filename=rules_filename,
        bot_evasion_in_training=True,
    )
    print(json.dumps({"registry": registry_entry}, indent=2))
    print("PATCH_COMPLETE qmodel_v0.7")


if __name__ == "__main__":
    main()
