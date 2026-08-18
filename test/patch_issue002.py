"""Blue-team patch for Issue 002 — live Playwright hop+jiggle evasion vs Qmodel V0.3."""

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
CYCLE_ID = 3

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.evasion_heuristics import append_rule_from_features, load_rules
from backend.qmodel_registry import QMODEL_DIR, register_new_version
from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from test.evasion_bot import events_to_csv_rows
from test.patch_detector import retrain_detector
from test.playwright_hop_bot import default_hop_cfg, generate_events, winning_feature_sample

V03_RULES = os.path.join(QMODEL_DIR, "qmodel_v0.3_evasion_rules.json")


def export_hop_dataset(cycle_id: int, episodes_per_type: int = 12, seed: int = 202) -> dict:
    os.makedirs(EVASION_DIR, exist_ok=True)
    rng = random.Random(seed)
    out_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}.csv")
    cfg_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}_cfg.json")

    all_rows: list[dict[str, str]] = []
    meta: list[dict] = []
    t_cursor = 0.0

    for ctype in CHALLENGE_TYPES:
        for _ in range(episodes_per_type):
            if ctype != "button_sequence":
                continue
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type=ctype,
            )
            cfg = default_hop_cfg()
            events = generate_events(challenge, cfg)
            all_rows.extend(events_to_csv_rows(events, t_cursor))
            last_t = max(float(e["t"]) for e in events)
            meta.append({
                "challenge_type": ctype,
                "seed": challenge.get("seed"),
                "events": len(events),
                "duration_ms": last_t,
            })
            t_cursor += last_t + rng.uniform(*AFK_GAP_MS_RANGE)

    hop_cfg_doc = {
        "family": "os_direct_v2",
        "step_px": [18.0, 42.0],
        "arc_px": [22.0, 55.0],
        "micro_jiggles": [1, 3],
        "playwright_hop_dt_ms": 8.0,
    }

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
        json.dump(hop_cfg_doc, f, indent=2)

    return {
        "csv_path": out_path,
        "cfg_path": cfg_path,
        "episodes": len(meta),
        "rows": len(all_rows),
    }


def build_v04_rules(feature_sample: dict) -> str:
    version_id = "qmodel_v0.4"
    rules_filename = f"{version_id}_evasion_rules.json"
    rules_path = os.path.join(QMODEL_DIR, rules_filename)

    if os.path.exists(V03_RULES):
        shutil.copy2(V03_RULES, rules_path)
    else:
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    append_rule_from_features(CYCLE_ID, feature_sample, rules_path)
    return rules_filename


def main() -> None:
    features = winning_feature_sample()

    print("Exporting Playwright hop+jiggle evasion dataset (cycle 003)...")
    export_info = export_hop_dataset(CYCLE_ID)
    print(json.dumps(export_info, indent=2))

    print("Building Qmodel V0.4 evasion rules...")
    rules_filename = build_v04_rules(features)
    rules_path = os.path.join(QMODEL_DIR, rules_filename)
    print(f"Rules: {rules_path} ({len(load_rules(rules_path))} entries)")

    print("Retraining random forest as qmodel_v0.4...")
    train_info = retrain_detector(qmodel_id="qmodel_v0.4")
    print(json.dumps(train_info, indent=2))
    if train_info.get("exit_code") != 0:
        raise SystemExit("Training failed")

    registry_entry = register_new_version(
        version_id="qmodel_v0.4",
        label="Qmodel V0.4",
        version="0.4",
        description="After Issue 002 live Playwright hop+jiggle evasion + kinematic hardening",
        patch_cycle=CYCLE_ID,
        evasion_rules_filename=rules_filename,
        bot_evasion_in_training=True,
    )
    print(json.dumps({"registry": registry_entry}, indent=2))
    print("PATCH_COMPLETE qmodel_v0.4")


if __name__ == "__main__":
    main()
