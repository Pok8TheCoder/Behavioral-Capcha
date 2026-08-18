"""Blue-team patch: export evasion telemetry, retrain RF, append dynamic heuristics."""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVASION_DIR = os.path.join(ROOT, "datasets", "bot_evasion")
SCREEN_W = 1920
SCREEN_H = 1080
AFK_GAP_MS_RANGE = (5_500.0, 7_500.0)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.evasion_heuristics import append_rule_from_features
from backend.qmodel_registry import QMODEL_DIR, next_patch_version, register_new_version
from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from test.adversarial_utils import human_confidence_from_events, load_classifier
from test.evasion_bot import events_to_csv_rows, generate_events, motor_to_dict
from test.fitts_motor import FittsMotorConfig


def export_evasion_dataset(
    motor: FittsMotorConfig,
    cycle_id: int,
    episodes_per_type: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(EVASION_DIR, exist_ok=True)
    rng = random.Random(seed + cycle_id)
    out_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}.csv")
    cfg_path = os.path.join(EVASION_DIR, f"evasion_cycle_{cycle_id:03d}_cfg.json")

    all_rows: List[Dict[str, str]] = []
    meta: List[Dict[str, Any]] = []
    t_cursor = 0.0

    for ctype in CHALLENGE_TYPES:
        for _ in range(episodes_per_type):
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type=ctype,
            )
            events = generate_events(challenge, motor)
            all_rows.extend(events_to_csv_rows(events, t_cursor))
            last_t = max(float(e["t"]) for e in events)
            meta.append({
                "challenge_type": ctype,
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
        json.dump(motor_to_dict(motor), f, indent=2)

    return {
        "csv_path": out_path,
        "cfg_path": cfg_path,
        "episodes": len(meta),
        "rows": len(all_rows),
        "challenge_types": CHALLENGE_TYPES,
    }


def retrain_detector(qmodel_id: Optional[str] = None) -> Dict[str, Any]:
    script = os.path.join(ROOT, "train_random_forest_balanced.py")
    env = os.environ.copy()
    if qmodel_id:
        env["QMODEL_OUTPUT_ID"] = qmodel_id
    started = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.time() - started
    return {
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def patch_detector(
    motor: FittsMotorConfig,
    cycle_id: int,
    feature_sample: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    export_info = export_evasion_dataset(motor, cycle_id)
    version_id = next_patch_version()
    version_label = f"Qmodel V{version_id.replace('qmodel_v', '')}"
    rules_filename = f"{version_id}_evasion_rules.json"
    rules_path = os.path.join(QMODEL_DIR, rules_filename)

    heuristic_rule = None
    if feature_sample:
        heuristic_rule = append_rule_from_features(cycle_id, feature_sample, rules_path)

    train_info = retrain_detector(qmodel_id=version_id)
    registry_entry = register_new_version(
        version_id=version_id,
        label=version_label,
        version=version_id.replace("qmodel_v", ""),
        description=f"After adversarial red-team cycle {cycle_id}",
        patch_cycle=cycle_id,
        evasion_rules_filename=rules_filename,
        bot_evasion_in_training=True,
    )
    return {
        "qmodel_id": version_id,
        "qmodel_label": version_label,
        "registry": registry_entry,
        "export": export_info,
        "heuristic_rule": heuristic_rule,
        "train": train_info,
    }


def sample_evasion_features(motor: FittsMotorConfig, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    clf = load_classifier("qmodel_v0.2")
    challenge = generate_challenge(
        seed=rng.randint(1, 2_000_000_000),
        challenge_type="button_sequence",
    )
    events = generate_events(challenge, motor)
    _, detail = human_confidence_from_events(events, clf)
    return detail.get("features", {})
