"""
Build Qmodel V0.1 (pre-patch) and V0.2 (post-patch) artifacts.

V0.1: RF trained without bot_evasion, no dynamic evasion rules.
V0.2: Current patched challenge model + cycle-1 evasion rules.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMODEL_DIR = os.path.join(ROOT, "backend", "qmodels")
SRC_MODEL = os.path.join(ROOT, "backend", "behavioral_rf_challenge_model.joblib")
SRC_META = os.path.join(ROOT, "backend", "behavioral_rf_challenge_metadata.json")
SRC_RULES = os.path.join(ROOT, "backend", "evasion_heuristic_rules.json")


def stamp_metadata(meta: dict, qmodel_id: str, label: str, version: str, **extra) -> dict:
    meta = dict(meta)
    meta["qmodel_id"] = qmodel_id
    meta["qmodel_label"] = label
    meta["qmodel_version"] = version
    for k, v in extra.items():
        meta[k] = v
    return meta


def build_v0_2() -> None:
    os.makedirs(QMODEL_DIR, exist_ok=True)
    dst_model = os.path.join(QMODEL_DIR, "qmodel_v0.2.joblib")
    dst_meta = os.path.join(QMODEL_DIR, "qmodel_v0.2_metadata.json")
    dst_rules = os.path.join(QMODEL_DIR, "qmodel_v0.2_evasion_rules.json")

    shutil.copy2(SRC_MODEL, dst_model)
    with open(SRC_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = stamp_metadata(
        meta,
        "qmodel_v0.2",
        "Qmodel V0.2",
        "0.2",
        patch_cycle=1,
        bot_evasion_in_training=True,
        procedural_mimic_heuristic=True,
    )
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(SRC_RULES):
        shutil.copy2(SRC_RULES, dst_rules)
    else:
        with open(dst_rules, "w", encoding="utf-8") as f:
            json.dump([], f)
    print(f"[Qmodel] V0.2 artifact: {dst_model}")


def build_v0_1() -> None:
    os.makedirs(QMODEL_DIR, exist_ok=True)
    env = os.environ.copy()
    env["QMODEL_SKIP_BOT_EVASION"] = "1"
    env["QMODEL_OUTPUT_ID"] = "qmodel_v0.1"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train_random_forest_balanced.py")],
        cwd=ROOT,
        env=env,
        capture_output=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("V0.1 training failed")
    meta_path = os.path.join(QMODEL_DIR, "qmodel_v0.1_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = stamp_metadata(
        meta,
        "qmodel_v0.1",
        "Qmodel V0.1",
        "0.1",
        patch_cycle=None,
        bot_evasion_in_training=False,
        procedural_mimic_heuristic=False,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[Qmodel] V0.1 trained into backend/qmodels/")


def main() -> None:
    print("=" * 60)
    print("Building Qmodel artifacts")
    print("=" * 60)
    build_v0_2()
    build_v0_1()
    print("[Done] Registry: backend/qmodel_registry.json")


if __name__ == "__main__":
    main()
