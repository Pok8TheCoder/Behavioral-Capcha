"""Train the next Xmodel iteration from V0.3 mix plus evasion-loop winners."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.xmodel_registry import (
    XMODEL_DIR,
    catalog_for_api,
    next_ea_xmodel_id,
    parse_ea_cycle,
    register_new_version,
    xmodel_display_label,
    xmodel_version_token,
)
from train_isolation_forest import FEATURE_NAMES
from train_qmodel_v081g import XGB_PARAMS, load_csv_windows
from train_random_forest_balanced import load_labeled_windows, session_split

HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
BOT_ROOT = os.path.join(ROOT, "datasets", "bot_xmodel_60hz")
LOOP_WINNERS_DIR = os.path.join(BOT_ROOT, "loop_winners")
TARGET_HZ = 60.0

BOT_DIRS_XMODEL = [
    ("bot_xmodel_challenge", os.path.join(BOT_ROOT, "challenge")),
    ("bot_xmodel_tiers", os.path.join(BOT_ROOT, "tiers")),
    ("bot_xmodel_evasion", os.path.join(BOT_ROOT, "evasion")),
    ("bot_xmodel_families", os.path.join(BOT_ROOT, "families")),
    ("bot_xmodel_legacy", os.path.join(BOT_ROOT, "legacy")),
    ("bot_xmodel_attack_winners", os.path.join(BOT_ROOT, "attack_winners")),
    ("bot_xmodel_loop_winners", LOOP_WINNERS_DIR),
    ("bot_xmodel_probe_fails", os.path.join(BOT_ROOT, "probe_fails")),
]


def next_xmodel_id() -> Tuple[str, str, int, str]:
    forced = os.getenv("XMODEL_OUTPUT_ID", "").strip()
    parent = os.getenv("XMODEL_PARENT_ID", "").strip()
    if not parent:
        parent = catalog_for_api().get("latest") or "xmodel_v0.3"
    version_id = forced or next_ea_xmodel_id(parent)
    version = xmodel_version_token(version_id)
    patch = parse_ea_cycle(version_id)
    return version_id, version, patch, parent


def load_human_60hz_native() -> Tuple[List[List[float]], List[str], List[Dict[str, Any]], Dict[str, int]]:
    vectors: List[List[float]] = []
    groups: List[str] = []
    summaries: List[Dict[str, Any]] = []
    counts = {"os_recordings_60hz_native": 0}
    csvs = sorted(
        p for p in glob.glob(os.path.join(HUMAN_DIR, "*.csv"))
        if "_60hz" in os.path.basename(p).lower()
    )
    print(f"[Data] Native 60 Hz human recordings: {len(csvs)} CSV files")
    for path in csvs:
        vecs, grps, summary = load_csv_windows(
            path, "human_60hz_native", os.path.basename(path), target_hz=None
        )
        vectors.extend(vecs)
        groups.extend(grps)
        summaries.append(summary)
        counts["os_recordings_60hz_native"] += len(vecs)
    return vectors, groups, summaries, counts


def main() -> None:
    version_id, version, patch, parent_id = next_xmodel_id()
    label = xmodel_display_label(version_id)
    model_path = os.path.join(XMODEL_DIR, f"{version_id}.joblib")
    metadata_path = os.path.join(XMODEL_DIR, f"{version_id}_metadata.json")
    print("=" * 72)
    print(f"Train {version_id} from {parent_id} — evasion-loop winners")
    print("=" * 72)

    xgb_params = dict(XGB_PARAMS)
    xgb_params["device"] = "cuda"
    xgb_params["tree_method"] = "hist"

    human_X, human_groups, human_summaries, human_counts = load_human_60hz_native()
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, Any]] = []
    bot_windows_by_source: Dict[str, int] = {}
    for prefix, bot_dir in BOT_DIRS_XMODEL:
        if not os.path.isdir(bot_dir):
            print(f"[Data] Skipping missing bot dir: {bot_dir}")
            continue
        vectors, groups, summaries = load_labeled_windows(bot_dir, prefix, target_hz=None)
        bot_X.extend(vectors)
        bot_groups.extend(groups)
        bot_summaries.extend(summaries)
        bot_windows_by_source[prefix] = len(vectors)

    if not human_X or not bot_X:
        raise SystemExit("Need both human and bot windows.")

    X = np.array(human_X + bot_X, dtype=np.float32)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=np.int32)
    groups = np.array(human_groups + bot_groups)
    X_train, X_test, y_train, y_test = session_split(X, y, groups)
    print(f"[Train] Human windows: {len(human_X):,}  {human_counts}")
    print(f"[Train] Bot windows:   {len(bot_X):,}  {bot_windows_by_source}")
    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")

    clf = XGBClassifier(**xgb_params)
    started = time.perf_counter()
    device_used = "cuda"
    try:
        clf.fit(X_train, y_train, verbose=False)
    except Exception as exc:
        print(f"[Train] CUDA fit failed ({exc}); retrying on CPU.")
        cpu_params = dict(xgb_params)
        cpu_params["device"] = "cpu"
        clf = XGBClassifier(**cpu_params)
        xgb_params = cpu_params
        device_used = "cpu"
        started = time.perf_counter()
        clf.fit(X_train, y_train, verbose=False)
    train_sec = time.perf_counter() - started

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = float(roc_auc_score(y_test, y_proba))
    acc = float((cm[0, 0] + cm[1, 1]) / len(y_test))
    print("=" * 72)
    print(f"{version_id}  ROC-AUC: {auc:.6f}  Accuracy: {acc:.6f}")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))

    metadata = {
        "model_type": "xgboost_gpu",
        "xmodel_id": version_id,
        "xmodel_label": label,
        "xmodel_version": version,
        "parent_model": parent_id,
        "target_hz": TARGET_HZ,
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_filter": "60hz_native_only",
        "bot_filter": "native_60hz_plus_loop_winners",
        "dataset_mix": "evasion_loop",
        "human_sources": human_counts,
        "bot_windows_by_source": bot_windows_by_source,
        "train_seconds": round(train_sec, 3),
        "xgb_params": xgb_params,
        "feature_names": FEATURE_NAMES,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "accuracy": acc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }
    os.makedirs(XMODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, model_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    register_new_version(
        version_id=version_id,
        label=label,
        version=version,
        description=(
            f"Evasion-arena child of {parent_id}: native 60 Hz mix plus dashboard-bot "
            "episodes that scored >=65% human confidence against the previous iteration."
        ),
        patch_cycle=patch,
        evasion_rules_filename=None,
        human_filter="60hz_native_only",
        make_latest=True,
    )
    print(f"[Save] {model_path}")
    print(f"TRAIN_COMPLETE {version_id} device={device_used}")


if __name__ == "__main__":
    main()
