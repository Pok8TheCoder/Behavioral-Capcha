"""Train Xmodel V0.3 — V0.2S data plus attack-winner bot hardening."""

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

from backend.hz60 import downsample_events_to_hz
from backend.xmodel_registry import XMODEL_DIR, register_new_version
from train_isolation_forest import FEATURE_NAMES
from train_qmodel_v081g import XGB_PARAMS, load_csv_windows
from train_random_forest_balanced import evaluate_bot_styles, load_labeled_windows, session_split
import train_random_forest_balanced as trf

VERSION_ID = os.getenv("XMODEL_OUTPUT_ID", "xmodel_v0.3")
TARGET_HZ = float(os.getenv("XMODEL_TARGET_HZ", "60"))
HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
BOT_ROOT = os.path.join(ROOT, "datasets", "bot_xmodel_60hz")
ATTACK_WINNERS_DIR = os.path.join(BOT_ROOT, "attack_winners")
ATTACK_CFG = os.path.join(ATTACK_WINNERS_DIR, "xmodel_v02s_attack_winners_cfg.json")
MODEL_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}.joblib")
METADATA_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}_metadata.json")

BOT_DIRS_XMODEL = [
    ("bot_xmodel_challenge", os.path.join(BOT_ROOT, "challenge")),
    ("bot_xmodel_tiers", os.path.join(BOT_ROOT, "tiers")),
    ("bot_xmodel_evasion", os.path.join(BOT_ROOT, "evasion")),
    ("bot_xmodel_families", os.path.join(BOT_ROOT, "families")),
    ("bot_xmodel_legacy", os.path.join(BOT_ROOT, "legacy")),
    ("bot_xmodel_attack_winners", ATTACK_WINNERS_DIR),
]


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
    print("=" * 72)
    print("Train Xmodel V0.3 — V0.2S mix + attack-winner bot hardening")
    print("=" * 72)

    if not os.path.isdir(ATTACK_WINNERS_DIR):
        raise SystemExit(
            f"Missing {ATTACK_WINNERS_DIR}. Run: python export_xmodel_attack_winners.py"
        )

    xgb_params = dict(XGB_PARAMS)
    xgb_params["device"] = "cuda"
    xgb_params["tree_method"] = "hist"
    print(f"[Train] XGBoost device={xgb_params['device']} native_hz={TARGET_HZ}", flush=True)

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
    print(f"Xmodel V0.3  ROC-AUC: {auc:.6f}  Accuracy: {acc:.6f}")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))

    orig_extract = trf.extract_features

    def extract_60hz(events):
        return orig_extract(downsample_events_to_hz(events, TARGET_HZ))

    trf.extract_features = extract_60hz
    try:
        style_eval = evaluate_bot_styles(clf, episodes_per_style=10)
    finally:
        trf.extract_features = orig_extract

    attack_export = {}
    if os.path.exists(ATTACK_CFG):
        with open(ATTACK_CFG, encoding="utf-8") as f:
            attack_export = json.load(f)

    bot_gen_summary = {}
    bot_summary_path = os.path.join(BOT_ROOT, "xmodel_60hz_bot_summary.json")
    if os.path.exists(bot_summary_path):
        with open(bot_summary_path, encoding="utf-8") as f:
            bot_gen_summary = json.load(f)

    metadata = {
        "model_type": "xgboost_gpu",
        "xmodel_id": VERSION_ID,
        "xmodel_label": "Xmodel V0.3",
        "xmodel_version": "0.3",
        "target_hz": TARGET_HZ,
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_filter": "60hz_native_only",
        "bot_filter": "native_60hz_regenerated_plus_attack_winners",
        "dataset_mix": "v02s_plus_attack_hardening",
        "parent_model": "xmodel_v0.2S",
        "attack_export": attack_export,
        "human_sources": human_counts,
        "bot_data_root": BOT_ROOT,
        "bot_generation_summary": bot_gen_summary,
        "train_seconds": round(train_sec, 3),
        "xgb_params": xgb_params,
        "feature_names": FEATURE_NAMES,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "bot_windows_by_source": bot_windows_by_source,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "accuracy": acc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
        "bot_style_eval": style_eval,
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }

    os.makedirs(XMODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    register_new_version(
        version_id=VERSION_ID,
        label="Xmodel V0.3",
        version="0.3",
        description=(
            "V0.2S native 60 Hz mix hardened with telemetry from the 11 dashboard-bot "
            "families that beat V0.2S during the timed red-team attack."
        ),
        patch_cycle=3,
        evasion_rules_filename=None,
        human_filter="60hz_native_only",
        make_latest=True,
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID} device={device_used}")


if __name__ == "__main__":
    main()
