"""Train Xmodel V0.1 — GPU XGBoost on 60 Hz human recordings + fresh 60 Hz bot data."""

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
from train_isolation_forest import FEATURE_NAMES, extract_active_windows_from_csv
from train_qmodel_v081g import XGB_PARAMS, load_csv_windows
from train_random_forest_balanced import evaluate_bot_styles, load_labeled_windows, session_split
import train_random_forest_balanced as trf

VERSION_ID = os.getenv("XMODEL_OUTPUT_ID", "xmodel_v0.1")
TARGET_HZ = float(os.getenv("XMODEL_TARGET_HZ", "60"))
HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
BOT_ROOT = os.path.join(ROOT, "datasets", "bot_xmodel_60hz")
MODEL_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}.joblib")
METADATA_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}_metadata.json")

BOT_DIRS_XMODEL = [
    ("bot_xmodel_challenge", os.path.join(BOT_ROOT, "challenge")),
    ("bot_xmodel_tiers", os.path.join(BOT_ROOT, "tiers")),
    ("bot_xmodel_families", os.path.join(BOT_ROOT, "families")),
    ("bot_xmodel_humanlike", os.path.join(BOT_ROOT, "humanlike")),
]


def load_human_60hz_only(
    target_hz: float = TARGET_HZ,
) -> Tuple[List[List[float]], List[str], List[Dict[str, Any]], Dict[str, int]]:
    vectors: List[List[float]] = []
    groups: List[str] = []
    summaries: List[Dict[str, Any]] = []
    counts = {"os_recordings_60hz": 0}

    csvs = sorted(
        p for p in glob.glob(os.path.join(HUMAN_DIR, "*.csv"))
        if "_60hz" in os.path.basename(p).lower()
    )
    print(f"[Data] Human 60 Hz OS recordings: {len(csvs)} CSV files")
    for path in csvs:
        vecs, grps, summary = load_csv_windows(
            path, "human_60hz", os.path.basename(path), target_hz=target_hz
        )
        vectors.extend(vecs)
        groups.extend(grps)
        summaries.append(summary)
        counts["os_recordings_60hz"] += len(vecs)

    return vectors, groups, summaries, counts


def main() -> None:
    print("=" * 72)
    print(f"Train Xmodel V0.1 — GPU XGBoost, 60 Hz human + 60 Hz bot")
    print("=" * 72)

    if not os.path.isdir(BOT_ROOT):
        raise SystemExit(
            f"Missing {BOT_ROOT}. Run: python generate_xmodel_60hz_bots.py"
        )

    xgb_params = dict(XGB_PARAMS)
    xgb_params["device"] = "cuda"
    xgb_params["tree_method"] = "hist"
    print(f"[Train] XGBoost device={xgb_params['device']} target_hz={TARGET_HZ}", flush=True)

    human_X, human_groups, human_summaries, human_counts = load_human_60hz_only(target_hz=TARGET_HZ)
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, Any]] = []
    bot_windows_by_source: Dict[str, int] = {}
    for prefix, bot_dir in BOT_DIRS_XMODEL:
        vectors, groups, summaries = load_labeled_windows(bot_dir, prefix, target_hz=TARGET_HZ)
        bot_X.extend(vectors)
        bot_groups.extend(groups)
        bot_summaries.extend(summaries)
        bot_windows_by_source[prefix] = len(vectors)

    if not human_X:
        raise SystemExit("No human 60 Hz windows found (expected datasets/human/*_60hz*.csv).")
    if not bot_X:
        raise SystemExit("No bot 60 Hz windows found. Run generate_xmodel_60hz_bots.py first.")

    X = np.array(human_X + bot_X, dtype=np.float32)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=np.int32)
    groups = np.array(human_groups + bot_groups)
    X_train, X_test, y_train, y_test = session_split(X, y, groups)

    print(f"[Train] Human windows: {len(human_X):,}  {human_counts}")
    print(f"[Train] Bot windows:   {len(bot_X):,}  {bot_windows_by_source}")
    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"[Train] Human fraction: {float(np.mean(y == 0)):.4f}  Bot fraction: {float(np.mean(y == 1)):.4f}")

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
    print(f"[Train] Fit time: {train_sec:.2f}s on {device_used}", flush=True)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = float(roc_auc_score(y_test, y_proba))
    tn, fp, fn, tp = cm.ravel()
    acc = float((tn + tp) / len(y_test))
    print("=" * 72)
    print("Xmodel V0.1")
    print(f"Device: {device_used}  target_hz={TARGET_HZ:.0f}  ROC-AUC: {auc:.6f}  Accuracy: {acc:.6f}")
    print("Confusion [[TN, FP], [FN, TP]]:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))

    importances = sorted(
        zip(FEATURE_NAMES, [float(v) for v in clf.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )
    print("[Eval] Feature importances:")
    for name, importance in importances:
        print(f"  {name:<24} {importance:.4f}")

    orig_extract = trf.extract_features

    def extract_60hz(events):
        return orig_extract(downsample_events_to_hz(events, TARGET_HZ))

    trf.extract_features = extract_60hz
    try:
        style_eval = evaluate_bot_styles(clf, episodes_per_style=10)
    finally:
        trf.extract_features = orig_extract

    bot_summary_path = os.path.join(BOT_ROOT, "xmodel_60hz_bot_summary.json")
    bot_gen_summary = {}
    if os.path.exists(bot_summary_path):
        with open(bot_summary_path, encoding="utf-8") as f:
            bot_gen_summary = json.load(f)

    metadata = {
        "model_type": "xgboost_gpu",
        "xmodel_id": VERSION_ID,
        "xmodel_label": "Xmodel V0.1",
        "xmodel_version": "0.1",
        "target_hz": TARGET_HZ,
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_filter": "60hz_filename_only",
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
        "feature_importances": [{"feature": n, "importance": v} for n, v in importances],
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
        label="Xmodel V0.1",
        version="0.1",
        description=(
            "GPU XGBoost trained on 60 Hz human OS recordings only and freshly "
            "regenerated 60 Hz bot datasets (challenge, tiers, families, humanlike)."
        ),
        patch_cycle=1,
        evasion_rules_filename=None,
        make_latest=True,
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"[Save] {METADATA_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID} device={device_used} target_hz={TARGET_HZ:.0f}")


if __name__ == "__main__":
    main()
