"""Train Qmodel V0.8.1G — GPU XGBoost on all human OS+dashboard CSVs and all bot data.

Human dashboard CSVs use the same 12 kinematic features as OS recordings (no spec extras).
No class balancing. Does not replace V0.8 patch heuristics; new artifact alongside.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.qmodel_registry import QMODEL_DIR, register_new_version
from train_isolation_forest import FEATURE_NAMES, extract_active_windows_from_csv
from train_random_forest_balanced import (
    BOT_DIRS_ALL,
    RF_PARAMS,
    evaluate_bot_styles,
    load_labeled_windows,
    session_split,
)

VERSION_ID = os.getenv("QMODEL_OUTPUT_ID", "qmodel_v0.8.1G")
HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
CHALLENGE_DIR = os.path.join(HUMAN_DIR, "challenges")
MODEL_PATH = os.path.join(QMODEL_DIR, f"{VERSION_ID}.joblib")
METADATA_PATH = os.path.join(QMODEL_DIR, f"{VERSION_ID}_metadata.json")
V08_RULES = os.path.join(QMODEL_DIR, "qmodel_v0.8_evasion_rules.json")

XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": int(RF_PARAMS["n_estimators"]),
    "max_depth": int(RF_PARAMS["max_depth"]),
    "min_child_weight": int(RF_PARAMS["min_samples_leaf"]),
    "subsample": 0.8,
    "colsample_bytree": 0.35,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "logloss",
    "random_state": int(RF_PARAMS["random_state"]),
    "n_jobs": 1,
}


def load_csv_windows(
    file_path: str,
    label_prefix: str,
    group_key: str,
    target_hz: Optional[float] = None,
) -> Tuple[List[List[float]], List[str], Dict[str, Any]]:
    samples, summary = extract_active_windows_from_csv(
        file_path, max_windows=100_000, target_hz=target_hz
    )
    vectors = [sample.vector for sample in samples]
    groups = [f"{label_prefix}::{group_key}"] * len(vectors)
    print(f"[Data] {label_prefix} {group_key}: {len(vectors):,} windows")
    return vectors, groups, summary


def load_human_os_and_dashboard(
    target_hz: Optional[float] = None,
) -> Tuple[List[List[float]], List[str], List[Dict[str, Any]], Dict[str, int]]:
    vectors: List[List[float]] = []
    groups: List[str] = []
    summaries: List[Dict[str, Any]] = []
    counts = {"os_recordings": 0, "dashboard_live": 0}

    os_csvs = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.csv")))
    print(f"[Data] OS recordings: {len(os_csvs)} CSV files")
    for path in os_csvs:
        vecs, grps, summary = load_csv_windows(path, "human_os", os.path.basename(path), target_hz=target_hz)
        vectors.extend(vecs)
        groups.extend(grps)
        summaries.append(summary)
        counts["os_recordings"] += len(vecs)

    dash_csvs = sorted(glob.glob(os.path.join(CHALLENGE_DIR, "**", "*.csv"), recursive=True))
    dash_csvs = [p for p in dash_csvs if "_archive_no_spec" not in os.path.normpath(p).split(os.sep)]
    print(f"[Data] Dashboard live CSVs (kinematics only): {len(dash_csvs)} files")
    for path in dash_csvs:
        rel = os.path.relpath(path, CHALLENGE_DIR).replace("\\", "/")
        vecs, grps, summary = load_csv_windows(path, "human_dashboard", rel, target_hz=target_hz)
        vectors.extend(vecs)
        groups.extend(grps)
        summaries.append(summary)
        counts["dashboard_live"] += len(vecs)

    return vectors, groups, summaries, counts


def print_confusion(cm: np.ndarray, auc: float, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    tn, fp, fn, tp = cm.ravel()
    print("=" * 72)
    print("Qmodel V0.8.1G  (GPU XGBoost, unbalanced)")
    print("=" * 72)
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Accuracy: {(tn + tp) / len(y_test):.6f}")
    print()
    print("                 Predicted")
    print("                 Human    Bot")
    print(f"Actual Human     {tn:5d}   {fp:5d}")
    print(f"Actual Bot       {fn:5d}   {tp:5d}")
    print()
    print("sklearn order [[TN, FP], [FN, TP]]:")
    print(cm)
    print()
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))


def main() -> None:
    print("=" * 72)
    print("Train Qmodel V0.8.1G — GPU XGBoost, all human+bot windows, no balance")
    print("=" * 72)

    human_X, human_groups, human_summaries, human_counts = load_human_os_and_dashboard()
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, Any]] = []
    bot_windows_by_source: Dict[str, int] = {}
    for prefix, bot_dir in BOT_DIRS_ALL:
        vectors, groups, summaries = load_labeled_windows(bot_dir, prefix)
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

    print(f"[Train] Human windows: {len(human_X):,}  (OS {human_counts['os_recordings']:,} + dashboard {human_counts['dashboard_live']:,})")
    print(f"[Train] Bot windows:   {len(bot_X):,}  {bot_windows_by_source}")
    print(f"[Train] Unbalanced total: {len(X):,}")
    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"[Train] Human fraction: {float(np.mean(y == 0)):.4f}  Bot fraction: {float(np.mean(y == 1)):.4f}")

    clf = XGBClassifier(**XGB_PARAMS)
    started = time.perf_counter()
    try:
        clf.fit(X_train, y_train, verbose=False)
        device_used = "cuda"
    except Exception as exc:
        print(f"[Train] CUDA fit failed ({exc}); retrying on CPU.")
        cpu_params = dict(XGB_PARAMS)
        cpu_params["device"] = "cpu"
        clf = XGBClassifier(**cpu_params)
        started = time.perf_counter()
        clf.fit(X_train, y_train, verbose=False)
        device_used = "cpu"
    train_sec = time.perf_counter() - started
    print(f"[Train] Fit time: {train_sec:.2f}s on {device_used}")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = float(roc_auc_score(y_test, y_proba))
    print_confusion(cm, auc, y_test, y_pred)

    importances = sorted(
        zip(FEATURE_NAMES, [float(v) for v in clf.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )
    print("[Eval] Feature importances:")
    for name, importance in importances:
        print(f"  {name:<24} {importance:.4f}")

    print("[Eval] Fresh bot-style episodes...")
    style_eval = evaluate_bot_styles(clf, episodes_per_style=10)

    rules_name = f"{VERSION_ID}_evasion_rules.json"
    rules_path = os.path.join(QMODEL_DIR, rules_name)
    if not os.path.exists(rules_path) and os.path.exists(V08_RULES):
        shutil.copy2(V08_RULES, rules_path)

    metadata = {
        "model_type": "xgboost_gpu",
        "qmodel_id": VERSION_ID,
        "qmodel_label": os.getenv("QMODEL_LABEL", "Qmodel V0.8.1G"),
        "qmodel_version": os.getenv("QMODEL_VERSION", "0.8.1G"),
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_sources": human_counts,
        "spec_features_used": False,
        "bot_evasion_in_training": True,
        "train_seconds": round(train_sec, 3),
        "xgb_params": {**XGB_PARAMS, "device": device_used},
        "feature_names": FEATURE_NAMES,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "bot_windows_by_source": bot_windows_by_source,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
        "feature_importances": [{"feature": n, "importance": v} for n, v in importances],
        "bot_style_eval": style_eval,
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }

    os.makedirs(QMODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    register_new_version(
        version_id=VERSION_ID,
        label=os.getenv("QMODEL_LABEL", "Qmodel V0.8.1G"),
        version=os.getenv("QMODEL_VERSION", "0.8.1G"),
        description=(
            "GPU XGBoost retrained on all OS human recordings plus dashboard live CSVs "
            "(same 12 kinematic features, no challenge spec extras) and all bot sources; unbalanced."
        ),
        patch_cycle=int(os.getenv("QMODEL_PATCH_CYCLE", "8")),
        evasion_rules_filename=rules_name,
        bot_evasion_in_training=True,
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"[Save] {METADATA_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID}")


if __name__ == "__main__":
    main()
