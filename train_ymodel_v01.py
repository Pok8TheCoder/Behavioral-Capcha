"""Train Ymodel V0.1 — GPU XGBoost on all human+bot CSVs plus generated bot families."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.qmodel_registry import QMODEL_DIR
from backend.ymodel_registry import YMODEL_DIR, register_new_version
from test.generate_bot_families import export_families
from train_isolation_forest import FEATURE_NAMES
from train_qmodel_v081g import XGB_PARAMS, load_human_os_and_dashboard
from train_random_forest_balanced import (
    BOT_DIRS_ALL,
    evaluate_bot_styles,
    load_labeled_windows,
    session_split,
)

VERSION_ID = os.getenv("YMODEL_OUTPUT_ID", "ymodel_v0.1")
MODEL_PATH = os.path.join(YMODEL_DIR, f"{VERSION_ID}.joblib")
METADATA_PATH = os.path.join(YMODEL_DIR, f"{VERSION_ID}_metadata.json")
FAMILY_DIR = os.path.join(ROOT, "datasets", "bot_families")
PREV_Q_RULES = os.path.join(QMODEL_DIR, "qmodel_v0.15_evasion_rules.json")


def main() -> None:
    print("=" * 72)
    print("Train Ymodel V0.1 — GPU XGBoost, all CSVs + bot family sweep")
    print("=" * 72)

    print("[Data] Generating bot family sweep...", flush=True)
    family_export = export_families()
    print(json.dumps(family_export, indent=2), flush=True)

    human_X, human_groups, human_summaries, human_counts = load_human_os_and_dashboard()
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, Any]] = []
    bot_windows_by_source: Dict[str, int] = {}
    for prefix, bot_dir in BOT_DIRS_ALL + [("bot_families", FAMILY_DIR)]:
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
    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")

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
    tn, fp, fn, tp = cm.ravel()
    acc = float((tn + tp) / len(y_test))
    print("=" * 72)
    print("Ymodel V0.1  (GPU XGBoost, family sweep)")
    print("=" * 72)
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Accuracy: {acc:.6f}")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))

    importances = sorted(
        zip(FEATURE_NAMES, [float(v) for v in clf.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )
    print("[Eval] Fresh bot-style episodes...")
    style_eval = evaluate_bot_styles(clf, episodes_per_style=10)

    os.makedirs(YMODEL_DIR, exist_ok=True)
    rules_name = f"{VERSION_ID}_evasion_rules.json"
    rules_path = os.path.join(YMODEL_DIR, rules_name)
    if os.path.exists(PREV_Q_RULES) and not os.path.exists(rules_path):
        shutil.copy2(PREV_Q_RULES, rules_path)

    metadata = {
        "model_type": "xgboost_gpu",
        "ymodel_id": VERSION_ID,
        "ymodel_label": os.getenv("YMODEL_LABEL", "Ymodel V0.1"),
        "ymodel_version": os.getenv("YMODEL_VERSION", "0.1"),
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_sources": human_counts,
        "spec_features_used": False,
        "bot_evasion_in_training": True,
        "family_sweep_in_training": True,
        "family_export": family_export,
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
        "accuracy": acc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
        "feature_importances": [{"feature": n, "importance": v} for n, v in importances],
        "bot_style_eval": style_eval,
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    register_new_version(
        version_id=VERSION_ID,
        label=os.getenv("YMODEL_LABEL", "Ymodel V0.1"),
        version=os.getenv("YMODEL_VERSION", "0.1"),
        description=(
            "GPU XGBoost on all OS+dashboard human CSVs, all bot CSVs, and a generated "
            "Playwright-like hop family sweep (named winners + random kinematic grid)."
        ),
        patch_cycle=int(os.getenv("YMODEL_PATCH_CYCLE", "1")),
        evasion_rules_filename=rules_name,
        bot_evasion_in_training=True,
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"[Save] {METADATA_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID}")


if __name__ == "__main__":
    main()
