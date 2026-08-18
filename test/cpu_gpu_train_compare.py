"""Compare CPU sklearn RF vs GPU XGBoost training on the same Qmodel V0.4 data split."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_isolation_forest import HUMAN_DIR, FEATURE_NAMES
from train_random_forest_balanced import (
    BOT_DIRS,
    RF_PARAMS,
    balanced_subsample,
    evaluate_bot_styles,
    load_labeled_windows,
    session_split,
)

V04_PATH = os.path.join(ROOT, "backend", "qmodels", "qmodel_v0.4.joblib")
V04_META = os.path.join(ROOT, "backend", "qmodels", "qmodel_v0.4_metadata.json")
OUT_PATH = os.path.join(ROOT, "test", "cpu_gpu_compare_result.json")

XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": RF_PARAMS["n_estimators"],
    "max_depth": RF_PARAMS["max_depth"],
    "min_child_weight": RF_PARAMS["min_samples_leaf"],
    "subsample": 0.8,
    "colsample_bytree": 0.35,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "logloss",
    "random_state": RF_PARAMS["random_state"],
    "n_jobs": 1,
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    human_X, human_groups, _ = load_labeled_windows(HUMAN_DIR, "human")
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    for prefix, bot_dir in BOT_DIRS:
        vectors, groups, _ = load_labeled_windows(bot_dir, prefix)
        bot_X.extend(vectors)
        bot_groups.extend(groups)

    X = np.array(human_X + bot_X, dtype=float)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=int)
    groups = np.array(human_groups + bot_groups)
    return balanced_subsample(X, y, groups)


def eval_classifier(clf: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = float(roc_auc_score(y_test, y_proba))
    tn, fp, fn, tp = cm.ravel()
    return {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_sklearn_order": "[[TN, FP], [FN, TP]]",
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "human_tnr": round(tn / (tn + fp), 6) if (tn + fp) else None,
        "bot_recall": round(tp / (tp + fn), 6) if (tp + fn) else None,
        "bot_precision": round(tp / (tp + fp), 6) if (tp + fp) else None,
        "accuracy": round((tn + tp) / len(y_test), 6),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
    }


def print_cm_block(title: str, metrics: Dict[str, Any], train_sec: float | None = None) -> None:
    print("=" * 72)
    print(title)
    if train_sec is not None:
        print(f"Training time: {train_sec:.2f}s")
    print("=" * 72)
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Human TNR: {metrics['human_tnr']:.6f}  ({metrics['tn']}/{metrics['tn'] + metrics['fp']})")
    print(f"Bot recall:{metrics['bot_recall']:.6f}  ({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    print(f"Bot prec:  {metrics['bot_precision']:.6f}  ({metrics['tp']}/{metrics['tp'] + metrics['fp']})")
    print()
    print("                 Predicted")
    print("                 Human    Bot")
    print(f"Actual Human     {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"Actual Bot       {metrics['fn']:5d}   {metrics['tp']:5d}")
    print()
    print("sklearn confusion_matrix [[TN, FP], [FN, TP]]:")
    print(np.array(metrics["confusion_matrix"]))
    print()
    print(metrics["classification_report"])


def main() -> None:
    print("Loading dataset (same pipeline as train_random_forest_balanced)...")
    X, y, groups = load_dataset()
    X_train, X_test, y_train, y_test = session_split(X, y, groups)
    print(f"Balanced pool: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")

    results: Dict[str, Any] = {
        "dataset": {
            "balanced_windows": len(X),
            "train_windows": len(X_train),
            "test_windows": len(X_test),
            "features": FEATURE_NAMES,
        },
        "note": (
            "sklearn RandomForestClassifier runs on CPU only. "
            "GPU comparison uses XGBoost (hist + CUDA) with matched tree depth/count."
        ),
    }

    # --- Saved Qmodel V0.4 artifact ---
    if os.path.exists(V04_PATH):
        artifact = joblib.load(V04_PATH)
        saved_clf = artifact["model"]
        saved_metrics = eval_classifier(saved_clf, X_test, y_test)
        with open(V04_META, encoding="utf-8") as f:
            meta = json.load(f)
        saved_metrics["artifact_train_confusion_matrix"] = meta.get("confusion_matrix")
        saved_metrics["artifact_train_roc_auc"] = meta.get("roc_auc")
        saved_metrics["artifact_oob_score"] = meta.get("oob_score")
        saved_metrics["artifact_train_windows"] = meta.get("train_windows")
        saved_metrics["artifact_test_windows"] = meta.get("test_windows")
        results["qmodel_v0.4_saved"] = saved_metrics
        print_cm_block("Qmodel V0.4 (saved artifact) — re-eval on same holdout split", saved_metrics)

    # --- CPU: fresh sklearn RF ---
    cpu_clf = RandomForestClassifier(**RF_PARAMS)
    cpu_start = time.perf_counter()
    cpu_clf.fit(X_train, y_train)
    cpu_train_sec = time.perf_counter() - cpu_start
    cpu_metrics = eval_classifier(cpu_clf, X_test, y_test)
    cpu_metrics["oob_score"] = float(cpu_clf.oob_score_) if hasattr(cpu_clf, "oob_score_") else None
    cpu_metrics["train_seconds"] = round(cpu_train_sec, 3)
    cpu_style = evaluate_bot_styles(cpu_clf, episodes_per_style=15)
    cpu_metrics["bot_style_eval"] = cpu_style
    results["cpu_sklearn_rf"] = cpu_metrics
    print_cm_block("CPU — sklearn RandomForest (fresh train)", cpu_metrics, cpu_train_sec)

    # --- GPU: XGBoost ---
    gpu_clf = XGBClassifier(**XGB_PARAMS)
    gpu_start = time.perf_counter()
    gpu_clf.fit(X_train, y_train, verbose=False)
    gpu_train_sec = time.perf_counter() - gpu_start
    gpu_metrics = eval_classifier(gpu_clf, X_test, y_test)
    gpu_metrics["train_seconds"] = round(gpu_train_sec, 3)
    gpu_metrics["xgb_params"] = XGB_PARAMS
    gpu_style = evaluate_bot_styles(gpu_clf, episodes_per_style=15)
    gpu_metrics["bot_style_eval"] = gpu_style
    results["gpu_xgboost"] = gpu_metrics
    print_cm_block("GPU — XGBoost (device=cuda)", gpu_metrics, gpu_train_sec)

    speedup = cpu_train_sec / gpu_train_sec if gpu_train_sec > 0 else None
    results["timing"] = {
        "cpu_sklearn_rf_seconds": round(cpu_train_sec, 3),
        "gpu_xgboost_seconds": round(gpu_train_sec, 3),
        "cpu_vs_gpu_speedup": round(speedup, 2) if speedup else None,
    }

    print("=" * 72)
    print("TIMING SUMMARY")
    print("=" * 72)
    print(f"CPU sklearn RF:  {cpu_train_sec:.2f}s")
    print(f"GPU XGBoost:     {gpu_train_sec:.2f}s")
    if speedup:
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")

    if results.get("qmodel_v0.4_saved"):
        saved = results["qmodel_v0.4_saved"]
        print()
        print("Saved V0.4 artifact (from last training run metadata):")
        print(f"  Train CM: {saved.get('artifact_train_confusion_matrix')}")
        print(f"  Train ROC-AUC: {saved.get('artifact_train_roc_auc')}")
        print(f"  OOB score: {saved.get('artifact_oob_score')}")

    print()
    print("Bot-style detection (fresh episodes, 15 per style):")
    for style in sorted(cpu_style.keys()):
        c = cpu_style[style]["bot_detection_rate"]
        g = gpu_style[style]["bot_detection_rate"]
        print(f"  {style:<22} CPU RF {c*100:5.1f}%  |  GPU XGB {g*100:5.1f}%")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
