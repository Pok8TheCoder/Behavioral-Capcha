"""
Train a balanced Random Forest on cleaned human windows and generated
human-like bot windows.

This uses the same AFK cleanup and 3s active-window extraction implemented in
train_isolation_forest.py, then trains a supervised classifier with roughly
equal human/bot support.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from train_isolation_forest import (
    FEATURE_NAMES,
    HUMAN_DIR,
    extract_active_windows_from_csv,
)


ROOT_DIR = os.path.dirname(__file__)
BOT_HUMANLIKE_DIR = os.path.join(ROOT_DIR, "datasets", "bot_humanlike")
MODEL_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_rf_humanlike_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_rf_humanlike_metadata.json")
RANDOM_STATE = 42


def load_labeled_windows(input_dir: str, label_name: str, max_windows_per_file: int = 100_000) -> Tuple[List[List[float]], List[str], List[Dict[str, object]]]:
    vectors: List[List[float]] = []
    groups: List[str] = []
    summaries: List[Dict[str, object]] = []
    csvs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    print(f"[Data] Loading {label_name}: {len(csvs)} CSV files")
    for file_path in csvs:
        samples, summary = extract_active_windows_from_csv(file_path, max_windows=max_windows_per_file)
        vectors.extend([sample.vector for sample in samples])
        groups.extend([os.path.basename(file_path)] * len(samples))
        summaries.append(summary)
        print(f"[Data] {label_name} {os.path.basename(file_path)}: {len(samples):,} windows")

    return vectors, groups, summaries


def balanced_subsample(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    human_idx = np.flatnonzero(y == 0)
    bot_idx = np.flatnonzero(y == 1)
    target = min(len(human_idx), len(bot_idx))
    rng = np.random.default_rng(RANDOM_STATE)
    human_keep = rng.choice(human_idx, size=target, replace=False)
    bot_keep = rng.choice(bot_idx, size=target, replace=False)
    keep = np.concatenate([human_keep, bot_keep])
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]


def session_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(set(groups)) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def main() -> None:
    print("=" * 72)
    print("BALANCED HUMAN-LIKE BOT RANDOM FOREST TRAINER")
    print("=" * 72)

    human_X, human_groups, human_summaries = load_labeled_windows(HUMAN_DIR, "human")
    bot_X, bot_groups, bot_summaries = load_labeled_windows(BOT_HUMANLIKE_DIR, "bot_humanlike")

    if not human_X:
        raise RuntimeError("No human windows found.")
    if not bot_X:
        raise RuntimeError("No bot_humanlike windows found. Run generate_humanlike_bot_dataset.py first.")

    X = np.array(human_X + bot_X, dtype=float)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=int)
    groups = np.array([f"human::{g}" for g in human_groups] + [f"bot::{g}" for g in bot_groups])
    X, y, groups = balanced_subsample(X, y, groups)

    X_train, X_test, y_train, y_test = session_split(X, y, groups)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"[Train] Total balanced windows: {len(X):,}")
    print(f"[Train] Train windows: {len(X_train):,}")
    print(f"[Train] Test windows: {len(X_test):,}")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0)
    auc = float(roc_auc_score(y_test, y_proba))

    print("[Eval] Confusion matrix:")
    print(cm)
    print(f"[Eval] ROC-AUC: {auc:.4f}")
    print("[Eval] Classification report:")
    print(report)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda item: item[1], reverse=True)
    print("[Eval] Feature importances:")
    for name, importance in importances:
        print(f"  {name:<24} {importance:.4f}")

    metadata = {
        "model_type": "random_forest_human_vs_humanlike_bot",
        "model_path": MODEL_PATH,
        "feature_names": FEATURE_NAMES,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "balanced_windows": len(X),
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "random_state": RANDOM_STATE,
        "classifier_params": clf.get_params(),
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "classification_report": report,
        "feature_importances": [{"feature": name, "importance": float(value)} for name, value in importances],
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Save] Model: {MODEL_PATH}")
    print(f"[Save] Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
