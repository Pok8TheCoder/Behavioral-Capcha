"""
Train balanced Random Forest on human + all challenge bot tiers.

Bot data sources:
  - datasets/bot_challenge/       (simulated humanoid episodes)
  - datasets/bot_challenge_tiers/ (linear, bezier, naive, fast/timed humanoid)
  - datasets/bot/                 (legacy tier CSVs, if present)
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from backend.active_window import (
    DEFAULT_ACTIVE_WINDOW_CONFIG,
    segment_episode_into_windows,
    _split_on_gaps,
)
from backend.feature_extractor import extract_features, feature_dict_to_vector
from bots.challenge_bot_factory import CHALLENGE_BOT_STYLES, prelude_bot
from bots.challenge_solver import solve_challenge
from bots.telemetry_export import ensure_export_timestamps
from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from generate_challenge_bot_dataset import challenge_entry_point
from train_isolation_forest import (
    FEATURE_NAMES,
    HUMAN_DIR,
    extract_active_windows_from_csv,
)

ROOT_DIR = os.path.dirname(__file__)
BOT_DIRS_ALL = [
    ("bot_challenge", os.path.join(ROOT_DIR, "datasets", "bot_challenge")),
    ("bot_challenge_tiers", os.path.join(ROOT_DIR, "datasets", "bot_challenge_tiers")),
    ("bot_evasion", os.path.join(ROOT_DIR, "datasets", "bot_evasion")),
    ("bot_tier_legacy", os.path.join(ROOT_DIR, "datasets", "bot")),
]
BOT_DIRS = (
    [d for d in BOT_DIRS_ALL if d[0] != "bot_evasion"]
    if os.getenv("QMODEL_SKIP_BOT_EVASION") == "1"
    else BOT_DIRS_ALL
)
QMODEL_OUTPUT_ID = os.getenv("QMODEL_OUTPUT_ID", "").strip()
if QMODEL_OUTPUT_ID:
    MODEL_PATH = os.path.join(ROOT_DIR, "backend", "qmodels", f"{QMODEL_OUTPUT_ID}.joblib")
    METADATA_PATH = os.path.join(ROOT_DIR, "backend", "qmodels", f"{QMODEL_OUTPUT_ID}_metadata.json")
else:
    MODEL_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_rf_challenge_model.joblib")
    METADATA_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_rf_challenge_metadata.json")
RANDOM_STATE = 42

RF_PARAMS: Dict[str, object] = {
    "n_estimators": 600,
    "max_depth": 16,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "oob_score": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_labeled_windows(
    input_dir: str,
    label_prefix: str,
    max_windows_per_file: int = 100_000,
    target_hz: Optional[float] = None,
) -> Tuple[List[List[float]], List[str], List[Dict[str, object]]]:
    vectors: List[List[float]] = []
    groups: List[str] = []
    summaries: List[Dict[str, object]] = []
    if not os.path.isdir(input_dir):
        return vectors, groups, summaries

    csvs = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    print(f"[Data] Loading {label_prefix}: {len(csvs)} CSV files from {input_dir}")
    for file_path in csvs:
        samples, summary = extract_active_windows_from_csv(
            file_path, max_windows=max_windows_per_file, target_hz=target_hz
        )
        relative = os.path.relpath(file_path, input_dir).replace("\\", "/")
        vectors.extend([sample.vector for sample in samples])
        groups.extend([f"{label_prefix}::{relative}"] * len(samples))
        summaries.append(summary)
        print(f"[Data] {label_prefix} {relative}: {len(samples):,} windows")

    return vectors, groups, summaries


def balanced_subsample(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    human_idx = np.flatnonzero(y == 0)
    bot_idx = np.flatnonzero(y == 1)
    target = min(len(human_idx), len(bot_idx))
    if target == 0:
        raise RuntimeError("Cannot balance: one class has zero samples.")
    rng = np.random.default_rng(RANDOM_STATE)
    human_keep = rng.choice(human_idx, size=target, replace=False)
    bot_keep = rng.choice(bot_idx, size=target, replace=False)
    keep = np.concatenate([human_keep, bot_keep])
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]


def session_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(set(groups)) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def evaluate_bot_styles(
    clf: RandomForestClassifier,
    episodes_per_style: int = 15,
) -> Dict[str, Dict[str, float]]:
    """Score each bot tier on fresh random challenges (dynamic windows)."""
    rng = random.Random(RANDOM_STATE + 99)
    cfg = DEFAULT_ACTIVE_WINDOW_CONFIG
    results: Dict[str, Dict[str, float]] = {}

    for style_id, factory in CHALLENGE_BOT_STYLES.items():
        detected = 0
        windows = 0
        for _ in range(episodes_per_style):
            bot, solve_style, fast = factory(rng)
            prelude_bot(bot, rng, 1920, 1080)
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type=rng.choice(CHALLENGE_TYPES),
            )
            entry_x, entry_y = challenge_entry_point(challenge)
            lx = getattr(bot, "_last_x", None) or float(bot.events[-1]["x"])
            ly = getattr(bot, "_last_y", None) or float(bot.events[-1]["y"])
            if getattr(bot, "simulated", False):
                bot.move_to(lx, ly, entry_x, entry_y)
                bot.advance(rng.uniform(40.0, 180.0))
            elif not fast:
                bot.move_to(lx, ly, entry_x, entry_y)
            solve_challenge(bot, challenge, style=solve_style, fast=fast)
            events = ensure_export_timestamps(list(bot.events), rng)
            sorted_events = sorted(events, key=lambda e: float(e["t"]))
            for episode in _split_on_gaps(sorted_events, cfg.afk_gap_ms):
                for window in segment_episode_into_windows(episode, cfg):
                    windows += 1
                    prob = float(
                        clf.predict_proba([feature_dict_to_vector(extract_features(window))])[0][1]
                    )
                    if prob >= 0.5:
                        detected += 1
        rate = detected / max(1, windows)
        results[style_id] = {
            "bot_detection_rate": round(rate, 4),
            "windows_scored": windows,
            "windows_flagged_bot": detected,
        }
        print(f"[Eval-style] {style_id}: {rate * 100:.1f}% detected ({detected}/{windows} windows)")
    return results


def main() -> None:
    print("=" * 72)
    print("MULTI-TIER CHALLENGE BOT RANDOM FOREST TRAINER")
    print("=" * 72)
    print("[Params]", json.dumps(RF_PARAMS, indent=2))

    human_X, human_groups, human_summaries = load_labeled_windows(HUMAN_DIR, "human")

    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, object]] = []
    bot_windows_by_source: Dict[str, int] = {}

    for prefix, bot_dir in BOT_DIRS:
        vectors, groups, summaries = load_labeled_windows(bot_dir, prefix)
        bot_X.extend(vectors)
        bot_groups.extend(groups)
        bot_summaries.extend(summaries)
        bot_windows_by_source[prefix] = len(vectors)

    if not human_X:
        raise RuntimeError("No human windows found.")
    if not bot_X:
        raise RuntimeError(
            "No bot windows found. Run generate_multi_tier_bot_dataset.py first."
        )

    X = np.array(human_X + bot_X, dtype=float)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=int)
    groups = np.array(human_groups + bot_groups)
    X, y, groups = balanced_subsample(X, y, groups)

    X_train, X_test, y_train, y_test = session_split(X, y, groups)
    clf = RandomForestClassifier(**RF_PARAMS)

    print(f"[Train] Human windows loaded: {len(human_X):,}")
    print(f"[Train] Bot windows loaded: {len(bot_X):,} by source: {bot_windows_by_source}")
    print(f"[Train] Balanced total: {len(X):,}")
    print(f"[Train] Train windows: {len(X_train):,}")
    print(f"[Train] Test windows: {len(X_test):,}")

    clf.fit(X_train, y_train)

    if hasattr(clf, "oob_score_"):
        print(f"[Train] OOB score: {clf.oob_score_:.4f}")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0)
    auc = float(roc_auc_score(y_test, y_proba))

    print("[Eval] Confusion matrix (held-out file groups):")
    print(cm)
    print(f"[Eval] ROC-AUC: {auc:.4f}")
    print("[Eval] Classification report:")
    print(report)

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    print("[Eval] Feature importances:")
    for name, importance in importances:
        print(f"  {name:<24} {importance:.4f}")

    style_eval = evaluate_bot_styles(clf)

    metadata = {
        "model_type": "random_forest_challenge",
        "qmodel_id": QMODEL_OUTPUT_ID or None,
        "qmodel_label": f"Qmodel V{QMODEL_OUTPUT_ID.replace('qmodel_v', '')}" if QMODEL_OUTPUT_ID else None,
        "qmodel_version": QMODEL_OUTPUT_ID.replace("qmodel_v", "") if QMODEL_OUTPUT_ID else None,
        "bot_evasion_in_training": os.getenv("QMODEL_SKIP_BOT_EVASION") != "1",
        "window_mode": "dynamic_active",
        "window_config": {
            "afk_gap_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.afk_gap_ms,
            "pause_gap_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.pause_gap_ms,
            "strong_pause_gap_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.strong_pause_gap_ms,
            "min_window_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.min_window_ms,
            "max_window_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.max_window_ms,
            "fallback_window_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.fallback_window_ms,
            "fallback_stride_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.fallback_stride_ms,
        },
        "model_path": MODEL_PATH,
        "feature_names": FEATURE_NAMES,
        "rf_params": RF_PARAMS,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "bot_windows_by_source": bot_windows_by_source,
        "bot_style_eval": style_eval,
        "balanced_windows": len(X),
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "oob_score": float(clf.oob_score_) if hasattr(clf, "oob_score_") else None,
        "random_state": RANDOM_STATE,
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "classification_report": report,
        "feature_importances": [
            {"feature": name, "importance": float(value)} for name, value in importances
        ],
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Save] Model: {MODEL_PATH}")
    print(f"[Save] Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
