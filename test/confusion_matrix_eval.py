"""Confusion matrix eval on the saved random_forest_challenge artifact."""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_random_forest_balanced import (
    BOT_DIRS,
    balanced_subsample,
    load_labeled_windows,
)
from train_isolation_forest import HUMAN_DIR

MODEL_PATH = os.path.join(ROOT, "backend", "behavioral_rf_challenge_model.joblib")


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    clf = artifact["model"]

    human_X, human_groups, _ = load_labeled_windows(HUMAN_DIR, "human")
    bot_X, bot_groups = [], []
    for prefix, bot_dir in BOT_DIRS:
        vectors, groups, _ = load_labeled_windows(bot_dir, prefix)
        bot_X.extend(vectors)
        bot_groups.extend(groups)

    X = np.array(human_X + bot_X, dtype=float)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=int)
    groups = np.array(human_groups + bot_groups)
    X, y, groups = balanced_subsample(X, y, groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()
    print("=" * 60)
    print("random_forest_challenge — held-out confusion matrix")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Test windows: {len(y_test):,} (20% group holdout, balanced pool)")
    print()
    print("                 Predicted")
    print("                 Human    Bot")
    print(f"Actual Human     {tn:5d}   {fp:5d}")
    print(f"Actual Bot       {fn:5d}   {tp:5d}")
    print()
    print("sklearn confusion_matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))
    print("Rates:")
    print(f"  Human accuracy (TNR): {tn / (tn + fp):.4f}  ({tn}/{tn + fp})")
    print(f"  Bot recall (TPR):     {tp / (tp + fn):.4f}  ({tp}/{tp + fn})")
    print(f"  Bot precision:        {tp / (tp + fp):.4f}  ({tp}/{tp + fp})")
    print(f"  Overall accuracy:     {(tn + tp) / len(y_test):.4f}")


if __name__ == "__main__":
    main()
