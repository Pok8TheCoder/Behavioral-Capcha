"""
Behavioral CAPTCHA Classification Model Trainer (v1.1 Balanced Movement & Click Edition)
-----------------------------------------------------------------------------------------
Pipeline:
1. Human Telemetry Sanitizer & Feature Extractor:
   - Extracts both Pure Mouse Movement windows and Movement + Click windows.
   - Sets avg_click_dwell_ms = -1.0 for pure movement windows.
   - Class Label: 0 (Human)

2. Bot Telemetry Suite & Feature Extractor:
   - Generates both Pure Movement bot paths and Movement + Click bot paths.
   - Across all 4 Bot Tiers: Naive, Linear, Bezier, Humanoid Fitts.
   - Class Label: 1 (Bot)
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from backend.feature_extractor import extract_features, feature_dict_to_vector
from bots.tier1_naive import NaiveBot
from bots.tier2_linear import LinearBot
from bots.tier3_bezier import BezierBot
from bots.tier4_humanoid import HumanoidFittsBot

def sanitize_and_extract_human_features(human_csv_files: List[str]) -> List[List[float]]:
    """Loads human CSV logs, applies AFK gap splitting (>3s), extracts 15s windows, and computes feature vectors."""
    human_vectors = []
    total_raw_events = 0
    total_active_windows = 0

    print("\n" + "=" * 65)
    print(f"[Sanitizer] Processing {len(human_csv_files)} Human Telemetry CSV Files...")
    print("=" * 65)

    for file_path in human_csv_files:
        try:
            df = pd.read_csv(file_path, comment='#')
            if len(df) < 15:
                continue

            total_raw_events += len(df)

            time_col = 'rel_time_ms' if 'rel_time_ms' in df.columns else ('t' if 't' in df.columns else None)
            x_col = 'x_px' if 'x_px' in df.columns else ('x' if 'x' in df.columns else None)
            y_col = 'y_px' if 'y_px' in df.columns else ('y' if 'y' in df.columns else None)
            evt_col = 'event_type' if 'event_type' in df.columns else ('type' if 'type' in df.columns else None)

            if not (time_col and x_col and y_col):
                continue

            # Detect AFK Gaps (> 3.0 seconds)
            df['dt_gap_s'] = df[time_col].diff() / 1000.0
            episode_ids = (df['dt_gap_s'] > 3.0).cumsum()
            df['episode_id'] = episode_ids

            for _, ep in df.groupby('episode_id'):
                if len(ep) < 15:
                    continue

                dx = ep[x_col].diff().dropna()
                dy = ep[y_col].diff().dropna()
                dist = np.sum(np.sqrt(dx**2 + dy**2))
                if dist < 50.0:
                    continue

                t_start = ep[time_col].iloc[0]
                t_end = ep[time_col].iloc[-1]
                ep_duration = (t_end - t_start) / 1000.0

                win_size_ms = 15000.0
                stride_ms = 5000.0

                if ep_duration <= 20.0:
                    events = [{'x': r[x_col], 'y': r[y_col], 't': r[time_col], 'type': r[evt_col]} for _, r in ep.iterrows()]
                    feats = extract_features(events)
                    human_vectors.append(feature_dict_to_vector(feats))
                    total_active_windows += 1
                else:
                    cur_t = t_start
                    while cur_t + win_size_ms <= t_end:
                        sub_ep = ep[(ep[time_col] >= cur_t) & (ep[time_col] < cur_t + win_size_ms)]
                        if len(sub_ep) >= 15:
                            events = [{'x': r[x_col], 'y': r[y_col], 't': r[time_col], 'type': r[evt_col]} for _, r in sub_ep.iterrows()]
                            feats = extract_features(events)
                            human_vectors.append(feature_dict_to_vector(feats))
                            total_active_windows += 1
                        cur_t += stride_ms

        except Exception as e:
            print(f"[Sanitizer Warning] Failed processing {os.path.basename(file_path)}: {e}")

    print(f"[Sanitizer] Loaded {total_raw_events:,} raw human events.")
    print(f"[Sanitizer] Extracted {total_active_windows:,} clean active human training windows.")
    return human_vectors

def generate_and_extract_bot_features(n_runs_per_bot: int = 180) -> List[List[float]]:
    """Generates balanced bot datasets (50% Pure Movement, 50% Movement + Click) across all 4 bot tiers."""
    bot_vectors = []

    print("\n" + "=" * 65)
    print(f"[Bot Suite] Generating Balanced Bot Datasets (Pure Move + Clicks) across 4 Tiers...")
    print("=" * 65)

    bots = [
        NaiveBot(),
        LinearBot(steps=35, step_delay_sec=0.008),
        BezierBot(steps=45, step_delay_sec=0.008),
        HumanoidFittsBot(target_width=40.0)
    ]

    targets = [
        (100.0, 100.0, 450.0, 280.0),
        (450.0, 280.0, 680.0, 520.0),
        (680.0, 520.0, 320.0, 180.0),
        (320.0, 180.0, 850.0, 340.0),
        (850.0, 340.0, 200.0, 600.0)
    ]

    for bot in bots:
        # 1. Pure Movement Runs (No Clicks)
        bot.reset()
        for idx in range(n_runs_per_bot // 2):
            sx, sy, tx, ty = targets[idx % len(targets)]
            bot.move_to(sx, sy, tx, ty)
            if (idx + 1) % 3 == 0 or idx == (n_runs_per_bot // 2) - 1:
                feats = extract_features(bot.events)
                bot_vectors.append(feature_dict_to_vector(feats))
                bot.reset()

        # 2. Movement + Click Runs
        bot.reset()
        for idx in range(n_runs_per_bot // 2):
            sx, sy, tx, ty = targets[idx % len(targets)]
            bot.move_to(sx, sy, tx, ty)
            bot.click_target(tx, ty, button="mouse_left")
            if (idx + 1) % 3 == 0 or idx == (n_runs_per_bot // 2) - 1:
                feats = extract_features(bot.events)
                bot_vectors.append(feature_dict_to_vector(feats))
                bot.reset()

    print(f"[Bot Suite] Extracted {len(bot_vectors):,} balanced bot feature vectors.")
    return bot_vectors

def main():
    print("=" * 65)
    print(" BEHAVIORAL CAPTCHA MODEL TRAINER (V1.1 BALANCED MOVEMENT & CLICK)")
    print("=" * 65)

    human_dir = os.path.join(os.path.dirname(__file__), "datasets", "human")
    human_csvs = glob.glob(os.path.join(human_dir, "*.csv"))
    if not human_csvs:
        human_csvs = glob.glob(os.path.join(os.path.dirname(__file__), "datasets", "*.csv"))

    human_X = sanitize_and_extract_human_features(human_csvs)
    if not human_X:
        print("[Error] No valid human feature vectors extracted!")
        return

    if len(human_X) > 1600:
        np.random.seed(42)
        indices = np.random.choice(len(human_X), 1600, replace=False)
        human_X = [human_X[i] for i in indices]

    bot_X = generate_and_extract_bot_features(n_runs_per_bot=200)

    X = np.array(human_X + bot_X)
    y = np.array([0] * len(human_X) + [1] * len(bot_X))

    print("\n" + "=" * 65)
    print(f"[Dataset Summary] Total Samples: {len(X):,} | Human (0): {len(human_X):,} | Bot (1): {len(bot_X):,}")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("\n[Model Training] Fitting Random Forest Classifier (150 Estimators, max_depth=10)...")
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    y_proba = rf_clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f" MODEL ACCURACY: {acc * 100:.2f}% | ROC-AUC SCORE: {roc_auc:.4f}")
    print("=" * 65)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "Bot (1)"]))

    print("--- Confusion Matrix ---")
    print(f" True Human (Passed):   {cm[0][0]:<4} | False Bot (Blocked): {cm[0][1]:<4}")
    print(f" False Human (Passed):  {cm[1][0]:<4} | True Bot (Blocked):  {cm[1][1]:<4}")

    feature_names = [
        "Straightness Index", "Velocity Mean", "Velocity Std", "Velocity Max",
        "Acceleration Mean", "Acceleration Std", "Jerk Mean", "Angle Change Mean",
        "Angle Change Std", "Jitter Ratio", "Idle Ratio", "Click Dwell Ms"
    ]
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- Biometric Feature Importance Ranking ---")
    for rank in range(len(feature_names)):
        idx = indices[rank]
        print(f" {rank+1:2d}. {feature_names[idx]:<22}: {importances[idx] * 100:.2f}%")

    model_path = os.path.join(os.path.dirname(__file__), "backend", "behavioral_rf_model.joblib")
    joblib.dump(rf_clf, model_path)
    print(f"\n[Saver] Production trained model exported to: [file:///{model_path.replace(os.sep, '/')}]")

if __name__ == "__main__":
    main()
