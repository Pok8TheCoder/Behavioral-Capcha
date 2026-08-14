"""
Human-likeness anomaly trainer for Behavioral CAPTCHA.

This script trains an IsolationForest on cleaned human mouse telemetry only.
AFK time is removed during preprocessing by splitting active episodes whenever
meaningful mouse movement is absent for more than 5 seconds. Active episodes
are then converted into 3 second windows and featurized with the backend
feature extractor.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from backend.feature_extractor import extract_features, feature_dict_to_vector


ROOT_DIR = os.path.dirname(__file__)
HUMAN_DIR = os.path.join(ROOT_DIR, "datasets", "human")
BOT_DIR = os.path.join(ROOT_DIR, "datasets", "bot")
MODEL_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_iforest_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "backend", "behavioral_iforest_metadata.json")

AFK_GAP_MS = 5_000.0
WINDOW_MS = 3_000.0
STRIDE_MS = 1_500.0
MIN_EVENTS_PER_WINDOW = 12
MIN_WINDOW_DISTANCE_PX = 20.0
MOVEMENT_EPSILON_PX = 0.5
CHUNKSIZE = 200_000
MAX_WINDOWS_PER_HUMAN_FILE = 6_000
RANDOM_STATE = 42


FEATURE_NAMES = [
    "straightness_index",
    "velocity_mean",
    "velocity_std",
    "velocity_max",
    "acceleration_mean",
    "acceleration_std",
    "jerk_mean",
    "angle_change_mean",
    "angle_change_std",
    "jitter_ratio",
    "idle_ratio",
    "avg_click_dwell_ms",
]


@dataclass
class WindowSample:
    vector: List[float]
    source_file: str


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalize_event(row: pd.Series, time_col: str, x_col: str, y_col: str, evt_col: Optional[str]) -> Dict[str, Any]:
    event_type = row[evt_col] if evt_col else "mouse_move"
    return {
        "t": float(row[time_col]),
        "x": float(row[x_col]),
        "y": float(row[y_col]),
        "type": str(event_type),
    }


def _event_distance(events: Sequence[Dict[str, Any]]) -> float:
    if len(events) < 2:
        return 0.0
    xs = np.array([e["x"] for e in events], dtype=float)
    ys = np.array([e["y"] for e in events], dtype=float)
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _window_episode(events: List[Dict[str, Any]], source_file: str) -> List[WindowSample]:
    if len(events) < MIN_EVENTS_PER_WINDOW:
        return []

    events = sorted(events, key=lambda event: event["t"])
    start_t = events[0]["t"]
    end_t = events[-1]["t"]
    samples: List[WindowSample] = []

    if end_t - start_t <= WINDOW_MS:
        if _event_distance(events) >= MIN_WINDOW_DISTANCE_PX:
            samples.append(WindowSample(feature_dict_to_vector(extract_features(events)), source_file))
        return samples

    cur_t = start_t
    while cur_t + WINDOW_MS <= end_t:
        window = [e for e in events if cur_t <= e["t"] < cur_t + WINDOW_MS]
        if len(window) >= MIN_EVENTS_PER_WINDOW and _event_distance(window) >= MIN_WINDOW_DISTANCE_PX:
            samples.append(WindowSample(feature_dict_to_vector(extract_features(window)), source_file))
        cur_t += STRIDE_MS

    return samples


def _subsample_per_file(samples: List[WindowSample], max_windows: int) -> List[WindowSample]:
    if len(samples) <= max_windows:
        return samples
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.sort(rng.choice(len(samples), size=max_windows, replace=False))
    return [samples[int(i)] for i in indices]


def extract_active_windows_from_csv(file_path: str, max_windows: int = MAX_WINDOWS_PER_HUMAN_FILE) -> Tuple[List[WindowSample], Dict[str, Any]]:
    samples: List[WindowSample] = []
    episode: List[Dict[str, Any]] = []
    last_active_t: Optional[float] = None
    last_x: Optional[float] = None
    last_y: Optional[float] = None
    raw_rows = 0
    afk_splits = 0

    try:
        preview = pd.read_csv(file_path, comment="#", nrows=0)
    except Exception as exc:
        return [], {"file": file_path, "error": str(exc)}

    time_col = _find_col(preview.columns, ["rel_time_ms", "t", "timestamp_ms"])
    x_col = _find_col(preview.columns, ["x_px", "x"])
    y_col = _find_col(preview.columns, ["y_px", "y"])
    evt_col = _find_col(preview.columns, ["event_type", "type"])

    if not (time_col and x_col and y_col):
        return [], {"file": file_path, "error": "missing required time/x/y columns"}

    usecols = [time_col, x_col, y_col]
    if evt_col:
        usecols.append(evt_col)

    for chunk in pd.read_csv(file_path, comment="#", usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk.dropna(subset=[time_col, x_col, y_col])
        raw_rows += len(chunk)
        if chunk.empty:
            continue

        if evt_col:
            event_types = chunk[evt_col].astype(str).str.lower().str.replace("_", "", regex=False)
        else:
            event_types = pd.Series(["mousemove"] * len(chunk), index=chunk.index)

        x_values = chunk[x_col].astype(float)
        y_values = chunk[y_col].astype(float)
        dx = x_values.diff()
        dy = y_values.diff()
        if last_x is not None and last_y is not None:
            first_idx = chunk.index[0]
            dx.loc[first_idx] = x_values.loc[first_idx] - last_x
            dy.loc[first_idx] = y_values.loc[first_idx] - last_y

        moved = np.hypot(dx.fillna(0.0), dy.fillna(0.0)) >= MOVEMENT_EPSILON_PX
        if last_x is None:
            moved.iloc[0] = True

        movement_mask = event_types.isin(["mousemove", "0"]) & moved
        click_mask = event_types.isin(["mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"])
        filtered = chunk[movement_mask | click_mask].copy()
        filtered["event_type_normalized"] = event_types.loc[filtered.index]

        for row in filtered.itertuples(index=False):
            values = row._asdict()
            event = {
                "t": float(values[time_col]),
                "x": float(values[x_col]),
                "y": float(values[y_col]),
                "type": str(values.get(evt_col, "mouse_move")) if evt_col else "mouse_move",
            }
            event_type = values["event_type_normalized"]
            is_movement = event_type in {"mousemove", "0"}
            is_click = event_type in {"mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"}

            if is_movement:
                if last_active_t is not None and event["t"] - last_active_t > AFK_GAP_MS:
                    samples.extend(_window_episode(episode, os.path.basename(file_path)))
                    episode = []
                    afk_splits += 1
                    if len(samples) >= max_windows:
                        return _subsample_per_file(samples, max_windows), {
                            "file": file_path,
                            "raw_rows": raw_rows,
                            "windows": max_windows,
                            "afk_splits": afk_splits,
                            "truncated": True,
                        }

                episode.append(event)
                last_active_t = event["t"]
            elif is_click and episode:
                episode.append(event)

        last_x = float(x_values.iloc[-1])
        last_y = float(y_values.iloc[-1])

    samples.extend(_window_episode(episode, os.path.basename(file_path)))
    samples = _subsample_per_file(samples, max_windows)
    return samples, {
        "file": file_path,
        "raw_rows": raw_rows,
        "windows": len(samples),
        "afk_splits": afk_splits,
        "truncated": False,
    }


def extract_windows_from_bot_csv(file_path: str) -> List[List[float]]:
    samples, _ = extract_active_windows_from_csv(file_path, max_windows=2_000)
    return [sample.vector for sample in samples]


def load_human_samples() -> Tuple[List[WindowSample], List[Dict[str, Any]]]:
    human_csvs = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.csv")))
    all_samples: List[WindowSample] = []
    summaries: List[Dict[str, Any]] = []

    print(f"[Data] Found {len(human_csvs)} human CSV files.")
    for file_path in human_csvs:
        samples, summary = extract_active_windows_from_csv(file_path)
        all_samples.extend(samples)
        summaries.append(summary)
        print(
            f"[Data] {os.path.basename(file_path)}: "
            f"{summary.get('windows', 0):,} windows, "
            f"{summary.get('afk_splits', 0):,} AFK splits"
        )

    return all_samples, summaries


def load_bot_vectors() -> List[List[float]]:
    bot_csvs = sorted(glob.glob(os.path.join(BOT_DIR, "*.csv")))
    vectors: List[List[float]] = []
    for file_path in bot_csvs:
        file_vectors = extract_windows_from_bot_csv(file_path)
        vectors.extend(file_vectors)
        print(f"[Eval] {os.path.basename(file_path)}: {len(file_vectors):,} bot windows")
    return vectors


def split_human_by_session(samples: List[WindowSample]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([sample.vector for sample in samples], dtype=float)
    groups = np.array([sample.source_file for sample in samples])

    if len(set(groups)) < 2 or len(X) < 20:
        split = int(max(1, len(X) * 0.8))
        return X[:split], X[split:]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, groups=groups))
    return X[train_idx], X[test_idx]


def main() -> None:
    print("=" * 72)
    print("BEHAVIORAL CAPTCHA ISOLATION FOREST TRAINER")
    print("=" * 72)

    human_samples, human_summaries = load_human_samples()
    if len(human_samples) < 30:
        raise RuntimeError(f"Not enough active human windows for training: {len(human_samples)}")

    X_train, X_human_test = split_human_by_session(human_samples)

    contamination = min(0.08, max(0.01, 20.0 / len(X_train)))
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=500,
                    contamination=contamination,
                    max_samples="auto",
                    bootstrap=False,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(f"[Train] Human train windows: {len(X_train):,}")
    print(f"[Train] Held-out human windows: {len(X_human_test):,}")
    print(f"[Train] Contamination: {contamination:.4f}")
    model.fit(X_train)

    bot_vectors = load_bot_vectors()
    metrics: Dict[str, Any] = {
        "human_train_windows": int(len(X_train)),
        "human_test_windows": int(len(X_human_test)),
        "bot_eval_windows": int(len(bot_vectors)),
        "contamination": float(contamination),
        "feature_names": FEATURE_NAMES,
        "preprocessing": {
            "afk_gap_ms": AFK_GAP_MS,
            "window_ms": WINDOW_MS,
            "stride_ms": STRIDE_MS,
            "min_events_per_window": MIN_EVENTS_PER_WINDOW,
            "min_window_distance_px": MIN_WINDOW_DISTANCE_PX,
            "movement_epsilon_px": MOVEMENT_EPSILON_PX,
        },
        "human_files": human_summaries,
    }

    if len(X_human_test) > 0:
        human_pred = model.predict(X_human_test)
        human_pass_rate = float(np.mean(human_pred == 1))
        metrics["heldout_human_pass_rate"] = human_pass_rate
        print(f"[Eval] Held-out human pass rate: {human_pass_rate * 100:.2f}%")

    if bot_vectors and len(X_human_test) > 0:
        X_bot = np.array(bot_vectors, dtype=float)
        y_true = np.array([0] * len(X_human_test) + [1] * len(X_bot))
        X_eval = np.vstack([X_human_test, X_bot])
        predictions = model.predict(X_eval)
        y_pred = np.where(predictions == -1, 1, 0)
        anomaly_scores = -model.decision_function(X_eval)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=["Human", "Bot"], zero_division=0)
        try:
            auc = float(roc_auc_score(y_true, anomaly_scores))
        except ValueError:
            auc = None

        metrics["bot_eval_confusion_matrix"] = cm.tolist()
        metrics["bot_eval_roc_auc"] = auc
        metrics["bot_eval_report"] = report

        print("[Eval] Human/Bot confusion matrix:")
        print(cm)
        if auc is not None:
            print(f"[Eval] ROC-AUC using anomaly score: {auc:.4f}")
        print("[Eval] Classification report:")
        print(report)

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "metadata": metrics}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Save] Model: {MODEL_PATH}")
    print(f"[Save] Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
