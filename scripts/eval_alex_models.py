"""Score Alex's 11-feature joblibs on our human vs bot windows."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_isolation_forest import WindowSample, extract_active_windows_from_csv
from train_isolation_forest import _window_episode as orig_window_episode  # noqa: F401
import train_isolation_forest as tif
from train_qmodel_v081g import load_human_os_and_dashboard
from train_random_forest_balanced import BOT_DIRS_ALL, load_labeled_windows, session_split

ALEX_DIR = os.path.join(ROOT, "Alex", "Models")
ALEX_NAMES = [
    "v_mean",
    "v_max",
    "v_std",
    "a_mean",
    "a_max",
    "jerk_std",
    "idle_time_ratio",
    "micro_jitter_ratio",
    "straightness_index",
    "tremor_ratio",
    "click_rate",
]
FAMILY_DIR = os.path.join(ROOT, "datasets", "bot_families")


def _event_type(event: Dict[str, Any]) -> str:
    return str(event.get("type", "mousemove")).lower().replace("_", "")


def extract_alex_vector(events: Sequence[Dict[str, Any]]) -> List[float]:
    if not events or len(events) < 2:
        return [0.0] * 11

    events = sorted(events, key=lambda e: float(e["t"]))
    duration_s = max(1e-3, (float(events[-1]["t"]) - float(events[0]["t"])) / 1000.0)

    xs, ys, ts = [], [], []
    n_clicks = 0
    for event in events:
        raw = _event_type(event)
        xs.append(float(event["x"]))
        ys.append(float(event["y"]))
        ts.append(float(event["t"]) / 1000.0)
        if raw in ("mousedown", "mouseclickdown", "1"):
            n_clicks += 1

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    t = np.asarray(ts, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    valid = dt > 1e-6
    if not np.any(valid):
        return [0.0] * 11
    dx, dy, dt = dx[valid], dy[valid], dt[valid]
    dist = np.hypot(dx, dy)
    speed = dist / dt  # px/s
    path = float(np.sum(dist))
    chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    tortuosity = path / max(chord, 1e-6)

    if len(speed) > 1:
        acc = np.diff(speed) / dt[1:]
        a_mean = float(np.mean(acc))
        a_max = float(np.max(np.abs(acc)))
    else:
        acc = np.array([])
        a_mean, a_max = 0.0, 0.0

    if len(acc) > 1 and len(dt) > 2:
        n = min(len(acc) - 1, len(dt) - 2)
        jerk = np.diff(acc)[:n] / dt[2 : 2 + n]
        jerk_std = float(np.std(jerk)) if n else 0.0
    else:
        jerk_std = 0.0

    idle = float(np.sum(dt[speed < 80.0]) / max(float(np.sum(dt)), 1e-6))
    if len(dx) > 1:
        angles = np.arctan2(dy, dx)
        diffs = np.abs(np.diff(angles))
        diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
        micro_jitter = float(np.mean(diffs > (np.pi / 4.0)))
    else:
        micro_jitter = 0.0

    tremor = _tremor_ratio(t, x, y)
    return [
        float(np.mean(speed)),
        float(np.max(speed)),
        float(np.std(speed)) if len(speed) > 1 else 0.0,
        a_mean,
        a_max,
        jerk_std,
        float(np.clip(idle, 0.0, 1.0)),
        micro_jitter,
        tortuosity,
        tremor,
        n_clicks / duration_s,
    ]


def _tremor_ratio(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 8:
        return 0.0
    duration = float(t[-1] - t[0])
    if duration < 0.08:
        return 0.0
    hz = 100.0
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    if len(grid) < 16:
        return 0.0
    xi = np.interp(grid, t, x)
    yi = np.interp(grid, t, y)
    sig = np.hypot(np.diff(xi), np.diff(yi))
    spec = np.abs(np.fft.rfft(sig - np.mean(sig))) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / hz)
    total = float(np.sum(spec[1:]))
    if total <= 0:
        return 0.0
    band = (freqs >= 7.0) & (freqs <= 15.0)
    return float(np.sum(spec[band]) / total)


def _alex_window_episode(events, source_file, target_hz=None):
    from backend.hz60 import downsample_events_to_hz
    from backend.active_window import DEFAULT_ACTIVE_WINDOW_CONFIG, segment_episode_into_windows

    if target_hz:
        events = downsample_events_to_hz(events, float(target_hz))
    if len(events) < 4:
        return []
    samples = []
    for window in segment_episode_into_windows(events, DEFAULT_ACTIVE_WINDOW_CONFIG):
        samples.append(WindowSample(extract_alex_vector(window), source_file))
    return samples


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).ravel()
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi <= lo:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)
    return np.asarray(model.predict(X), dtype=float)


def _orient(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, bool]:
    auc = float(roc_auc_score(y_true, scores))
    if auc >= 0.5:
        return scores, False
    return 1.0 - scores, True


def score_model(name: str, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    raw = _predict_proba(model, X)
    scores, flipped = _orient(y, raw)
    pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "name": name,
        "auc": float(roc_auc_score(y, scores)),
        "accuracy": float((tn + tp) / len(y)),
        "flipped_labels": flipped,
        "confusion": cm.tolist(),
        "human_as_bot_rate": float(fp / max(tn + fp, 1)),
        "bot_miss_rate": float(fn / max(fn + tp, 1)),
        "mean_bot_score_on_humans": float(np.mean(scores[y == 0])),
        "mean_bot_score_on_bots": float(np.mean(scores[y == 1])),
        "ml_human_on_humans": float(1.0 - np.mean(scores[y == 0])),
        "ml_human_on_bots": float(1.0 - np.mean(scores[y == 1])),
        "report": classification_report(y, pred, target_names=["Human", "Bot"], zero_division=0),
    }


def main() -> None:
    tif._window_episode = _alex_window_episode
    print("[Alex] Loading windows in Alex 11-feature space (native Hz)")
    human_X, human_groups, _, human_counts = load_human_os_and_dashboard()
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    for prefix, bot_dir in BOT_DIRS_ALL + [("bot_families", FAMILY_DIR)]:
        vectors, groups, _ = load_labeled_windows(bot_dir, prefix)
        bot_X.extend(vectors)
        bot_groups.extend(groups)

    X = np.array(human_X + bot_X, dtype=np.float32)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=np.int32)
    groups = np.array(human_groups + bot_groups)
    _, X_test, _, y_test = session_split(X, y, groups)
    print(f"[Alex] Human windows {len(human_X):,}  Bot {len(bot_X):,}  Test {len(X_test):,}")
    print(f"[Alex] Feature means (test): {dict(zip(ALEX_NAMES, np.mean(X_test, axis=0).tolist()))}")
    print(f"[Alex] Human source counts: {human_counts}")

    rf = joblib.load(os.path.join(ALEX_DIR, "random_forest.joblib"))
    svm = joblib.load(os.path.join(ALEX_DIR, "svm_rbf.joblib"))
    xgb = joblib.load(os.path.join(ALEX_DIR, "xgboost.joblib"))
    scaler = joblib.load(os.path.join(ALEX_DIR, "scaler.joblib"))
    X_scaled = scaler.transform(X_test)

    results = [
        score_model("random_forest", rf, X_test, y_test),
        score_model("random_forest + scaler", rf, X_scaled, y_test),
        score_model("svm_rbf + scaler", svm, X_scaled, y_test),
        score_model("xgboost", xgb, X_test, y_test),
        score_model("xgboost + scaler", xgb, X_scaled, y_test),
    ]
    for row in results:
        print("=" * 72)
        print(row["name"])
        print(f"ROC-AUC {row['auc']:.4f}  Acc {row['accuracy']:.4f}  flipped={row['flipped_labels']}")
        print(f"Confusion {row['confusion']}")
        print(f"ML-human on humans {row['ml_human_on_humans']:.3f}  on bots {row['ml_human_on_bots']:.3f}")
        print(row["report"])

    out = os.path.join(ALEX_DIR, "eval_on_our_windows.json")
    import json
    dump = [{k: v for k, v in row.items() if k != "report"} | {"report": row["report"]} for row in results]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_windows": int(len(X_test)),
                "human_windows": int(len(human_X)),
                "bot_windows": int(len(bot_X)),
                "feature_names": ALEX_NAMES,
                "results": dump,
            },
            f,
            indent=2,
        )
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
