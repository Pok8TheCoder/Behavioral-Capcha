"""Alex 11-feature schema (third-party models in Alex/Models)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

ALEX_FEATURE_NAMES = [
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


def _event_type(event: Dict[str, Any]) -> str:
    return str(event.get("type", "mousemove")).lower().replace("_", "")


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


def extract_alex_features(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    empty = {name: 0.0 for name in ALEX_FEATURE_NAMES}
    empty.update({
        "event_count": len(events) if events else 0,
        "duration_ms": 0.0,
        "total_distance_px": 0.0,
        "velocity_mean": 0.0,
        "velocity_std": 0.0,
        "velocity_max": 0.0,
        "jitter_ratio": 0.0,
        "idle_ratio": 0.0,
        "avg_click_dwell_ms": -1.0,
        "jerk_mean": 0.0,
        "acceleration_mean": 0.0,
        "feature_schema": "alex11",
    })
    if not events or len(events) < 2:
        return empty

    events = sorted(events, key=lambda e: float(e["t"]))
    duration_ms = max(1.0, float(events[-1]["t"]) - float(events[0]["t"]))
    duration_s = duration_ms / 1000.0

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
        empty["event_count"] = len(events)
        empty["duration_ms"] = round(duration_ms, 2)
        return empty
    dx, dy, dt = dx[valid], dy[valid], dt[valid]
    dist = np.hypot(dx, dy)
    speed = dist / dt
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

    idle = float(np.clip(np.sum(dt[speed < 80.0]) / max(float(np.sum(dt)), 1e-6), 0.0, 1.0))
    if len(dx) > 1:
        angles = np.arctan2(dy, dx)
        diffs = np.abs(np.diff(angles))
        diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
        micro_jitter = float(np.mean(diffs > (np.pi / 4.0)))
    else:
        micro_jitter = 0.0

    v_mean = float(np.mean(speed))
    v_std = float(np.std(speed)) if len(speed) > 1 else 0.0
    v_max = float(np.max(speed))
    click_rate = n_clicks / max(duration_s, 1e-3)
    tremor = _tremor_ratio(t, x, y)
    return {
        "event_count": len(events),
        "duration_ms": round(duration_ms, 2),
        "total_distance_px": round(path, 2),
        "v_mean": v_mean,
        "v_max": v_max,
        "v_std": v_std,
        "a_mean": a_mean,
        "a_max": a_max,
        "jerk_std": jerk_std,
        "idle_time_ratio": idle,
        "micro_jitter_ratio": micro_jitter,
        "straightness_index": tortuosity,
        "tremor_ratio": tremor,
        "click_rate": click_rate,
        "velocity_mean": v_mean,
        "velocity_std": v_std,
        "velocity_max": v_max,
        "jitter_ratio": micro_jitter,
        "idle_ratio": idle,
        "avg_click_dwell_ms": -1.0,
        "jerk_mean": 0.0,
        "acceleration_mean": a_mean,
        "feature_schema": "alex11",
    }


def alex_dict_to_vector(features: Dict[str, Any]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in ALEX_FEATURE_NAMES]
