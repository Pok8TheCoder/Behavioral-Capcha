"""Synthetic wobble_mid family (Issue 014 vs Qmodel V0.15)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "wobble_mid": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=14, max_points=38,
        arc_px=(18.0, 38.0), tremor_px=1.8, aim_sigma=2.3,
        pause_ms=(60.0, 140.0), name="wobble_mid",
    ),
    "mid_str_fast": SparseConfig(
        spacing_px=(10.0, 17.0), min_points=12, max_points=32,
        arc_px=(5.0, 14.0), tremor_px=1.0, aim_sigma=1.8,
        s_curve=False, name="mid_str_fast",
    ),
    "dense_slow": SparseConfig(
        spacing_px=(5.5, 10.0), min_points=24, max_points=64,
        arc_px=(8.0, 20.0), tremor_px=0.85, hop_dt_ms=(10.0, 28.0),
        irregular=False, name="dense_slow",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.0681,
        "jitter_ratio": 0.2298,
        "velocity_mean": 1.139,
        "velocity_max": 16.7862,
        "jerk_mean": 0.2468964,
        "acceleration_mean": 0.360127,
        "event_count": 163,
        "duration_ms": 4431.5,
        "avg_click_dwell_ms": 89.55,
        "idle_ratio": 0.2317,
    }
