"""Synthetic hi_jit family (Issue 011 vs Qmodel V0.12)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "hi_jit": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=14, max_points=38,
        arc_px=(14.0, 32.0), tremor_px=2.8, aim_sigma=2.6,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 120.0),
        s_curve=True, irregular=True, name="hi_jit",
    ),
    "low_idle_fast": SparseConfig(
        spacing_px=(7.0, 12.0), min_points=18, max_points=48,
        arc_px=(8.0, 20.0), tremor_px=1.15, aim_sigma=2.0,
        dwell_ms=(30.0, 70.0), pause_ms=(30.0, 70.0),
        s_curve=True, irregular=True, name="low_idle_fast",
    ),
    "high_str": SparseConfig(
        spacing_px=(10.0, 18.0), min_points=12, max_points=30,
        arc_px=(3.0, 8.0), tremor_px=0.55, aim_sigma=1.4,
        dwell_ms=(40.0, 90.0), pause_ms=(40.0, 90.0),
        s_curve=False, irregular=False, name="high_str",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.2964,
        "jitter_ratio": 0.3162,
        "velocity_mean": 1.2298,
        "velocity_max": 14.7573,
        "jerk_mean": 0.28420734,
        "acceleration_mean": 0.449907,
        "event_count": 119,
        "duration_ms": 3281.3,
        "avg_click_dwell_ms": 79.97,
        "idle_ratio": 0.1693,
    }
