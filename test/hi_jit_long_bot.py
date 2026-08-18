"""Synthetic hi_jit_long family (Issue 013 vs Qmodel V0.14)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "hi_jit_long": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=20, max_points=50,
        arc_px=(16.0, 36.0), tremor_px=2.6, aim_sigma=2.5,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 100.0),
        s_curve=True, irregular=True, name="hi_jit_long",
    ),
    "long_trace": SparseConfig(
        spacing_px=(5.0, 9.0), min_points=28, max_points=70,
        arc_px=(12.0, 28.0), tremor_px=1.15, aim_sigma=2.1,
        dwell_ms=(40.0, 90.0), pause_ms=(40.0, 90.0),
        s_curve=True, irregular=True, name="long_trace",
    ),
    "low_dyn": SparseConfig(
        spacing_px=(9.0, 16.0), min_points=14, max_points=36,
        arc_px=(8.0, 18.0), tremor_px=0.9, aim_sigma=1.7,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 100.0),
        s_curve=False, irregular=True, name="low_dyn",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.052,
        "jitter_ratio": 0.288,
        "velocity_mean": 1.2276,
        "velocity_max": 20.0998,
        "jerk_mean": 0.30171774,
        "acceleration_mean": 0.473234,
        "event_count": 252,
        "duration_ms": 6271.3,
        "avg_click_dwell_ms": 83.13,
        "idle_ratio": 0.1756,
    }
