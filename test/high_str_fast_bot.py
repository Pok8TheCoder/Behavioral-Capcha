"""Synthetic high_str_fast family (Issue 009 vs Qmodel V0.10)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "high_str_fast": SparseConfig(
        spacing_px=(14.0, 24.0), min_points=8, max_points=20,
        arc_px=(2.0, 8.0), tremor_px=0.7, aim_sigma=1.6,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 120.0),
        s_curve=False, irregular=False, name="high_str_fast",
    ),
    "fast_low_jit": SparseConfig(
        spacing_px=(12.0, 20.0), min_points=10, max_points=28,
        arc_px=(4.0, 12.0), tremor_px=0.45, aim_sigma=1.5,
        dwell_ms=(40.0, 90.0), pause_ms=(40.0, 100.0),
        s_curve=False, irregular=False, name="fast_low_jit",
    ),
    "mid_str": SparseConfig(
        spacing_px=(9.0, 16.0), min_points=14, max_points=36,
        arc_px=(6.0, 16.0), tremor_px=1.1, aim_sigma=1.9,
        s_curve=False, irregular=True, name="mid_str",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.1056,
        "jitter_ratio": 0.2034,
        "velocity_mean": 2.0278,
        "velocity_max": 23.4458,
        "jerk_mean": 0.53980913,
        "acceleration_mean": 0.905019,
        "event_count": 61,
        "duration_ms": 2013.1,
        "avg_click_dwell_ms": 85.55,
        "idle_ratio": 0.2879,
    }
