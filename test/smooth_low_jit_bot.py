"""Synthetic smooth_low_jit family (Issue 012 vs Qmodel V0.13)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "smooth_low_jit": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=16, max_points=40,
        arc_px=(6.0, 14.0), tremor_px=0.4, aim_sigma=1.5,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 100.0),
        s_curve=True, irregular=False, name="smooth_low_jit",
    ),
    "burst_hi_vel": SparseConfig(
        spacing_px=(11.0, 18.0), min_points=10, max_points=24,
        arc_px=(8.0, 18.0), tremor_px=1.0, aim_sigma=1.8,
        dwell_ms=(40.0, 80.0), pause_ms=(40.0, 80.0),
        s_curve=False, irregular=True, name="burst_hi_vel",
    ),
    "wide_low_jerk": SparseConfig(
        spacing_px=(9.0, 15.0), min_points=14, max_points=36,
        arc_px=(20.0, 42.0), tremor_px=1.1, aim_sigma=2.2,
        dwell_ms=(60.0, 120.0), pause_ms=(60.0, 120.0),
        s_curve=True, irregular=True, name="wide_low_jerk",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.0794,
        "jitter_ratio": 0.0973,
        "velocity_mean": 1.1995,
        "velocity_max": 18.893,
        "jerk_mean": 0.30545036,
        "acceleration_mean": 0.444383,
        "event_count": 115,
        "duration_ms": 3095.4,
        "avg_click_dwell_ms": 80.65,
        "idle_ratio": 0.2012,
    }
