"""Synthetic mid_count family (Issue 008 vs Qmodel V0.9)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "mid_count": SparseConfig(
        spacing_px=(8.0, 14.0), min_points=18, max_points=48,
        arc_px=(8.0, 22.0), tremor_px=1.4, aim_sigma=2.1,
        dwell_ms=(50.0, 110.0), pause_ms=(50.0, 120.0),
        s_curve=True, irregular=True, name="mid_count",
    ),
    "high_str": SparseConfig(
        spacing_px=(14.0, 24.0), min_points=8, max_points=20,
        arc_px=(2.0, 8.0), tremor_px=0.7, aim_sigma=1.6,
        dwell_ms=(50.0, 100.0), pause_ms=(50.0, 120.0),
        s_curve=False, irregular=False, name="high_str",
    ),
    "low_jit_dense": SparseConfig(
        spacing_px=(6.0, 11.0), min_points=22, max_points=55,
        arc_px=(10.0, 22.0), tremor_px=0.55, aim_sigma=1.5,
        dwell_ms=(60.0, 120.0), pause_ms=(60.0, 140.0),
        s_curve=False, irregular=False, name="low_jit_dense",
    ),
}
FAMILY_NAMES = list(FAMILY_PRESETS.keys())


def winning_feature_sample() -> dict[str, float]:
    return {
        "straightness_index": 0.0282,
        "jitter_ratio": 0.2,
        "velocity_mean": 1.3047,
        "velocity_max": 23.2555,
        "jerk_mean": 0.33883843,
        "acceleration_mean": 0.501202,
        "event_count": 127,
        "duration_ms": 3480.4,
        "avg_click_dwell_ms": 92.67,
        "idle_ratio": 0.2853,
    }
