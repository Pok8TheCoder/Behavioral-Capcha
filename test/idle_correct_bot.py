"""Synthetic idle_correct family (Issue 010 vs Qmodel V0.11)."""

from __future__ import annotations

from test.sparse_irregular_bot import SparseConfig, generate_events  # noqa: F401

FAMILY_PRESETS = {
    "idle_correct": SparseConfig(
        spacing_px=(8.0, 15.0), min_points=12, max_points=32,
        arc_px=(10.0, 26.0), tremor_px=1.25, aim_sigma=2.4,
        dwell_ms=(80.0, 160.0), pause_ms=(120.0, 280.0),
        s_curve=True, irregular=True, name="idle_correct",
    ),
    "wide_arc": SparseConfig(
        spacing_px=(7.0, 13.0), min_points=16, max_points=42,
        arc_px=(18.0, 40.0), tremor_px=1.4, aim_sigma=2.2,
        s_curve=True, irregular=True, name="wide_arc",
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
        "straightness_index": 0.0964,
        "jitter_ratio": 0.1739,
        "velocity_mean": 1.8562,
        "velocity_max": 29.2914,
        "jerk_mean": 0.57256681,
        "acceleration_mean": 0.817513,
        "event_count": 163,
        "duration_ms": 5807.9,
        "avg_click_dwell_ms": 132.32,
        "idle_ratio": 0.4602,
    }
