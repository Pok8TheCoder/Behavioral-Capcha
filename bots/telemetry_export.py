"""Normalize bot event timestamps for CSV export and training windows."""

from __future__ import annotations

import random
from typing import Any, Dict, List


def ensure_export_timestamps(
    events: List[Dict[str, Any]],
    rng: random.Random,
    min_step_ms: float = 1.0,
    max_step_ms: float = 12.0,
    min_duration_ms: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Fast dashboard bots often record every event at t≈0. Spread timestamps
    monotonically so dynamic windows are valid while preserving bot kinematics.
    """
    if not events:
        return events

    duration = max(float(e["t"]) for e in events) - min(float(e["t"]) for e in events)
    if duration >= min_duration_ms:
        return events

    stamped: List[Dict[str, Any]] = []
    t = 0.0
    for event in events:
        stamped.append({**event, "t": round(t, 3)})
        t += rng.uniform(min_step_ms, max_step_ms)
    return stamped


def ensure_60hz_export_events(
    events: List[Dict[str, Any]],
    rng: random.Random,
    *,
    min_step_ms: float = 14.0,
    max_step_ms: float = 18.0,
) -> List[Dict[str, Any]]:
    """Spread collapsed bot timestamps, then downsample to 60 Hz for Xmodel training."""
    from backend.hz60 import downsample_events_to_hz

    stamped = ensure_export_timestamps(
        events,
        rng,
        min_step_ms=min_step_ms,
        max_step_ms=max_step_ms,
    )
    return downsample_events_to_hz(stamped, 60.0)
