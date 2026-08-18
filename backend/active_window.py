"""
Dynamic active-window extraction for behavioral telemetry.

Instead of fixed 3s slices, windows are built from natural movement structure:
  1. AFK gaps (>5s) split sessions into episodes (CSV ingestion).
  2. Pause gaps (>pause_gap_ms) split episodes into movement bursts.
  3. Bursts shorter than max_window_ms become one sample; longer bursts are
     split on stronger pauses or a sliding fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.feature_extractor import extract_features, feature_dict_to_vector


@dataclass(frozen=True)
class ActiveWindowConfig:
    afk_gap_ms: float = 5_000.0
    pause_gap_ms: float = 400.0
    strong_pause_gap_ms: float = 900.0
    min_window_ms: float = 500.0
    max_window_ms: float = 12_000.0
    fallback_window_ms: float = 3_000.0
    fallback_stride_ms: float = 1_500.0
    min_events_per_window: int = 12
    min_window_distance_px: float = 20.0
    movement_epsilon_px: float = 0.5


DEFAULT_ACTIVE_WINDOW_CONFIG = ActiveWindowConfig()


@dataclass
class WindowSample:
    vector: List[float]
    source_file: str
    window_ms: float
    event_count: int
    mode: str


def _event_distance(events: Sequence[Dict[str, Any]]) -> float:
    if len(events) < 2:
        return 0.0
    xs = np.array([float(e["x"]) for e in events], dtype=float)
    ys = np.array([float(e["y"]) for e in events], dtype=float)
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _is_movement_event(raw_type: str) -> bool:
    normalized = raw_type.lower().replace("_", "")
    return normalized in {"mousemove", "0"}


def _is_click_event(raw_type: str) -> bool:
    normalized = raw_type.lower().replace("_", "")
    return normalized in {"mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"}


def _valid_window(events: List[Dict[str, Any]], cfg: ActiveWindowConfig) -> bool:
    if len(events) < cfg.min_events_per_window:
        return False
    duration = float(events[-1]["t"]) - float(events[0]["t"])
    if duration < cfg.min_window_ms:
        return False
    return _event_distance(events) >= cfg.min_window_distance_px


def _split_on_gaps(
    events: List[Dict[str, Any]],
    gap_ms: float,
) -> List[List[Dict[str, Any]]]:
    if not events:
        return []
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [events[0]]
    for event in events[1:]:
        if float(event["t"]) - float(current[-1]["t"]) > gap_ms:
            segments.append(current)
            current = [event]
        else:
            current.append(event)
    if current:
        segments.append(current)
    return segments


def _sliding_windows(
    events: List[Dict[str, Any]],
    cfg: ActiveWindowConfig,
) -> List[List[Dict[str, Any]]]:
    start_t = float(events[0]["t"])
    end_t = float(events[-1]["t"])
    if end_t - start_t <= cfg.fallback_window_ms:
        return [events] if _valid_window(events, cfg) else []

    windows: List[List[Dict[str, Any]]] = []
    cur_t = start_t
    while cur_t + cfg.fallback_window_ms <= end_t:
        window = [
            e for e in events
            if cur_t <= float(e["t"]) < cur_t + cfg.fallback_window_ms
        ]
        if _valid_window(window, cfg):
            windows.append(window)
        cur_t += cfg.fallback_stride_ms
    return windows


def segment_episode_into_windows(
    events: List[Dict[str, Any]],
    cfg: ActiveWindowConfig = DEFAULT_ACTIVE_WINDOW_CONFIG,
) -> List[List[Dict[str, Any]]]:
    """Split one active episode into variable-length movement windows."""
    if len(events) < cfg.min_events_per_window:
        return []

    events = sorted(events, key=lambda e: float(e["t"]))
    duration_ms = float(events[-1]["t"]) - float(events[0]["t"])

    if duration_ms <= cfg.max_window_ms:
        if _valid_window(events, cfg):
            return [events]
        return []

    # Long episode: pause-based bursts first.
    bursts = _split_on_gaps(events, cfg.pause_gap_ms)
    windows: List[List[Dict[str, Any]]] = []

    for burst in bursts:
        burst_dur = float(burst[-1]["t"]) - float(burst[0]["t"])
        if burst_dur <= cfg.max_window_ms:
            if _valid_window(burst, cfg):
                windows.append(burst)
            continue

        strong = _split_on_gaps(burst, cfg.strong_pause_gap_ms)
        for piece in strong:
            piece_dur = float(piece[-1]["t"]) - float(piece[0]["t"])
            if piece_dur <= cfg.max_window_ms and _valid_window(piece, cfg):
                windows.append(piece)
            else:
                windows.extend(_sliding_windows(piece, cfg))

    return windows


def extract_feature_windows_from_events(
    events: List[Dict[str, Any]],
    source_label: str = "live",
    cfg: ActiveWindowConfig = DEFAULT_ACTIVE_WINDOW_CONFIG,
) -> List[WindowSample]:
    """Build feature vectors from a live or offline event list."""
    if not events:
        return []

    episodes = _split_on_gaps(sorted(events, key=lambda e: float(e["t"])), cfg.afk_gap_ms)
    samples: List[WindowSample] = []

    for episode in episodes:
        for window in segment_episode_into_windows(episode, cfg):
            duration = float(window[-1]["t"]) - float(window[0]["t"])
            mode = "burst" if duration <= cfg.max_window_ms else "sliding"
            samples.append(
                WindowSample(
                    vector=feature_dict_to_vector(extract_features(window)),
                    source_file=source_label,
                    window_ms=duration,
                    event_count=len(window),
                    mode=mode,
                )
            )
    return samples


def aggregate_window_evaluations(
    evaluations: List[Dict[str, Any]],
    strategy: str = "max_risk",
) -> Dict[str, Any]:
    """Combine per-window ML evaluations into one decision."""
    if not evaluations:
        return {
            "decision": "UNKNOWN",
            "is_bot": None,
            "risk_score": 0.0,
            "bot_probability": None,
            "window_count": 0,
            "flags": ["No scorable active windows."],
            "model_type": evaluations[0].get("model_type") if evaluations else "unloaded",
        }

    if strategy == "max_risk":
        best = max(evaluations, key=lambda e: float(e.get("risk_score", 0.0)))
    else:
        best = evaluations[0]

    all_flags: List[str] = []
    for ev in evaluations:
        for flag in ev.get("flags", []):
            if flag not in all_flags:
                all_flags.append(flag)

    bot_probs = [e.get("bot_probability") for e in evaluations if e.get("bot_probability") is not None]
    avg_bot_prob = float(np.mean(bot_probs)) if bot_probs else best.get("bot_probability")

    return {
        **best,
        "bot_probability": round(avg_bot_prob, 4) if bot_probs else best.get("bot_probability"),
        "risk_score": round(float(best.get("risk_score", 0.0)), 4),
        "window_count": len(evaluations),
        "flags": all_flags,
        "aggregation": strategy,
    }
