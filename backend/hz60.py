"""Downsample pointer streams to a target poll rate (default 60 Hz)."""

from __future__ import annotations

from typing import Any, Dict, List


def _is_click(raw: Any) -> bool:
    key = str(raw or "mousemove").lower().replace("_", "")
    return key in {"mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"}


def downsample_events_to_hz(events: List[Dict[str, Any]], hz: float = 60.0) -> List[Dict[str, Any]]:
    """Keep clicks; keep moves at most once per 1000/hz ms. Always keep first and last."""
    if not events or hz <= 0:
        return list(events)
    ordered = sorted(events, key=lambda e: float(e.get("t", 0.0)))
    if len(ordered) < 3:
        return ordered
    period = 1000.0 / hz
    out: List[Dict[str, Any]] = []
    last_keep_t: float | None = None
    last_idx = len(ordered) - 1
    for i, event in enumerate(ordered):
        t = float(event.get("t", 0.0))
        keep_click = _is_click(event.get("type", event.get("event_type")))
        if i == 0 or i == last_idx or keep_click:
            out.append(event)
            last_keep_t = t
            continue
        if last_keep_t is None or (t - last_keep_t) >= period * 0.92:
            out.append(event)
            last_keep_t = t
    return out
