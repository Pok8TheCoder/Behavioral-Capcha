"""Normalize bot telemetry dicts for API / feature extraction."""

from __future__ import annotations

from typing import Any, Dict, List


def normalize_bot_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for event in events:
        raw_type = str(event.get("type", event.get("event_type", "mousemove"))).lower()
        if raw_type in ("mouse_move", "0", "mousemove"):
            evt_type = "mousemove"
        elif raw_type in ("mouse_click_down", "mousedown", "1"):
            evt_type = "mousedown"
        elif raw_type in ("mouse_click_up", "mouseup", "2"):
            evt_type = "mouseup"
        else:
            evt_type = raw_type
        normalized.append({
            "x": float(event["x"]),
            "y": float(event["y"]),
            "t": float(event.get("t", 0)),
            "type": evt_type,
        })
    return normalized
