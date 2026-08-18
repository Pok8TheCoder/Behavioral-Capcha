"""Save dashboard-labeled human challenge telemetry for later training."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.hz60 import downsample_events_to_hz

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_DIR = os.path.join(ROOT_DIR, "datasets", "human")
CHALLENGE_HUMAN_DIR = os.path.join(HUMAN_DIR, "challenges")
MANIFEST_PATH = os.path.join(CHALLENGE_HUMAN_DIR, "manifest.json")
CANVAS_W = 960.0
CANVAS_H = 640.0
LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0"})

EVENT_TYPE_CSV = {
    "mousemove": "mouse_move",
    "mouse_move": "mouse_move",
    "mousedown": "mouse_click_down",
    "mouse_click_down": "mouse_click_down",
    "mouseup": "mouse_click_up",
    "mouse_click_up": "mouse_click_up",
}


def sanitize_challenge_type(challenge_type: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(challenge_type or "unknown")).strip("_")
    return cleaned.lower() or "unknown"


def _csv_event_type(raw: Any) -> str:
    key = str(raw or "mousemove").lower()
    return EVENT_TYPE_CSV.get(key, "mouse_move")


def sanitize_ip(ip: str) -> str:
    cleaned = re.sub(r"[^0-9a-fA-F.:]", "", str(ip or "unknown"))
    return cleaned.strip("[]") or "unknown"


def origin_bucket(ip: str) -> str:
    host = sanitize_ip(ip).lower()
    if host.startswith("::ffff:"):
        host = host[7:]
    if host in LOCAL_HOSTS or host.startswith("127."):
        return "local"
    return "lan"


def _update_manifest(entry: Dict[str, Any]) -> None:
    os.makedirs(CHALLENGE_HUMAN_DIR, exist_ok=True)
    doc: Dict[str, Any] = {"by_challenge": {}}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                doc = loaded
        except (OSError, json.JSONDecodeError):
            pass
    by_challenge = doc.setdefault("by_challenge", {})
    ctype = entry["challenge_type"]
    bucket = by_challenge.setdefault(ctype, {"count": 0, "files": []})
    bucket["count"] = int(bucket.get("count", 0)) + 1
    files = bucket.setdefault("files", [])
    files.append({
        "csv": entry["csv"],
        "meta": entry["meta"],
        "saved_at": entry["saved_at"],
        "event_count": entry["event_count"],
        "task_passed": entry.get("task_passed"),
        "seed": entry.get("seed"),
        "client_ip": entry.get("client_ip"),
        "origin": entry.get("origin"),
        "challenge_features_summary": entry.get("challenge_features_summary"),
    })
    doc["updated_at"] = entry["saved_at"]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)


def write_human_capture(
    events: List[Dict[str, Any]],
    *,
    session_id: str,
    challenge_type: str,
    seed: Any = None,
    task_passed: Optional[bool] = None,
    features: Optional[Dict[str, Any]] = None,
    spec: Optional[Dict[str, Any]] = None,
    challenge_features: Optional[Dict[str, Any]] = None,
    client_ip: Optional[str] = None,
    target_hz: Optional[float] = None,
) -> Dict[str, Any]:
    ctype = sanitize_challenge_type(challenge_type)
    ip = sanitize_ip(client_ip or "unknown")
    origin = origin_bucket(ip)
    out_dir = os.path.join(CHALLENGE_HUMAN_DIR, origin, ctype)
    os.makedirs(out_dir, exist_ok=True)

    hz_tag = ""
    hz_value: Optional[float] = None
    if target_hz is not None and float(target_hz) > 0:
        hz_value = float(target_hz)
        events = downsample_events_to_hz(events, hz_value)
        hz_tag = f"_{int(round(hz_value))}hz"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_session = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:24]
    base = f"{ctype}_{stamp}_{safe_session or 'session'}{hz_tag}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    meta_path = os.path.join(out_dir, f"{base}.json")
    saved_at = datetime.now(timezone.utc).isoformat()

    t0 = min(float(e.get("t", 0)) for e in events)
    rows: List[Dict[str, str]] = []
    for event in events:
        x = float(event.get("x", 0))
        y = float(event.get("y", 0))
        rows.append({
            "rel_time_ms": f"{max(0.0, float(event.get('t', 0)) - t0):.3f}",
            "x_px": f"{x:.1f}",
            "y_px": f"{y:.1f}",
            "x_norm": f"{x / CANVAS_W:.5f}",
            "y_norm": f"{y / CANVAS_H:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": "0",
            "scroll_dy": "0",
            "event_type": _csv_event_type(event.get("type", event.get("event_type"))),
            "target_item": f"dashboard_human:{ctype}",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        handle.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({int(CANVAS_W)}x{int(CANVAS_H)})\n")
        handle.write(
            f"# source=dashboard origin={origin} client_ip={ip} session_id={session_id} "
            f"challenge_type={ctype} seed={seed} task_passed={task_passed}"
            f"{f' target_hz={hz_value:g}' if hz_value else ''}\n"
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
                "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
                "event_type", "target_item",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    meta = {
        "label": "human",
        "source": "dashboard",
        "origin": origin,
        "client_ip": ip,
        "session_id": session_id,
        "challenge_type": ctype,
        "seed": seed,
        "task_passed": task_passed,
        "target_hz": hz_value,
        "event_count": len(events),
        "features": _jsonable(features or {}),
        "challenge_features": _jsonable(challenge_features or {}),
        "spec": _jsonable(spec or {}),
        "csv": os.path.basename(csv_path),
        "relative_dir": os.path.join("challenges", origin, ctype).replace("\\", "/"),
        "saved_at": saved_at,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    summary = _jsonable(challenge_features or {})
    keep_keys = (
        "click_offset_mean_px", "coverage_ratio", "mean_distance_to_guide_px",
        "shape", "gates_passed", "hits", "sequence_ok", "avg_click_dwell_ms",
    )
    _update_manifest({
        "challenge_type": ctype,
        "csv": os.path.join("challenges", origin, ctype, os.path.basename(csv_path)).replace("\\", "/"),
        "meta": os.path.join("challenges", origin, ctype, os.path.basename(meta_path)).replace("\\", "/"),
        "saved_at": saved_at,
        "event_count": len(events),
        "task_passed": task_passed,
        "seed": seed,
        "client_ip": ip,
        "origin": origin,
        "challenge_features_summary": {k: summary[k] for k in keep_keys if k in summary},
    })

    return {
        "csv_path": csv_path,
        "meta_path": meta_path,
        "filename": os.path.basename(csv_path),
        "challenge_type": ctype,
        "relative_dir": os.path.join("challenges", origin, ctype).replace("\\", "/"),
        "event_count": len(events),
        "client_ip": ip,
        "origin": origin,
    }
