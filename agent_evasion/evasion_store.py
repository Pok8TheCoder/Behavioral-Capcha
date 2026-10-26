"""Persist successful evasion episodes as native 60 Hz bot CSVs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

_MANIFEST_LOCK = threading.Lock()

from backend.active_window import extract_feature_windows_from_events

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP_DIR = os.path.join(ROOT, "datasets", "bot_xmodel_60hz", "loop_winners")
PROBE_FAILS_DIR = os.path.join(ROOT, "datasets", "bot_xmodel_60hz", "probe_fails")
MANIFEST_PATH = os.path.join(LOOP_DIR, "manifest.json")
CANVAS_W = 960.0
CANVAS_H = 640.0
CSV_FIELDS = [
    "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
    "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
    "event_type", "target_item",
]


def count_windows(events: List[Dict[str, Any]]) -> int:
    samples = extract_feature_windows_from_events(events, source_label="loop")
    return max(1, len(samples)) if events else 0


def _event_type(event: Dict[str, Any]) -> str:
    raw = str(event.get("type") or event.get("event_type") or "mousemove").lower()
    key = raw.replace("_", "")
    if key in {"mousedown", "mouseclickdown", "1"}:
        return "mousedown"
    if key in {"mouseup", "mouseclickup", "2"}:
        return "mouseup"
    return "mousemove"


def save_episode(
    events: List[Dict[str, Any]],
    *,
    bot_id: str,
    model_id: str,
    iteration: int,
    challenge_type: str,
    seed: Any,
    human_confidence: float,
    tagged: str,
) -> Dict[str, Any]:
    os.makedirs(LOOP_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    windows = count_windows(events)
    fname = f"loop_i{iteration:02d}_{bot_id}_{tagged}_{stamp}_60hz.csv"
    path = os.path.join(LOOP_DIR, fname)
    t0 = min(float(e.get("t", 0.0)) for e in events) if events else 0.0
    rows = []
    for event in events:
        x = float(event.get("x", 0.0))
        y = float(event.get("y", 0.0))
        rows.append({
            "rel_time_ms": f"{max(0.0, float(event.get('t', 0.0)) - t0):.3f}",
            "x_px": f"{x:.1f}",
            "y_px": f"{y:.1f}",
            "x_norm": f"{x / CANVAS_W:.5f}",
            "y_norm": f"{y / CANVAS_H:.5f}",
            "scroll_x": "0.0",
            "scroll_y": "0.0",
            "scroll_dx": "0",
            "scroll_dy": "0",
            "event_type": _event_type(event),
            "target_item": f"loop:{bot_id}:{tagged}",
        })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(
            f"# source=evasion_loop model={model_id} iter={iteration} bot={bot_id} "
            f"type={challenge_type} seed={seed} hc={human_confidence:.4f} tag={tagged}\n"
        )
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    entry = {
        "csv": path,
        "filename": fname,
        "bot_id": bot_id,
        "model_id": model_id,
        "iteration": iteration,
        "challenge_type": challenge_type,
        "seed": seed,
        "human_confidence": human_confidence,
        "tagged": tagged,
        "windows": windows,
        "events": len(events),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_manifest(entry)
    return entry


def _append_manifest(entry: Dict[str, Any]) -> None:
    with _MANIFEST_LOCK:
        doc: Dict[str, Any] = {"episodes": []}
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    doc = loaded
            except (OSError, json.JSONDecodeError):
                pass
        doc.setdefault("episodes", []).append({
            **entry,
            "csv": os.path.relpath(entry["csv"], ROOT).replace("\\", "/"),
        })
        doc["updated_at"] = entry["saved_at"]
        doc["episode_count"] = len(doc["episodes"])
        with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2)


def events_from_loop_csv(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        rows = csv.DictReader(row for row in handle if not row.startswith("#"))
        for row in rows:
            etype = str(row.get("event_type") or "mousemove").lower()
            events.append({
                "t": float(row.get("rel_time_ms") or 0.0),
                "x": float(row.get("x_px") or 0.0),
                "y": float(row.get("y_px") or 0.0),
                "type": etype,
            })
    if not events:
        raise ValueError(f"No events in {path}")
    return events


def _load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {"episodes": []}
    with _MANIFEST_LOCK:
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"episodes": []}
    return doc if isinstance(doc, dict) else {"episodes": []}


def list_recent_episodes(count: int = 2, tagged: str = "win") -> List[Dict[str, Any]]:
    episodes = list(_load_manifest().get("episodes") or [])
    wins = [e for e in episodes if str(e.get("tagged") or "") == tagged]
    pool = wins if wins else episodes
    return pool[-max(1, int(count)):]


def list_probe_datasets() -> List[Dict[str, Any]]:
    episodes = list(_load_manifest().get("episodes") or [])
    by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for entry in episodes:
        try:
            iteration = int(entry.get("iteration") or 1)
        except (TypeError, ValueError):
            iteration = 1
        by_iter.setdefault(iteration, []).append(entry)
    datasets: List[Dict[str, Any]] = []
    for iteration, rows in sorted(by_iter.items()):
        for tagged, label in (("win", "winners"), ("mark", "marks")):
            subset = [r for r in rows if str(r.get("tagged") or "") == tagged]
            if not subset:
                continue
            windows = sum(int(r.get("windows") or 0) for r in subset)
            featured = tagged == "win" and iteration == 1
            datasets.append({
                "id": f"loop_i{iteration:02d}_{tagged}",
                "label": (
                    f"Cycle {iteration} harvest"
                    if featured else
                    f"Cycle {iteration} {label}"
                ),
                "detail": f"{windows} windows / {len(subset)} episodes",
                "kind": "loop",
                "iteration": iteration,
                "tagged": tagged,
                "windows": windows,
                "episodes": len(subset),
                "featured": featured,
            })
    datasets.sort(key=lambda row: (0 if row.get("featured") else 1, -int(row.get("windows") or 0)))
    return datasets


def episodes_for_dataset(dataset_id: str) -> List[Dict[str, Any]]:
    wanted = (dataset_id or "").strip()
    if not wanted:
        return []
    episodes = list(_load_manifest().get("episodes") or [])
    parts = wanted.split("_")
    # loop_i01_win / loop_i01_mark
    if wanted.startswith("loop_i") and len(parts) >= 3:
        try:
            iteration = int(parts[1][1:])
        except (TypeError, ValueError):
            iteration = 1
        tagged = parts[2]
        return [
            e for e in episodes
            if int(e.get("iteration") or 1) == iteration
            and str(e.get("tagged") or "") == tagged
        ]
    return []


def export_failed_windows(probe: Dict[str, Any], dest_dir: str = PROBE_FAILS_DIR) -> Dict[str, Any]:
    """Copy episode CSVs the current model failed to catch (still beat, >=65% HC)."""
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    windows = 0
    for entry in probe.get("episodes") or []:
        if entry.get("error"):
            continue
        leak_n = int(entry.get("windows_passed") or 0)
        if leak_n < 1:
            continue
        rel = str(entry.get("path") or entry.get("csv") or "")
        if not rel:
            continue
        src = rel if os.path.isabs(rel) else os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
        copied += 1
        windows += leak_n
    if copied < 1:
        raise RuntimeError("No failed/leaking windows to train on. Test the dataset first.")
    return {
        "dir": dest_dir,
        "episodes": copied,
        "windows": windows,
        "model_id": probe.get("model_id"),
        "dataset_id": probe.get("dataset_id"),
    }
