"""AFK cleanup + dynamic window inventory for human and bot CSVs.

Uses the same pipeline as training:
  AFK gap >5s splits episodes; pause/burst windows; min 12 events and 20px path.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_isolation_forest import FEATURE_NAMES, extract_active_windows_from_csv

INVENTORY_PATH = ROOT / "datasets" / "preprocessing_inventory.json"
MAX_WINDOWS = 100_000


def _iter_csvs(directory: Path, recursive: bool = True) -> List[Path]:
    if not directory.is_dir():
        return []
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def process_files(label: str, files: List[Path]) -> Dict[str, Any]:
    print(f"\n=== {label}: {len(files)} CSV files ===", flush=True)
    by_subtype: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "files": 0, "raw_rows": 0, "windows": 0, "afk_splits": 0, "errors": 0,
    })
    total = {
        "files": 0,
        "raw_rows": 0,
        "windows": 0,
        "afk_splits": 0,
        "errors": 0,
        "empty_files": 0,
        "truncated_files": 0,
    }
    file_rows: List[Dict[str, Any]] = []

    for path in files:
        samples, summary = extract_active_windows_from_csv(str(path), max_windows=MAX_WINDOWS)
        windows = int(summary.get("windows") or len(samples))
        rows = int(summary.get("raw_rows") or 0)
        afk = int(summary.get("afk_splits") or 0)
        err = summary.get("error")
        truncated = bool(summary.get("truncated"))

        rel = path.relative_to(ROOT / "datasets").as_posix()
        subtype = path.parent.name
        bucket = by_subtype[subtype]
        bucket["files"] += 1
        bucket["raw_rows"] += rows
        bucket["windows"] += windows
        bucket["afk_splits"] += afk
        if err:
            bucket["errors"] += 1

        total["files"] += 1
        total["raw_rows"] += rows
        total["windows"] += windows
        total["afk_splits"] += afk
        if err:
            total["errors"] += 1
        if windows == 0:
            total["empty_files"] += 1
        if truncated:
            total["truncated_files"] += 1

        print(
            f"  {rel}: rows={rows:,} windows={windows:,} afk={afk}"
            + (f" ERROR {err}" if err else ""),
            flush=True,
        )
        file_rows.append({
            "file": rel,
            "windows": windows,
            "raw_rows": rows,
            "afk_splits": afk,
            "error": err,
            "truncated": truncated,
        })

    return {
        "label": label,
        "feature_names": FEATURE_NAMES,
        "totals": total,
        "by_subtype": dict(by_subtype),
        "files": file_rows,
    }


def main() -> None:
    human_dir = ROOT / "datasets" / "human"
    os_recordings = _iter_csvs(human_dir, recursive=False)
    live_dashboard = [
        p for p in _iter_csvs(human_dir / "challenges")
        if "_archive_no_spec" not in p.parts
    ]
    archived = [
        p for p in _iter_csvs(human_dir / "challenges" / "_archive_no_spec")
    ]

    bot_groups = [
        ("bot_challenge", ROOT / "datasets" / "bot_challenge"),
        ("bot_challenge_tiers", ROOT / "datasets" / "bot_challenge_tiers"),
        ("bot_evasion", ROOT / "datasets" / "bot_evasion"),
        ("bot_legacy", ROOT / "datasets" / "bot"),
        ("bot_humanlike", ROOT / "datasets" / "bot_humanlike"),
    ]

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "afk_gap_ms": 5000,
            "pause_gap_ms": 400,
            "min_window_ms": 500,
            "max_window_ms": 12000,
            "min_events": 12,
            "min_distance_px": 20,
            "features": FEATURE_NAMES,
            "max_windows_per_file": MAX_WINDOWS,
        },
        "human": {},
        "bot": {},
    }

    report["human"]["os_recordings"] = process_files("human_os_recordings", os_recordings)
    report["human"]["dashboard_live"] = process_files("human_dashboard_live", live_dashboard)
    report["human"]["dashboard_archive_generic"] = process_files(
        "human_dashboard_archive (no spec, not for challenge-feature training)",
        archived,
    )

    bot_total_windows = 0
    for name, directory in bot_groups:
        result = process_files(name, _iter_csvs(directory))
        report["bot"][name] = result
        bot_total_windows += result["totals"]["windows"]

    human_os = report["human"]["os_recordings"]["totals"]["windows"]
    human_dash = report["human"]["dashboard_live"]["totals"]["windows"]
    human_arch = report["human"]["dashboard_archive_generic"]["totals"]["windows"]
    human_train = human_os + human_dash

    report["summary"] = {
        "human_training_windows": human_train,
        "human_os_recording_windows": human_os,
        "human_dashboard_live_windows": human_dash,
        "human_dashboard_archive_windows_excluded_from_primary": human_arch,
        "bot_training_windows": bot_total_windows,
        "bot_by_source": {
            name: report["bot"][name]["totals"]["windows"]
            for name, _ in bot_groups
        },
        "balanced_pair_limit": min(human_train, bot_total_windows) if human_train and bot_total_windows else 0,
    }

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n========== TRAINING-WORTH WINDOWS ==========", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {INVENTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
