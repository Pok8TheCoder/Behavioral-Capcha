"""
Analyze human CSV telemetry: AFK cleanup stats and merged 3s window counts.

Does not physically merge large CSV files; produces a summary JSON used to size
bot challenge dataset generation.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(__file__)
HUMAN_DIR = os.path.join(ROOT_DIR, "datasets", "human")
SUMMARY_PATH = os.path.join(HUMAN_DIR, "merged_analysis_summary.json")

sys.path.insert(0, ROOT_DIR)

from train_isolation_forest import extract_active_windows_from_csv, AFK_GAP_MS
from backend.active_window import DEFAULT_ACTIVE_WINDOW_CONFIG


def analyze_human_dataset(max_windows_per_file: int = 10_000_000) -> Dict[str, Any]:
    files = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.csv")))
    per_file: List[Dict[str, Any]] = []
    total_windows = 0
    total_raw_rows = 0
    total_afk_splits = 0

    print("=" * 72)
    print("HUMAN DATASET ANALYSIS (AFK-cleaned 3s windows)")
    print("=" * 72)
    print(f"[Data] CSV files: {len(files)}")

    for file_path in files:
        samples, summary = extract_active_windows_from_csv(file_path, max_windows=max_windows_per_file)
        windows = len(samples)
        total_windows += windows
        total_raw_rows += int(summary.get("raw_rows", 0))
        total_afk_splits += int(summary.get("afk_splits", 0))
        entry = {
            "file": os.path.basename(file_path),
            "windows": windows,
            "raw_rows": summary.get("raw_rows", 0),
            "afk_splits": summary.get("afk_splits", 0),
            "truncated": summary.get("truncated", False),
            "error": summary.get("error"),
        }
        per_file.append(entry)
        print(
            f"  {entry['file']}: {windows:,} windows, "
            f"{entry['afk_splits']:,} AFK splits, {entry['raw_rows']:,} rows"
        )

    doc = {
        "human_dir": HUMAN_DIR,
        "file_count": len(files),
        "merged_virtual_windows": total_windows,
        "total_raw_rows": total_raw_rows,
        "total_afk_splits": total_afk_splits,
        "preprocessing": {
            "window_mode": "dynamic_active",
            "afk_gap_ms": AFK_GAP_MS,
            "pause_gap_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.pause_gap_ms,
            "max_window_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.max_window_ms,
            "fallback_window_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.fallback_window_ms,
            "fallback_stride_ms": DEFAULT_ACTIVE_WINDOW_CONFIG.fallback_stride_ms,
        },
        "per_file": per_file,
        "bot_targets": {
            "2x_windows": total_windows * 2,
            "3x_windows": total_windows * 3,
            "4x_windows": total_windows * 4,
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print("-" * 72)
    print(f"[Done] Merged virtual 3s windows: {total_windows:,}")
    print(f"[Done] Bot target (2x): {doc['bot_targets']['2x_windows']:,}")
    print(f"[Done] Summary: {SUMMARY_PATH}")
    return doc


if __name__ == "__main__":
    analyze_human_dataset()
