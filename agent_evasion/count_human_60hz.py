"""Quick inventory: 60Hz human CSVs, AFK splits, active event counts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "datasets" / "human"
AFK_GAP_MS = 5000.0
MOVEMENT_EPSILON_PX = 0.5


def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def count_active_events(path: Path) -> dict:
    preview = pd.read_csv(path, comment="#", nrows=0)
    time_col = find_col(preview.columns, ["rel_time_ms", "t", "timestamp_ms"])
    x_col = find_col(preview.columns, ["x_px", "x"])
    y_col = find_col(preview.columns, ["y_px", "y"])
    evt_col = find_col(preview.columns, ["event_type", "type"])
    if not (time_col and x_col and y_col):
        return {"error": "missing cols", "raw_rows": 0, "active_events": 0, "afk_splits": 0, "duration_s": 0.0}

    usecols = [time_col, x_col, y_col]
    if evt_col:
        usecols.append(evt_col)

    raw_rows = 0
    active_events = 0
    afk_splits = 0
    last_active_t = None
    last_x = last_y = None
    t_min = t_max = None

    for chunk in pd.read_csv(path, comment="#", usecols=usecols, chunksize=200_000):
        chunk = chunk.dropna(subset=[time_col, x_col, y_col])
        raw_rows += len(chunk)
        if chunk.empty:
            continue
        if t_min is None:
            t_min = float(chunk[time_col].iloc[0])
        t_max = float(chunk[time_col].iloc[-1])

        if evt_col:
            event_types = chunk[evt_col].astype(str).str.lower().str.replace("_", "", regex=False)
        else:
            event_types = pd.Series(["mousemove"] * len(chunk), index=chunk.index)

        x_values = chunk[x_col].astype(float)
        y_values = chunk[y_col].astype(float)
        dx = x_values.diff()
        dy = y_values.diff()
        if last_x is not None and last_y is not None:
            first_idx = chunk.index[0]
            dx.loc[first_idx] = x_values.loc[first_idx] - last_x
            dy.loc[first_idx] = y_values.loc[first_idx] - last_y
        moved = np.hypot(dx.fillna(0.0), dy.fillna(0.0)) >= MOVEMENT_EPSILON_PX
        if last_x is None:
            moved.iloc[0] = True
        movement_mask = event_types.isin(["mousemove", "0"]) & moved
        click_mask = event_types.isin(["mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"])
        filtered = chunk[movement_mask | click_mask]
        event_types_f = event_types.loc[filtered.index]

        for idx, row in filtered.iterrows():
            t = float(row[time_col])
            etn = event_types_f.loc[idx]
            is_movement = etn in {"mousemove", "0"}
            is_click = etn in {"mousedown", "mouseup", "mouseclickdown", "mouseclickup", "1", "2"}
            if is_movement:
                if last_active_t is not None and t - last_active_t > AFK_GAP_MS:
                    afk_splits += 1
                active_events += 1
                last_active_t = t
            elif is_click:
                active_events += 1

        last_x = float(x_values.iloc[-1])
        last_y = float(y_values.iloc[-1])

    duration_s = ((t_max - t_min) / 1000.0) if (t_min is not None and t_max is not None) else 0.0
    return {
        "raw_rows": raw_rows,
        "active_events": active_events,
        "afk_splits": afk_splits,
        "duration_s": duration_s,
    }


def summarize(group: list[Path], label: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for f in group:
        is_60 = "_60hz" in f.name.lower()
        info = count_active_events(f)
        rows.append(
            {
                "file": f.name,
                "is_60hz": is_60,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                **info,
            }
        )
        print(
            f"  {f.name}: raw={info['raw_rows']:,} active={info['active_events']:,} "
            f"afk={info['afk_splits']} dur={info['duration_s']:.0f}s 60hz={is_60}"
        )

    tot = {
        "label": label,
        "files": len(rows),
        "raw_rows": sum(r["raw_rows"] for r in rows),
        "active_events": sum(r["active_events"] for r in rows),
        "afk_splits": sum(r["afk_splits"] for r in rows),
        "duration_s": sum(r["duration_s"] for r in rows),
        "files_60hz": sum(1 for r in rows if r["is_60hz"]),
        "raw_rows_60hz": sum(r["raw_rows"] for r in rows if r["is_60hz"]),
        "active_events_60hz": sum(r["active_events"] for r in rows if r["is_60hz"]),
        "duration_s_60hz": sum(r["duration_s"] for r in rows if r["is_60hz"]),
        "files_legacy": sum(1 for r in rows if not r["is_60hz"]),
        "raw_rows_legacy": sum(r["raw_rows"] for r in rows if not r["is_60hz"]),
        "active_events_legacy": sum(r["active_events"] for r in rows if not r["is_60hz"]),
        "duration_s_legacy": sum(r["duration_s"] for r in rows if not r["is_60hz"]),
    }
    print(f"\n=== {label} TOTAL ===")
    print(json.dumps(tot, indent=2))
    return rows, tot


def main() -> None:
    files = sorted(HUMAN.glob("*.csv"))
    primary = [f for f in files if not f.name.startswith("recovered_")]
    recovered = [f for f in files if f.name.startswith("recovered_")]

    print("PRIMARY SESSION FILES")
    _, primary_tot = summarize(primary, "primary")
    print("\nRECOVERED SNAPSHOT FILES (overlap with primary sessions)")
    _, rec_tot = summarize(recovered, "recovered")

    out = {"primary": primary_tot, "recovered": rec_tot}
    out_path = HUMAN / "human_60hz_inventory.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
