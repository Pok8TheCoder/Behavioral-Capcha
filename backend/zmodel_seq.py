"""Zmodel sequence encoding: relative Δx, Δy, Δt inside a dynamic window."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SEQ_LEN = 192
DX_SCALE = 40.0
DT_SCALE = 40.0
DX_CLIP = 120.0
DT_CLIP = 80.0


def events_to_relative_sequence(
    events: Sequence[Dict[str, Any]],
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((seq_len, 3), dtype=np.float32)
    mask = np.zeros((seq_len,), dtype=np.float32)
    if not events or len(events) < 2:
        return seq, mask
    ordered = sorted(events, key=lambda e: float(e.get("t", 0.0)))
    n = 0
    for i in range(1, len(ordered)):
        if n >= seq_len:
            break
        dx = float(ordered[i]["x"]) - float(ordered[i - 1]["x"])
        dy = float(ordered[i]["y"]) - float(ordered[i - 1]["y"])
        dt = float(ordered[i]["t"]) - float(ordered[i - 1]["t"])
        seq[n, 0] = np.clip(dx, -DX_CLIP, DX_CLIP) / DX_SCALE
        seq[n, 1] = np.clip(dy, -DX_CLIP, DX_CLIP) / DX_SCALE
        seq[n, 2] = np.clip(dt, 0.0, DT_CLIP) / DT_SCALE
        mask[n] = 1.0
        n += 1
    return seq, mask
