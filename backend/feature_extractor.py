import math
import numpy as np
from typing import List, Dict, Any

def extract_features(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts 12 biometric movement & timing features from raw telemetry events.
    Robustly handles all event string representations ('mouse_move', 'mousemove', 'mousedown', etc.)
    and separates pure movement windows (no clicks) from click windows.
    """
    if not events or len(events) < 2:
        return {
            "event_count": len(events) if events else 0,
            "duration_ms": 0.0,
            "total_distance_px": 0.0,
            "euclidean_distance_px": 0.0,
            "straightness_index": 1.0,
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "velocity_max": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_std": 0.0,
            "acceleration_max": 0.0,
            "jerk_mean": 0.0,
            "angle_change_mean": 0.0,
            "angle_change_std": 0.0,
            "jitter_ratio": 0.0,
            "idle_ratio": 0.0,
            "avg_click_dwell_ms": -1.0
        }

    # Sort events by timestamp t
    events = sorted(events, key=lambda e: e['t'])

    t_start = events[0]['t']
    t_end = events[-1]['t']
    duration_ms = max(1.0, t_end - t_start)

    coords = []
    click_downs = {}
    click_dwell_times = []

    for e in events:
        raw_type = str(e.get('type', 'mousemove')).lower().replace('_', '')
        x, y, t = float(e['x']), float(e['y']), float(e['t'])

        if raw_type in ('mousemove', '0'):
            coords.append((x, y, t))
        elif raw_type in ('mousedown', 'mouseclickdown', '1'):
            coords.append((x, y, t))
            btn = e.get('button', 'left')
            click_downs[btn] = t
        elif raw_type in ('mouseup', 'mouseclickup', '2'):
            coords.append((x, y, t))
            btn = e.get('button', 'left')
            if btn in click_downs:
                dwell = max(0.0, t - click_downs[btn])
                click_dwell_times.append(dwell)
                del click_downs[btn]

    if len(coords) < 2:
        return {
            "event_count": len(events),
            "duration_ms": duration_ms,
            "total_distance_px": 0.0,
            "euclidean_distance_px": 0.0,
            "straightness_index": 1.0,
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "velocity_max": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_std": 0.0,
            "acceleration_max": 0.0,
            "jerk_mean": 0.0,
            "angle_change_mean": 0.0,
            "angle_change_std": 0.0,
            "jitter_ratio": 0.0,
            "idle_ratio": 0.0,
            "avg_click_dwell_ms": float(np.mean(click_dwell_times)) if click_dwell_times else -1.0
        }

    # Trajectory Distances
    x_coords = np.array([c[0] for c in coords])
    y_coords = np.array([c[1] for c in coords])
    t_coords = np.array([c[2] for c in coords])

    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    dt = np.diff(t_coords)

    # Filter out 0-dt duplicates
    valid_mask = dt > 0.001
    if not np.any(valid_mask):
        return {
            "event_count": len(events),
            "duration_ms": duration_ms,
            "total_distance_px": 0.0,
            "euclidean_distance_px": 0.0,
            "straightness_index": 1.0,
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "velocity_max": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_std": 0.0,
            "acceleration_max": 0.0,
            "jerk_mean": 0.0,
            "angle_change_mean": 0.0,
            "angle_change_std": 0.0,
            "jitter_ratio": 0.0,
            "idle_ratio": 0.0,
            "avg_click_dwell_ms": float(np.mean(click_dwell_times)) if click_dwell_times else -1.0
        }

    dx = dx[valid_mask]
    dy = dy[valid_mask]
    dt = dt[valid_mask]

    step_distances = np.sqrt(dx**2 + dy**2)
    total_distance_px = float(np.sum(step_distances))

    euclidean_distance_px = float(np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2))

    if total_distance_px > 0.01:
        straightness_index = float(np.clip(euclidean_distance_px / total_distance_px, 0.0, 1.0))
    else:
        straightness_index = 1.0

    # Velocities (px / ms)
    velocities = step_distances / dt
    velocity_mean = float(np.mean(velocities))
    velocity_std = float(np.std(velocities)) if len(velocities) > 1 else 0.0
    velocity_max = float(np.max(velocities))

    # Accelerations (px / ms^2)
    if len(velocities) > 1:
        dv = np.diff(velocities)
        dt_acc = dt[1:]
        valid_acc = dt_acc > 0.001
        if np.any(valid_acc):
            accelerations = np.abs(dv[valid_acc] / dt_acc[valid_acc])
            acc_mean = float(np.mean(accelerations))
            acc_std = float(np.std(accelerations)) if len(accelerations) > 1 else 0.0
            acc_max = float(np.max(accelerations))
        else:
            acc_mean, acc_std, acc_max = 0.0, 0.0, 0.0
    else:
        acc_mean, acc_std, acc_max = 0.0, 0.0, 0.0

    # Jerk (px / ms^3)
    if len(velocities) > 2 and 'accelerations' in locals() and len(accelerations) > 1:
        da = np.diff(accelerations)
        dt_jerk = dt[2:]
        valid_jerk = dt_jerk > 0.001
        if np.any(valid_jerk):
            jerks = np.abs(da[valid_jerk] / dt_jerk[valid_jerk])
            jerk_mean = float(np.mean(jerks))
        else:
            jerk_mean = 0.0
    else:
        jerk_mean = 0.0

    # Angles & Curvature
    if len(dx) > 1:
        angles = np.arctan2(dy, dx)
        angle_diffs = np.abs(np.diff(angles))
        angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
        angle_mean = float(np.mean(angle_diffs))
        angle_std = float(np.std(angle_diffs)) if len(angle_diffs) > 1 else 0.0

        # Jitter Ratio (high frequency direction flips > 45 deg)
        jitter_count = np.sum(angle_diffs > (np.pi / 4.0))
        jitter_ratio = float(jitter_count / len(angle_diffs))
    else:
        angle_mean, angle_std, jitter_ratio = 0.0, 0.0, 0.0

    # Idle Ratio (pauses > 100ms)
    idle_time_ms = float(np.sum(dt[dt > 100.0]))
    idle_ratio = float(np.clip(idle_time_ms / duration_ms, 0.0, 1.0))

    # Click Dwell Time: -1.0 means no click in this window
    avg_click_dwell = float(np.mean(click_dwell_times)) if click_dwell_times else -1.0

    return {
        "event_count": len(events),
        "duration_ms": round(duration_ms, 2),
        "total_distance_px": round(total_distance_px, 2),
        "euclidean_distance_px": round(euclidean_distance_px, 2),
        "straightness_index": round(straightness_index, 4),
        "velocity_mean": round(velocity_mean, 4),
        "velocity_std": round(velocity_std, 4),
        "velocity_max": round(velocity_max, 4),
        "acceleration_mean": round(acc_mean, 6),
        "acceleration_std": round(acc_std, 6),
        "acceleration_max": round(acc_max, 6),
        "jerk_mean": round(jerk_mean, 8),
        "angle_change_mean": round(angle_mean, 4),
        "angle_change_std": round(angle_std, 4),
        "jitter_ratio": round(jitter_ratio, 4),
        "idle_ratio": round(idle_ratio, 4),
        "avg_click_dwell_ms": round(avg_click_dwell, 2)
    }

def feature_dict_to_vector(f: Dict[str, Any]) -> List[float]:
    """Converts feature dictionary into ordered 12-element vector for ML models."""
    return [
        float(f.get("straightness_index", 1.0)),
        float(f.get("velocity_mean", 0.0)),
        float(f.get("velocity_std", 0.0)),
        float(f.get("velocity_max", 0.0)),
        float(f.get("acceleration_mean", 0.0)),
        float(f.get("acceleration_std", 0.0)),
        float(f.get("jerk_mean", 0.0)),
        float(f.get("angle_change_mean", 0.0)),
        float(f.get("angle_change_std", 0.0)),
        float(f.get("jitter_ratio", 0.0)),
        float(f.get("idle_ratio", 0.0)),
        float(f.get("avg_click_dwell_ms", -1.0))
    ]
