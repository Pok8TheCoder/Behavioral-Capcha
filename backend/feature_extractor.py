import math
import numpy as np

def extract_features(events: list) -> dict:
    """
    Extracts biometric feature vector from raw interaction telemetry events.
    Each event in `events` should be a dict: {'x': float, 'y': float, 't': float, 'type': str}
    """
    if not events or len(events) < 2:
        return _get_empty_features()

    # Filter mouse movements
    mouse_events = [e for e in events if e.get('type') in ['mousemove', 'move', 'drag']]
    click_events = [e for e in events if e.get('type') in ['mousedown', 'mouseup', 'click']]
    key_events = [e for e in events if e.get('type') in ['keydown', 'keyup']]

    # If minimal mouse points, extract basic fallback features
    if len(mouse_events) < 2:
        # Check if events overall have x/y coordinates (in case type wasn't explicitly mousemove)
        coords = [(e['x'], e['y'], e['t']) for e in events if 'x' in e and 'y' in e]
        if len(coords) >= 2:
            mouse_events = [{'x': c[0], 'y': c[1], 't': c[2], 'type': 'mousemove'} for c in coords]
        else:
            return _get_empty_features()

    # Sort events by timestamp
    mouse_events.sort(key=lambda e: e['t'])

    x = np.array([float(e['x']) for e in mouse_events])
    y = np.array([float(e['y']) for e in mouse_events])
    t = np.array([float(e['t']) for e in mouse_events])

    # Calculate time deltas (in milliseconds)
    dt = np.diff(t)
    # Prevent zero division by replacing 0 with 0.1ms
    dt = np.where(dt <= 0, 0.1, dt)

    # Calculate spatial deltas
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)

    total_path_length = float(np.sum(distances))
    start_end_dist = float(math.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    
    # Straightness Index (1.0 = perfect straight line bot)
    straightness = start_end_dist / (total_path_length + 1e-6)
    straightness = min(1.0, max(0.0, straightness))

    # Velocity (px/ms)
    velocities = distances / dt
    mean_vel = float(np.mean(velocities))
    std_vel = float(np.std(velocities))
    max_vel = float(np.max(velocities))

    # Acceleration (px/ms^2)
    if len(velocities) > 1:
        accelerations = np.diff(velocities) / dt[:-1]
        mean_acc = float(np.mean(np.abs(accelerations)))
        std_acc = float(np.std(accelerations))
        max_acc = float(np.max(np.abs(accelerations)))
    else:
        mean_acc, std_acc, max_acc = 0.0, 0.0, 0.0

    # Jerk (px/ms^3) - 3rd derivative
    if len(velocities) > 2:
        jerks = np.diff(np.diff(velocities) / dt[:-1]) / dt[:-2]
        mean_jerk = float(np.mean(np.abs(jerks)))
    else:
        mean_jerk = 0.0

    # Angular analysis (direction changes)
    angles = np.arctan2(dy, dx)
    if len(angles) > 1:
        angle_diffs = np.diff(angles)
        # Normalize angle diffs to [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        mean_angle_change = float(np.mean(np.abs(angle_diffs)))
        std_angle_change = float(np.std(angle_diffs))
        
        # Micro-jitter ratio: high angle change (>45 deg) with small displacement (< 15px)
        jitter_mask = (np.abs(angle_diffs) > (np.pi / 4)) & (distances[1:] < 15)
        jitter_ratio = float(np.sum(jitter_mask) / (len(angle_diffs) + 1e-6))
    else:
        mean_angle_change, std_angle_change, jitter_ratio = 0.0, 0.0, 0.0

    # Idle time ratio (velocity < 0.02 px/ms)
    idle_time = float(np.sum(dt[velocities < 0.02]))
    total_duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    idle_ratio = idle_time / (total_duration + 1e-6)

    # Click dwell time extraction
    click_dwell_times = []
    downs = {}
    for c in click_events:
        c_type = c.get('type')
        c_time = c.get('t', 0)
        if c_type == 'mousedown':
            downs['mouse'] = c_time
        elif c_type == 'mouseup' and 'mouse' in downs:
            click_dwell_times.append(c_time - downs['mouse'])
            del downs['mouse']

    avg_click_dwell = float(np.mean(click_dwell_times)) if click_dwell_times else 0.0

    return {
        "event_count": len(mouse_events),
        "duration_ms": round(total_duration, 2),
        "total_distance_px": round(total_path_length, 2),
        "euclidean_distance_px": round(start_end_dist, 2),
        "straightness_index": round(straightness, 4),
        "velocity_mean": round(mean_vel, 4),
        "velocity_std": round(std_vel, 4),
        "velocity_max": round(max_vel, 4),
        "acceleration_mean": round(mean_acc, 6),
        "acceleration_std": round(std_acc, 6),
        "acceleration_max": round(max_acc, 6),
        "jerk_mean": round(mean_jerk, 8),
        "angle_change_mean": round(mean_angle_change, 4),
        "angle_change_std": round(std_angle_change, 4),
        "jitter_ratio": round(jitter_ratio, 4),
        "idle_ratio": round(idle_ratio, 4),
        "avg_click_dwell_ms": round(avg_click_dwell, 2)
    }

def feature_dict_to_vector(feat: dict) -> list:
    """Converts feature dict into an ordered numerical list for ML model input."""
    return [
        feat.get("straightness_index", 0.0),
        feat.get("velocity_mean", 0.0),
        feat.get("velocity_std", 0.0),
        feat.get("velocity_max", 0.0),
        feat.get("acceleration_mean", 0.0),
        feat.get("acceleration_std", 0.0),
        feat.get("jerk_mean", 0.0),
        feat.get("angle_change_mean", 0.0),
        feat.get("angle_change_std", 0.0),
        feat.get("jitter_ratio", 0.0),
        feat.get("idle_ratio", 0.0),
        feat.get("avg_click_dwell_ms", 0.0)
    ]

def _get_empty_features() -> dict:
    return {
        "event_count": 0,
        "duration_ms": 0.0,
        "total_distance_px": 0.0,
        "euclidean_distance_px": 0.0,
        "straightness_index": 0.0,
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
        "avg_click_dwell_ms": 0.0
    }
