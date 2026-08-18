"""Challenge session security — blocks API-only synthetic telemetry bypass."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

CAPTCHA_STRICT = os.getenv("CAPTCHA_STRICT", "1") == "1"
SESSION_TTL_SEC = float(os.getenv("CHALLENGE_SESSION_TTL_SEC", "180"))
MAX_VERIFY_ATTEMPTS = int(os.getenv("CHALLENGE_MAX_VERIFY_ATTEMPTS", "4"))
MIN_SESSION_AGE_MS = 400.0
WALL_CLOCK_TOLERANCE = 0.45
MICRO_STEP_PX = 2.0
MICRO_JIGGLE_RATIO_THRESHOLD = 0.30
MAX_LIVE_BATCH = 64
LIVE_ARRIVAL_RATIO = 0.65
AUTOMATION_JITTER_MIN = 0.38
AUTOMATION_JERK_MAX = 0.22
AUTOMATION_ACC_MAX = 0.35
AUTOMATION_VEL_MAX = 0.92
AUTOMATION_EVENTS_MAX = 380

CLICK_CHALLENGE_TYPES = frozenset({
    "button_sequence",
    "pin_entry",
    "target_hunt",
})
DRAG_CHALLENGE_TYPES = frozenset({
    "shape_trace",
    "maze_navigate",
    "path_gates",
    "zigzag_trace",
    "spiral_trace",
    "connect_dots",
    "hazard_run",
})


def _event_type(raw: str) -> str:
    t = raw.lower()
    if t in ("mouse_move", "0", "mousemove"):
        return "mousemove"
    if t in ("mouse_click_down", "mousedown", "1"):
        return "mousedown"
    if t in ("mouse_click_up", "mouseup", "2"):
        return "mouseup"
    return t


def _has_pointer_attestation(events: List[Dict[str, Any]]) -> bool:
    for e in events:
        if e.get("pid") is not None or e.get("pointer_id") is not None:
            return True
    return False


def _pointer_down_up_count(events: List[Dict[str, Any]]) -> Tuple[int, int]:
    downs, ups = 0, 0
    for e in events:
        et = _event_type(str(e.get("type", e.get("event_type", ""))))
        if et == "mousedown":
            downs += 1
        elif et == "mouseup":
            ups += 1
    return downs, ups


def _event_duration_ms(events: List[Dict[str, Any]]) -> float:
    if len(events) < 2:
        return 0.0
    times = sorted(float(e.get("t", 0)) for e in events)
    return max(0.0, times[-1] - times[0])


def _micro_jiggle_ratio(events: List[Dict[str, Any]]) -> float:
    """Share of *moving* steps that are tiny. Zero-length hover/rounding is ignored."""
    moves = [
        e for e in events
        if _event_type(str(e.get("type", e.get("event_type", "")))) == "mousemove"
    ]
    if len(moves) < 12:
        return 0.0
    tiny = 0
    moving = 0
    for i in range(1, len(moves)):
        dx = float(moves[i].get("x", 0)) - float(moves[i - 1].get("x", 0))
        dy = float(moves[i].get("y", 0)) - float(moves[i - 1].get("y", 0))
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            continue
        moving += 1
        if dist <= MICRO_STEP_PX:
            tiny += 1
    if moving < 8:
        return 0.0
    return tiny / moving


def _automation_kinematic_reject(features: Dict[str, Any]) -> Tuple[bool, str]:
    """Playwright automation kinematic profiles (issues 002 + 003)."""
    jitter = float(features.get("jitter_ratio", 0.0))
    jerk = float(features.get("jerk_mean", 1.0))
    acc = float(features.get("acceleration_mean", 1.0))
    vel = float(features.get("velocity_mean", 0.0))
    events = int(features.get("event_count", 0))

    # Issue 002: high jitter + low dynamics + moderate velocity
    if (
        jitter >= AUTOMATION_JITTER_MIN
        and jerk <= AUTOMATION_JERK_MAX
        and acc <= AUTOMATION_ACC_MAX
        and vel <= AUTOMATION_VEL_MAX
        and events <= AUTOMATION_EVENTS_MAX
        and events >= 65
    ):
        return True, (
            f"Automation kinematic profile (jitter={jitter:.2f}, jerk={jerk:.3f}, acc={acc:.3f})"
        )

    # Issue 003: mid jitter + elevated velocity + flat dynamics
    if (
        0.28 <= jitter <= 0.52
        and jerk <= 0.12
        and acc <= 0.22
        and 0.82 <= vel <= 1.50
        and 85 <= events <= 230
    ):
        return True, (
            f"Fast-hop flat dynamics (jitter={jitter:.2f}, vel={vel:.2f}, jerk={jerk:.3f})"
        )

    # Issue 004 family: low jitter + moderate dynamics + dense hops
    if (
        0.06 <= jitter <= 0.24
        and 0.10 <= jerk <= 0.35
        and 0.18 <= acc <= 0.48
        and 0.45 <= vel <= 1.28
        and 50 <= events <= 280
    ):
        return True, (
            f"Dense-hop Playwright family (jitter={jitter:.2f}, vel={vel:.2f}, "
            f"jerk={jerk:.3f}, acc={acc:.3f})"
        )

    # Issue 005: high-jitter dart hops (steps >2px so micro-jiggle gate may not fire)
    if (
        0.38 <= jitter <= 0.65
        and 1.02 <= vel <= 1.55
        and 70 <= events <= 200
        and 0.08 <= jerk <= 0.35
        and 0.18 <= acc <= 0.55
    ):
        return True, (
            f"Dart-hop Playwright family (jitter={jitter:.2f}, vel={vel:.2f}, jerk={jerk:.3f})"
        )

    # Issue 006: min-jerk dense overflow (events above the #18 280 cap)
    if (
        280 <= events <= 550
        and 0.10 <= jitter <= 0.22
        and 0.60 <= vel <= 1.05
        and 0.08 <= jerk <= 0.30
        and 0.14 <= acc <= 0.40
    ):
        return True, (
            f"Min-jerk dense overflow (events={events}, jitter={jitter:.2f}, vel={vel:.2f})"
        )

    # Issue 007: sparse irregular (~60 events, high vel/acc, mid jitter)
    if (
        45 <= events <= 110
        and 1.15 <= vel <= 1.85
        and 0.18 <= jitter <= 0.32
        and jerk >= 0.10
        and acc >= 0.28
    ):
        return True, (
            f"Sparse irregular Playwright min-jerk (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    # Issue 008: mid_count overflow of the #24 event cap
    if (
        111 <= events <= 250
        and 1.05 <= vel <= 1.70
        and 0.14 <= jitter <= 0.28
        and jerk >= 0.18
        and acc >= 0.32
    ):
        return True, (
            f"Mid-count Playwright min-jerk (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    # Issue 009: high_str_fast — elevated velocity, sparse events
    if (
        45 <= events <= 95
        and vel >= 1.70
        and 0.14 <= jitter <= 0.28
        and jerk >= 0.28
        and acc >= 0.45
    ):
        return True, (
            f"High-str fast Playwright hops (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    idle = float(features.get("idle_ratio", 0.0))
    dwell = float(features.get("avg_click_dwell_ms", 0.0))
    # Issue 010: idle_correct — longer pauses, vel still elevated, events ~160
    if (
        120 <= events <= 220
        and vel >= 1.55
        and 0.14 <= jitter <= 0.24
        and jerk >= 0.35
        and acc >= 0.50
        and idle >= 0.32
        and dwell >= 90.0
    ):
        return True, (
            f"Idle-correct Playwright long-dwell (events={events}, vel={vel:.2f}, idle={idle:.2f})"
        )

    # Issue 011: hi_jit — jitter above the #25 0.28 cap, ~119 events
    if (
        100 <= events <= 145
        and 1.05 <= vel <= 1.50
        and jitter >= 0.28
        and jerk >= 0.18
        and acc >= 0.32
    ):
        return True, (
            f"High-jitter Playwright mid-count (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    # Issue 012: smooth_low_jit — jitter below the #25 0.14 floor, ~115 events
    if (
        100 <= events <= 160
        and 1.05 <= vel <= 1.50
        and jitter <= 0.14
        and jerk >= 0.18
        and acc >= 0.32
    ):
        return True, (
            f"Smooth low-jitter Playwright hops (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    # Issue 013: hi_jit_long — overflow of the #25 event cap (~252 events)
    if (
        220 <= events <= 340
        and 1.05 <= vel <= 1.50
        and 0.24 <= jitter <= 0.36
        and jerk >= 0.18
        and acc >= 0.32
    ):
        return True, (
            f"High-jitter long Playwright trace (events={events}, vel={vel:.2f}, jitter={jitter:.2f})"
        )

    return False, ""


def _sanitize_live_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        x = round(float(raw.get("x")), 2)
        y = round(float(raw.get("y")), 2)
        t = float(raw.get("t", 0.0))
        et = str(raw.get("type", raw.get("event_type", "mousemove")))
    except (TypeError, ValueError):
        return None
    evt: Dict[str, Any] = {"x": x, "y": y, "t": t, "type": et}
    if raw.get("pid") is not None or raw.get("pointer_id") is not None:
        evt["pid"] = raw.get("pid", raw.get("pointer_id"))
    if raw.get("pt") or raw.get("pointer_type"):
        evt["pt"] = raw.get("pt") or raw.get("pointer_type")
    if raw.get("src"):
        evt["src"] = raw.get("src")
    return evt


def ingest_live_pointer_events(
    session: Dict[str, Any],
    raw_events: List[Dict[str, Any]],
) -> Tuple[bool, str, int]:
    """Accept a small live canvas frame. Reject dumped JSON batches."""
    live = session.setdefault("live_events", [])
    recv = session.setdefault("live_recv", [])
    if session.get("verified"):
        return False, "Challenge session already completed.", len(live)
    if not isinstance(raw_events, list) or not raw_events:
        return False, "Empty live telemetry frame.", len(live)
    if len(raw_events) > MAX_LIVE_BATCH:
        return False, "Live canvas frame too large (JSON dump rejected).", len(live)
    now = time.time()
    for raw in raw_events:
        if not isinstance(raw, dict):
            return False, "Invalid live telemetry event.", len(live)
        evt = _sanitize_live_event(raw)
        if evt is None:
            return False, "Invalid live telemetry event.", len(live)
        live.append(evt)
        recv.append(now)
    return True, "", len(live)


def live_events_for_session(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(session.get("live_events") or [])


def _live_arrival_ok(session: Dict[str, Any], events: List[Dict[str, Any]]) -> Tuple[bool, str]:
    recv = session.get("live_recv") or []
    duration_ms = _event_duration_ms(events)
    if duration_ms < 400 or len(recv) < 2:
        return True, ""
    recv_span_ms = max(0.0, (float(recv[-1]) - float(recv[0])) * 1000.0)
    if recv_span_ms < duration_ms * LIVE_ARRIVAL_RATIO:
        return False, "Telemetry duration exceeds live canvas arrival time (injected stream)."
    return True, ""


def validate_verify_request(
    session: Dict[str, Any],
    events: List[Dict[str, Any]],
    verify_token: Optional[str],
    client_started_ms: Optional[float],
    client_submitted_ms: Optional[float],
    source: Optional[str],
    participant: str,
) -> Tuple[bool, str, List[str]]:
    flags: List[str] = []
    now = time.time()

    if session.get("verified"):
        return False, "Challenge session already completed.", flags

    attempts = int(session.get("verify_attempts", 0))
    if attempts >= MAX_VERIFY_ATTEMPTS:
        return False, "Maximum verification attempts exceeded for this session.", flags

    expected_token = session.get("verify_token")
    if CAPTCHA_STRICT and expected_token:
        if not verify_token or verify_token != expected_token:
            return False, "Invalid or missing verify token.", flags

    created_at = float(session.get("created_at", now))
    if now - created_at > SESSION_TTL_SEC:
        return False, "Challenge session expired.", flags

    if CAPTCHA_STRICT and source != "dashboard":
        return False, "Verification must originate from the dashboard client.", flags

    if len(events) < 4:
        return False, "Insufficient telemetry events.", flags

    duration_ms = _event_duration_ms(events)
    ctype = session.get("challenge_type", "")
    session_age_ms = (now - created_at) * 1000.0

    if CAPTCHA_STRICT:
        if not _has_pointer_attestation(events):
            return False, "Missing browser pointer attestation (live interaction required).", flags

        downs, ups = _pointer_down_up_count(events)
        if ctype in CLICK_CHALLENGE_TYPES and downs < 1:
            return False, "Click challenge requires pointer down events from the canvas.", flags
        if ctype in DRAG_CHALLENGE_TYPES and downs < 1:
            return False, "Drag challenge requires a pointer drag stroke from the canvas.", flags

        # Synthetic API bots submit instantly — telemetry span cannot exceed wall-clock session age.
        if duration_ms > 0 and session_age_ms < duration_ms * 0.65:
            return False, "Telemetry duration exceeds wall-clock session time (injected stream).", flags

        arrival_ok, arrival_reason = _live_arrival_ok(session, events)
        if not arrival_ok:
            return False, arrival_reason, flags

        if session_age_ms < MIN_SESSION_AGE_MS:
            return False, "Interaction completed too quickly for a live attempt.", flags

        if client_started_ms is not None and client_submitted_ms is not None:
            wall_ms = float(client_submitted_ms) - float(client_started_ms)
            if wall_ms < MIN_SESSION_AGE_MS:
                return False, "Client interaction window too short.", flags
            if duration_ms > 0:
                ratio = wall_ms / duration_ms
                if ratio < (1.0 - WALL_CLOCK_TOLERANCE) or ratio > (1.0 + WALL_CLOCK_TOLERANCE) + 2.0:
                    flags.append(
                        f"Wall-clock mismatch (wall={wall_ms:.0f}ms vs telemetry={duration_ms:.0f}ms)"
                    )
                    return False, "Telemetry timing does not match live session duration.", flags
        else:
            return False, "Missing client timing attestation.", flags

        if ctype in CLICK_CHALLENGE_TYPES and duration_ms < 800.0:
            return False, "Telemetry duration too short for a genuine click challenge.", flags

        micro_ratio = _micro_jiggle_ratio(events)
        # Soft signal only. Hard-rejecting here blocked humans (hover, rounding,
        # double-click) more often than it caught Playwright. Let ML/heuristics score.
        if micro_ratio >= MICRO_JIGGLE_RATIO_THRESHOLD:
            flags.append(f"High micro-jiggle ratio ({micro_ratio:.2f})")

        # Issue 005 dart hops use steps > 2px, so the micro-jiggle gate above may not fire.
        if ctype in CLICK_CHALLENGE_TYPES:
            try:
                from backend.feature_extractor import extract_features

                kin = extract_features(events)
                dart_reject, dart_reason = _automation_kinematic_reject(kin)
                if dart_reject and "Dart-hop" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright dart-hop automation.", flags
                if dart_reject and "Min-jerk dense overflow" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright min-jerk dense overflow.", flags
                if dart_reject and "Sparse irregular" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright sparse-irregular automation.", flags
                if dart_reject and "Mid-count" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright mid-count automation.", flags
                if dart_reject and "High-str fast" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright high-str-fast automation.", flags
                if dart_reject and "Idle-correct" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright idle-correct automation.", flags
                if dart_reject and "High-jitter" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright high-jitter automation.", flags
                if dart_reject and "Smooth low-jitter" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright smooth low-jitter automation.", flags
                if dart_reject and "High-jitter long" in dart_reason:
                    flags.append(dart_reason)
                    return False, "Detected Playwright high-jitter-long automation.", flags
            except Exception:
                pass

        suspicious_participants = ("bot_", "evasion", "script", "api_inject")
        pl = participant.lower()
        if any(p in pl for p in suspicious_participants):
            return False, "Invalid participant label for human verification.", flags

    return True, "", flags


def is_open_telemetry_blocked() -> bool:
    return CAPTCHA_STRICT
