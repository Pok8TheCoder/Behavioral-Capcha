"""Dynamic evasion heuristic rules — per-Qmodel version files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_RULES_PATH = os.path.join(os.path.dirname(__file__), "evasion_heuristic_rules.json")


def load_rules(rules_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = rules_path or DEFAULT_RULES_PATH
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def append_rule_from_features(
    cycle_id: int,
    features: Dict[str, Any],
    rules_path: Optional[str] = None,
) -> Dict[str, Any]:
    path = rules_path or DEFAULT_RULES_PATH
    rules = load_rules(path)
    rule = {
        "cycle": cycle_id,
        "straightness_max": min(0.35, features.get("straightness_index", 0.2) + 0.08),
        "straightness_min": max(0.0, features.get("straightness_index", 0.2) - 0.08),
        "dwell_min": max(60.0, features.get("avg_click_dwell_ms", 120) - 40),
        "duration_min": max(2000.0, features.get("duration_ms", 5000) - 1500),
        "event_count_min": max(10, int(features.get("event_count", 20) - 5)),
        "score_boost": 0.34,
        "label": f"evasion_cycle_{cycle_id:03d}",
    }
    jitter = features.get("jitter_ratio", 0.3)
    vel_mean = features.get("velocity_mean", 0.35)
    vel_max = features.get("velocity_max", 1.2)
    jerk_mean = features.get("jerk_mean", 0.3)
    acc_mean = features.get("acceleration_mean", 0.4)
    if jitter < 0.15:
        rule["jitter_min"] = 0.0
        rule["jitter_max"] = min(0.15, jitter + 0.06)
    elif jitter > 0.40 and vel_mean > 1.05:
        # Dart-hop high-jitter family (issue 005 fast_direct) — before generic high-jitter+low-jerk
        rule["jitter_min"] = max(0.38, jitter - 0.14)
        rule["jitter_max"] = min(0.70, jitter + 0.14)
        rule["jerk_mean_min"] = max(0.06, jerk_mean - 0.08)
        rule["jerk_mean_max"] = min(0.35, jerk_mean + 0.12)
        rule["acc_mean_min"] = max(0.16, acc_mean - 0.12)
        rule["acc_mean_max"] = min(0.55, acc_mean + 0.14)
        rule["vel_mean_min"] = max(1.00, vel_mean - 0.22)
        rule["vel_mean_max"] = min(1.70, vel_mean + 0.28)
        rule["event_count_min"] = max(70, int(features.get("event_count", 110) - 40))
        rule["event_count_max"] = int(features.get("event_count", 110) + 80)
        rule["straightness_max"] = min(0.40, features.get("straightness_index", 0.16) + 0.16)
    elif features.get("event_count", 0) >= 270 and jitter < 0.24 and vel_mean > 0.60:
        # Min-jerk dense overflow (issue 006 — events above #18 cap)
        n = int(features.get("event_count", 360))
        rule["jitter_min"] = 0.08
        rule["jitter_max"] = min(0.24, jitter + 0.08)
        rule["jerk_mean_min"] = max(0.08, jerk_mean - 0.08)
        rule["jerk_mean_max"] = min(0.32, jerk_mean + 0.10)
        rule["acc_mean_min"] = max(0.14, acc_mean - 0.10)
        rule["acc_mean_max"] = min(0.40, acc_mean + 0.12)
        rule["vel_mean_min"] = max(0.55, vel_mean - 0.25)
        rule["vel_mean_max"] = min(1.15, vel_mean + 0.25)
        rule["event_count_min"] = max(260, n - 80)
        rule["event_count_max"] = n + 160
        rule["straightness_max"] = min(0.35, features.get("straightness_index", 0.13) + 0.16)
    elif jitter > 0.40 and jerk_mean < 0.22:
        # Playwright hop+jiggle inflation (issue 002)
        rule["jitter_min"] = max(0.40, jitter - 0.15)
        rule["jerk_mean_max"] = min(0.25, jerk_mean + 0.10)
        rule["acc_mean_max"] = min(0.40, acc_mean + 0.12)
        rule["event_count_max"] = int(features.get("event_count", 200) + 80)
    elif 0.26 <= jitter <= 0.55 and vel_mean > 0.78 and jerk_mean < 0.14:
        # Fast-hop flat dynamics (issue 003 hi_dyn)
        rule["jitter_min"] = max(0.26, jitter - 0.12)
        rule["jitter_max"] = min(0.55, jitter + 0.12)
        rule["jerk_mean_max"] = min(0.16, jerk_mean + 0.06)
        rule["acc_mean_max"] = min(0.26, acc_mean + 0.08)
        rule["vel_mean_min"] = max(0.72, vel_mean - 0.22)
        rule["vel_mean_max"] = min(1.55, vel_mean + 0.22)
        rule["event_count_min"] = max(80, int(features.get("event_count", 140) - 30))
        rule["event_count_max"] = int(features.get("event_count", 140) + 70)
    elif jitter < 0.24 and jerk_mean > 0.08 and acc_mean > 0.18:
        # Dense-hop Playwright family (issue 004 — moderate dynamics, low jitter)
        rule["jitter_min"] = 0.0
        rule["jitter_max"] = min(0.24, jitter + 0.08)
        rule["jerk_mean_min"] = max(0.06, jerk_mean - 0.08)
        rule["jerk_mean_max"] = min(0.35, jerk_mean + 0.12)
        rule["acc_mean_min"] = max(0.15, acc_mean - 0.10)
        rule["acc_mean_max"] = min(0.48, acc_mean + 0.12)
        rule["vel_mean_min"] = max(0.45, vel_mean - 0.25)
        rule["vel_mean_max"] = min(1.25, vel_mean + 0.25)
        rule["event_count_min"] = max(50, int(features.get("event_count", 150) - 40))
        rule["event_count_max"] = int(features.get("event_count", 150) + 80)
        rule["straightness_max"] = min(0.50, features.get("straightness_index", 0.25) + 0.15)
    else:
        rule["jitter_min"] = max(0.15, jitter - 0.08)
    if "vel_mean_min" not in rule:
        rule["vel_mean_min"] = max(0.15, vel_mean - 0.15)
    if "vel_mean_max" not in rule:
        rule["vel_mean_max"] = min(1.05, vel_mean + 0.15)
    rule["vel_max_max"] = min(10.0, vel_max + 1.0)
    rules.append(rule)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
    return rule


def score_dynamic_rules(features: Dict[str, Any], rules: List[Dict[str, Any]]) -> tuple[float, List[str]]:
    boost = 0.0
    flags: List[str] = []
    for rule in rules:
        s = features.get("straightness_index", 0.0)
        if s > rule.get("straightness_max", 1.0):
            continue
        if s < rule.get("straightness_min", 0.0):
            continue
        if features.get("jitter_ratio", 0) < rule.get("jitter_min", 0):
            continue
        if rule.get("jitter_max") is not None and features.get("jitter_ratio", 0) > rule.get("jitter_max"):
            continue
        vm = features.get("velocity_mean", 0)
        if vm < rule.get("vel_mean_min", 0) or vm > rule.get("vel_mean_max", 99):
            continue
        if features.get("velocity_max", 0) > rule.get("vel_max_max", 99):
            continue
        dwell = features.get("avg_click_dwell_ms", -1)
        if dwell >= 0 and dwell < rule.get("dwell_min", 0):
            continue
        if features.get("duration_ms", 0) < rule.get("duration_min", 0):
            continue
        if features.get("event_count", 0) < rule.get("event_count_min", 0):
            continue
        if rule.get("event_count_max") is not None and features.get("event_count", 0) > rule.get("event_count_max"):
            continue
        if rule.get("jerk_mean_max") is not None and features.get("jerk_mean", 0) > rule.get("jerk_mean_max"):
            continue
        if rule.get("jerk_mean_min") is not None and features.get("jerk_mean", 0) < rule.get("jerk_mean_min"):
            continue
        if rule.get("acc_mean_max") is not None and features.get("acceleration_mean", 0) > rule.get("acc_mean_max"):
            continue
        if rule.get("acc_mean_min") is not None and features.get("acceleration_mean", 0) < rule.get("acc_mean_min"):
            continue
        if rule.get("straightness_max") is not None and features.get("straightness_index", 0) > rule.get("straightness_max"):
            continue
        boost = max(boost, rule.get("score_boost", 0.3))
        flags.append(f"Dynamic evasion rule match ({rule.get('label', 'unknown')})")
    return boost, flags
