"""
Red/blue adversarial loop — Fitts-law evasion vs hardened detector.

Session: 60 min cap (ADVERSARIAL_SESSION_SEC)
Cycle: 20 min red team (ADVERSARIAL_CYCLE_SEC)
Target: >=70% avg human confidence across all challenge types

On cycle pass: patch detector early, start next cycle.
On cycle fail (20 min): stop session and report.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from test.adversarial_utils import (
    detection_rate,
    human_confidence_from_events,
    load_classifier,
)
from test.evasion_bot import default_cfg, generate_events, mutate_cfg
from test.fitts_motor import FittsMotorConfig
from test.patch_detector import patch_detector, sample_evasion_features

SESSION_SEC = int(os.getenv("ADVERSARIAL_SESSION_SEC", "3600"))
CYCLE_SEC = int(os.getenv("ADVERSARIAL_CYCLE_SEC", "1200"))
TARGET_HC = float(os.getenv("ADVERSARIAL_TARGET_HC", "0.70"))
BLUE_DETECTION_TARGET = float(os.getenv("ADVERSARIAL_BLUE_TARGET", "0.70"))
VALIDATION_RUNS_PER_TYPE = int(os.getenv("ADVERSARIAL_VALIDATION_PER_TYPE", "2"))
PROBE_RUNS = int(os.getenv("ADVERSARIAL_PROBE_RUNS", "6"))
REPORT_PATH = os.path.join(ROOT, "test", "adversarial_report.json")
LOG_PATH = os.path.join(ROOT, "test", "adversarial_log.txt")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def score_mixed(
    motor: FittsMotorConfig,
    classifier,
    runs: int,
    rng: random.Random,
) -> Tuple[float, float, List[float]]:
    scores: List[float] = []
    for _ in range(runs):
        ctype = rng.choice(CHALLENGE_TYPES)
        challenge = generate_challenge(
            seed=rng.randint(1, 2_000_000_000),
            challenge_type=ctype,
        )
        events = generate_events(challenge, motor)
        hc, _ = human_confidence_from_events(events, classifier)
        scores.append(hc)
    avg = sum(scores) / len(scores)
    return avg, max(scores), scores


def score_all_types(
    motor: FittsMotorConfig,
    classifier,
    rng: random.Random,
    runs_per_type: int = VALIDATION_RUNS_PER_TYPE,
) -> Dict[str, Any]:
    by_type: Dict[str, List[float]] = {}
    for ctype in CHALLENGE_TYPES:
        scores: List[float] = []
        for _ in range(runs_per_type):
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type=ctype,
            )
            events = generate_events(challenge, motor)
            hc, _ = human_confidence_from_events(events, classifier)
            scores.append(hc)
        by_type[ctype] = scores

    type_avgs = {k: sum(v) / len(v) for k, v in by_type.items()}
    all_scores = [s for vals in by_type.values() for s in vals]
    return {
        "overall_avg": sum(all_scores) / len(all_scores),
        "overall_max": max(all_scores),
        "min_type_avg": min(type_avgs.values()),
        "type_avgs": {k: round(v, 4) for k, v in type_avgs.items()},
        "by_type": by_type,
        "scores": all_scores,
    }


def optimize_red(
    classifier,
    cycle_deadline: float,
    seed_motor: FittsMotorConfig,
    rng: random.Random,
) -> Dict[str, Any]:
    best = seed_motor
    baseline = score_mixed(best, classifier, 4, rng)
    best_avg, best_max = baseline[0], baseline[1]
    attempts = 0
    log(f"Red baseline: mixed avg={best_avg:.4f} max={best_max:.4f}")

    while time.time() < cycle_deadline:
        attempts += 1
        candidate = mutate_cfg(best, rng, scale=1.0 if attempts % 4 else 1.5)
        probe_avg, probe_max, _ = score_mixed(candidate, classifier, PROBE_RUNS, rng)
        if probe_avg > best_avg or (probe_avg == best_avg and probe_max > best_max):
            best = candidate
            best_avg = probe_avg
            best_max = probe_max
            log(f"Red attempt {attempts}: probe avg={best_avg:.4f} max={best_max:.4f}")

        if best_avg >= TARGET_HC:
            validation = score_all_types(best, classifier, rng, runs_per_type=VALIDATION_RUNS_PER_TYPE)
            log(
                f"Red probe hit target — validating: overall_avg={validation['overall_avg']:.4f} "
                f"min_type={validation['min_type_avg']:.4f}"
            )
            if validation["overall_avg"] >= TARGET_HC:
                return {
                    "success": True,
                    "motor": best,
                    "avg": validation["overall_avg"],
                    "max": validation["overall_max"],
                    "min_type_avg": validation["min_type_avg"],
                    "type_avgs": validation["type_avgs"],
                    "scores": validation["scores"],
                    "attempts": attempts,
                }

    validation = score_all_types(best, classifier, rng, runs_per_type=VALIDATION_RUNS_PER_TYPE)
    return {
        "success": validation["overall_avg"] >= TARGET_HC,
        "motor": best,
        "avg": validation["overall_avg"],
        "max": validation["overall_max"],
        "min_type_avg": validation["min_type_avg"],
        "type_avgs": validation["type_avgs"],
        "scores": validation["scores"],
        "attempts": attempts,
        "cycle_timeout": True,
    }


def evaluate_blue(motor: FittsMotorConfig, classifier, rng: random.Random) -> Dict[str, Any]:
    scores: List[float] = []
    for ctype in CHALLENGE_TYPES:
        for _ in range(2):
            challenge = generate_challenge(
                seed=rng.randint(1, 2_000_000_000),
                challenge_type=ctype,
            )
            events = generate_events(challenge, motor)
            hc, _ = human_confidence_from_events(events, classifier)
            scores.append(hc)
    rate = detection_rate(scores, threshold=0.50)
    return {
        "detection_rate": round(rate, 4),
        "avg_human_confidence": round(sum(scores) / len(scores), 4),
        "max_human_confidence": round(max(scores), 4),
        "scores": [round(s, 4) for s in scores],
        "blue_success": rate >= BLUE_DETECTION_TARGET,
    }


def save_report(report: Dict[str, Any]) -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def main() -> None:
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    session_start = time.time()
    session_deadline = session_start + SESSION_SEC
    rng = random.Random(20260816 + 70)
    classifier = load_classifier("random_forest_challenge")

    report: Dict[str, Any] = {
        "session_sec_limit": SESSION_SEC,
        "cycle_sec_limit": CYCLE_SEC,
        "target_human_confidence": TARGET_HC,
        "blue_detection_target": BLUE_DETECTION_TARGET,
        "motor": "fitts_minimum_jerk",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycles": [],
        "outcome": None,
    }

    cycle_num = 0
    seed_motor = default_cfg()

    log("=" * 72)
    log(f"Session {SESSION_SEC}s | cycles {CYCLE_SEC}s | target HC>={TARGET_HC}")
    log("Motor: Fitts-law + minimum-jerk + tremor/overshoot")

    while time.time() < session_deadline:
        cycle_num += 1
        cycle_start = time.time()
        cycle_deadline = min(cycle_start + CYCLE_SEC, session_deadline)
        log(f"--- Cycle {cycle_num} (red team, up to {CYCLE_SEC}s) ---")

        red = optimize_red(classifier, cycle_deadline, seed_motor, rng)
        record = {
            "cycle": cycle_num,
            "red_success": red["success"],
            "red_avg_hc": round(red["avg"], 4),
            "red_max_hc": round(red["max"], 4),
            "red_min_type_avg": round(red.get("min_type_avg", 0), 4),
            "red_type_avgs": red.get("type_avgs", {}),
            "red_attempts": red["attempts"],
            "motor_cfg": asdict(red["motor"]),
            "cycle_timeout": red.get("cycle_timeout", False),
            "elapsed_sec": round(time.time() - cycle_start, 2),
        }

        if red["success"]:
            log(f"Cycle {cycle_num} RED PASS avg={red['avg']:.4f} — blue team patch")
            features = sample_evasion_features(red["motor"], seed=cycle_num)
            patch_info = patch_detector(red["motor"], cycle_num, feature_sample=features)
            record["patch"] = patch_info
            classifier = load_classifier("random_forest_challenge")
            blue = evaluate_blue(red["motor"], classifier, rng)
            record["blue_eval"] = blue
            log(
                f"Cycle {cycle_num} BLUE detection={blue['detection_rate']:.4f} "
                f"avg_hc={blue['avg_human_confidence']:.4f}"
            )
            seed_motor = mutate_cfg(red["motor"], rng, scale=0.6)
            report["cycles"].append(record)
            save_report(report)
            if time.time() >= session_deadline:
                break
            continue

        log(f"Cycle {cycle_num} RED FAIL avg={red['avg']:.4f} max={red['max']:.4f} — stopping session")
        record["red_scores_sample"] = [round(s, 4) for s in red["scores"][:20]]
        report["cycles"].append(record)
        report["outcome"] = "stopped_on_red_failure"
        save_report(report)
        break

    report["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["elapsed_sec"] = round(time.time() - session_start, 2)
    report["cycle_count"] = cycle_num
    if report["outcome"] is None:
        report["outcome"] = "session_time_expired"
    save_report(report)
    log(f"Finished outcome={report['outcome']} in {report['elapsed_sec']}s → {REPORT_PATH}")


if __name__ == "__main__":
    main()
