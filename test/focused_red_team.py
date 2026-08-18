"""Focused red-team search on button_sequence until validation avg >= target."""

from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataclasses import asdict
from test.adversarial_utils import human_confidence_from_events, load_classifier
from test.evasion_bot import default_cfg, generate_events, mutate_cfg

TARGET = 0.80
VALIDATION_RUNS = 15
MAX_ATTEMPTS = 500
OUT = os.path.join(ROOT, "test", "focused_red_result.json")


def validate(cfg, clf, rng, runs=VALIDATION_RUNS):
    scores = []
    for _ in range(runs):
        ch = generate_challenge(
            seed=rng.randint(1, 2_000_000_000),
            challenge_type="button_sequence",
        )
        ev = generate_events(ch, cfg)
        hc, _ = human_confidence_from_events(ev, clf)
        scores.append(hc)
    avg = sum(scores) / len(scores)
    return avg, max(scores), scores


def main() -> None:
    rng = random.Random(42)
    clf = load_classifier("random_forest_challenge")
    best = default_cfg()
    best_avg, best_max, best_scores = validate(best, clf, rng, 8)
    print(f"baseline avg={best_avg:.4f} max={best_max:.4f}")

    for i in range(1, MAX_ATTEMPTS + 1):
        cand = mutate_cfg(best, rng, scale=1.0 if i % 4 else 1.5)
        probe_avg, probe_max, _ = validate(cand, clf, rng, 4)
        if probe_avg > best_avg or (probe_avg == best_avg and probe_max > best_max):
            best = cand
            best_avg = probe_avg
            best_max = probe_max
            print(f"attempt {i}: probe avg={best_avg:.4f} max={best_max:.4f}")

        if best_avg >= TARGET:
            val_avg, val_max, val_scores = validate(best, clf, rng, VALIDATION_RUNS)
            print(f"validation avg={val_avg:.4f} max={val_max:.4f}")
            if val_avg >= TARGET:
                out = {
                    "success": True,
                    "avg": val_avg,
                    "max": val_max,
                    "scores": val_scores,
                    "cfg": asdict(best),
                    "attempts": i,
                }
                with open(OUT, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                print("SUCCESS", OUT)
                return
    val_avg, val_max, val_scores = validate(best, clf, rng, VALIDATION_RUNS)
    out = {
        "success": val_avg >= TARGET,
        "avg": val_avg,
        "max": val_max,
        "scores": val_scores,
        "cfg": asdict(best),
        "attempts": MAX_ATTEMPTS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("DONE avg=", val_avg, "max=", val_max)


if __name__ == "__main__":
    main()
