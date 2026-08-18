"""One-shot blue-team patch using sharepad BEST_CFG evasion bot."""

from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test.adversarial_utils import detection_rate, human_confidence_from_events, load_classifier
from test.evasion_bot import generate_events
from test.patch_detector import export_evasion_dataset, retrain_detector
from test.test_best_config import BEST_CFG, cfg_from_dict
from challenge_server.generator import generate_challenge

PRE_PATCH_RUNS = 12
POST_PATCH_RUNS = 12


def main() -> None:
    cfg = cfg_from_dict(BEST_CFG)
    clf = load_classifier("random_forest_challenge")

    pre_scores = []
    for i in range(PRE_PATCH_RUNS):
        ch = generate_challenge(
            seed=1000 + i,
            challenge_type="button_sequence",
        )
        ev = generate_events(ch, cfg)
        hc, _ = human_confidence_from_events(ev, clf)
        pre_scores.append(hc)
    pre_det = detection_rate(pre_scores)

    export_info = export_evasion_dataset(cfg, cycle_id=1, episodes_per_type=6, seed=99)
    train_info = retrain_detector()

    clf = load_classifier("random_forest_challenge")
    post_scores = []
    for i in range(POST_PATCH_RUNS):
        ch = generate_challenge(
            seed=2000 + i,
            challenge_type="button_sequence",
        )
        ev = generate_events(ch, cfg)
        hc, _ = human_confidence_from_events(ev, clf)
        post_scores.append(hc)
    post_det = detection_rate(post_scores)

    out = {
        "pre_patch": {
            "avg_hc": sum(pre_scores) / len(pre_scores),
            "max_hc": max(pre_scores),
            "detection_rate": pre_det,
            "scores": pre_scores,
        },
        "post_patch": {
            "avg_hc": sum(post_scores) / len(post_scores),
            "max_hc": max(post_scores),
            "detection_rate": post_det,
            "scores": post_scores,
        },
        "export": export_info,
        "train_exit_code": train_info.get("exit_code"),
    }
    path = os.path.join(ROOT, "test", "blue_team_patch_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
