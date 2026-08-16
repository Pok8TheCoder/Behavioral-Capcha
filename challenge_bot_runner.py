"""
Run bot tiers against algorithmically random behavioral challenges.

Evaluates task completion plus ML human/bot classification via the challenge
test server (default http://127.0.0.1:8020).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))

from bots.challenge_solver import solve_challenge
from bots.event_utils import normalize_bot_events
from bots.tier2_linear import LinearBot
from bots.tier3_bezier import BezierBot
from bots.tier4_humanoid import HumanoidFittsBot
from challenge_server.generator import CHALLENGE_TYPES, generate_challenge, validate_completion
from backend.model import BehavioralClassifier


CHALLENGE_API = os.getenv("CHALLENGE_API", "http://127.0.0.1:8020")
DETECTION_API = os.getenv("BEHAVIORAL_DETECTION_API", "http://127.0.0.1:8010")


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_bot_on_challenge(
    bot_name: str,
    bot,
    challenge: dict,
    style: str,
    classifier: BehavioralClassifier | None = None,
) -> dict:
    bot.reset()
    solve_challenge(bot, challenge, style=style)
    events = normalize_bot_events(bot.events)
    task_result = validate_completion(challenge, events)

    ml_eval = {}
    if classifier is not None:
        from backend.feature_extractor import extract_features

        features = extract_features(events)
        ml_eval = classifier.evaluate(features)

    return {
        "bot": bot_name,
        "style": style,
        "challenge_type": challenge["challenge_type"],
        "seed": challenge["seed"],
        "task_passed": task_result.get("passed", False),
        "task_reason": task_result.get("reason", ""),
        "event_count": len(events),
        "ml_decision": ml_eval.get("decision"),
        "ml_is_bot": ml_eval.get("is_bot"),
        "ml_risk_score": ml_eval.get("risk_score"),
        "ml_bot_probability": ml_eval.get("bot_probability"),
        "ml_flags": ml_eval.get("flags", []),
    }


def run_benchmark(runs_per_type: int, seed: int | None, use_api: bool) -> list[dict]:
    classifier = BehavioralClassifier()
    results: list[dict] = []

    bots = [
        ("linear", LinearBot(steps=30, step_delay_sec=0.008)),
        ("bezier", BezierBot(steps=40, step_delay_sec=0.008)),
        ("humanoid", HumanoidFittsBot(target_width=40.0)),
    ]

    for challenge_type in CHALLENGE_TYPES:
        for run_idx in range(runs_per_type):
            challenge_seed = (seed + run_idx) if seed is not None else None
            challenge = generate_challenge(seed=challenge_seed, challenge_type=challenge_type)
            challenge["seed"] = challenge.get("seed", run_idx)

            for style_name, bot in bots:
                solve_style = "humanoid" if style_name == "humanoid" else "linear"
                result = evaluate_bot_on_challenge(style_name, bot, challenge, solve_style, classifier)
                results.append(result)
                print(
                    f"[{challenge_type}] {style_name:8} task={'PASS' if result['task_passed'] else 'FAIL'} "
                    f"ml={result['ml_decision']} risk={result['ml_risk_score']}"
                )

                if use_api:
                    try:
                        api_result = _post_json(
                            f"{CHALLENGE_API}/api/challenge/verify",
                            {
                                "session_id": challenge["session_id"],
                                "events": normalize_bot_events(bot.events),
                                "participant": f"bot_{style_name}",
                            },
                        )
                        result["api_task_passed"] = api_result.get("task_result", {}).get("passed")
                        result["api_ml_decision"] = api_result.get("evaluation", {}).get("decision")
                    except urllib.error.URLError as exc:
                        result["api_error"] = str(exc)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bots on random behavioral challenges")
    parser.add_argument("--runs", type=int, default=2, help="Runs per challenge type")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    parser.add_argument("--use-api", action="store_true", help="Also submit results to challenge server API")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    print("=" * 72)
    print("BEHAVIORAL CHALLENGE BOT BENCHMARK")
    print("=" * 72)

    results = run_benchmark(args.runs, args.seed, args.use_api)

    blocked = sum(1 for r in results if r.get("ml_is_bot"))
    task_pass = sum(1 for r in results if r.get("task_passed"))
    print(f"\nSummary: {len(results)} runs, {task_pass} task passes, {blocked} ML blocks")

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
