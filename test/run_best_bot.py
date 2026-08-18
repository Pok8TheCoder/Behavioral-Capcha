"""Run the saved best bot config and report max human confidence."""

from __future__ import annotations

import json
from pathlib import Path

from test_best_config import BEST_CFG, api_post, cfg_from_dict, generate_events

RUNS = 20


def main() -> None:
    cfg = cfg_from_dict(BEST_CFG)
    scores: list[float] = []
    for i in range(RUNS):
        challenge = api_post("/api/challenge/new", {"challenge_type": "button_sequence"})
        events = generate_events(challenge, cfg)
        result = api_post(
            "/api/challenge/verify",
            {
                "session_id": challenge["session_id"],
                "events": events,
                "participant": "bot_custom_humanlike",
            },
        )
        hc = float(result.get("human_confidence") or 0)
        scores.append(hc)
        model = (result.get("evaluation") or {}).get("model_type")
        print(f"{i+1:02d} model={model} hc={hc:.4f}")

    summary = {
        "max_human_confidence": max(scores),
        "avg_human_confidence": sum(scores) / len(scores),
        "runs": RUNS,
        "scores": scores,
    }
    out = Path(__file__).with_name("session_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MAX", summary["max_human_confidence"])
    print("AVG", summary["avg_human_confidence"])


if __name__ == "__main__":
    main()
