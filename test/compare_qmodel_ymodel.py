"""Compare latest Qmodel vs Ymodel V0.1 on live winners and a fresh family holdout."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.feature_extractor import extract_features, feature_dict_to_vector
from backend.qmodel_registry import QMODEL_DIR, catalog_for_api as q_catalog
from backend.ymodel_registry import YMODEL_DIR, catalog_for_api as y_catalog
from challenge_server.generator import generate_challenge
from test.generate_bot_families import NAMED_FAMILIES, random_family
from test.sparse_irregular_bot import generate_events

OUT_PATH = os.path.join(ROOT, "test", "ymodel_v01_vs_qmodel.json")


def _load_clf(path: str):
    package = joblib.load(path)
    return package["model"] if isinstance(package, dict) else package


def _bot_prob(clf, features: dict) -> float:
    vec = np.array([feature_dict_to_vector(features)], dtype=np.float32)
    return float(clf.predict_proba(vec)[0][1])


def _live_winners() -> list[dict]:
    rows = []
    for path in sorted(Path(ROOT, "test").glob("timed_v*_live_result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        feats = data.get("features") or {}
        if not feats:
            continue
        mh = data.get("model_human")
        if mh is None:
            mh = data.get("best_model_human")
        if mh is None:
            continue
        rows.append({
            "file": path.name,
            "attacked_model": data.get("model"),
            "family": data.get("family") or data.get("best_family"),
            "live_model_human": mh,
            "features": feats,
        })
    return rows


def main() -> None:
    q_latest = q_catalog().get("latest") or "qmodel_v0.15"
    y_latest = y_catalog().get("latest") or "ymodel_v0.1"
    q_path = os.path.join(QMODEL_DIR, f"{q_latest}.joblib")
    y_path = os.path.join(YMODEL_DIR, f"{y_latest}.joblib")
    if not os.path.exists(q_path):
        raise SystemExit(f"Missing Qmodel artifact: {q_path}")
    if not os.path.exists(y_path):
        raise SystemExit(f"Missing Ymodel artifact: {y_path}")

    q_clf = _load_clf(q_path)
    y_clf = _load_clf(y_path)

    winner_rows = []
    q_beats = y_beats = 0
    for row in _live_winners():
        q_p = _bot_prob(q_clf, row["features"])
        y_p = _bot_prob(y_clf, row["features"])
        item = {
            **{k: v for k, v in row.items() if k != "features"},
            "qmodel_bot_prob": round(q_p, 4),
            "ymodel_bot_prob": round(y_p, 4),
            "qmodel_human": round(1.0 - q_p, 4),
            "ymodel_human": round(1.0 - y_p, 4),
        }
        if q_p < 0.30:
            q_beats += 1
        if y_p < 0.30:
            y_beats += 1
        winner_rows.append(item)

    rng = random.Random(2099)
    holdout = []
    jobs = list(NAMED_FAMILIES.values()) + [random_family(rng, i) for i in range(24)]
    for cfg in jobs:
        challenge = generate_challenge(seed=rng.randint(1, 2_000_000_000), challenge_type="button_sequence")
        events = generate_events(challenge, cfg)
        feats = extract_features(events)
        q_p = _bot_prob(q_clf, feats)
        y_p = _bot_prob(y_clf, feats)
        holdout.append({
            "family": cfg.name,
            "event_count": feats.get("event_count"),
            "velocity_mean": round(float(feats.get("velocity_mean", 0.0)), 4),
            "jitter_ratio": round(float(feats.get("jitter_ratio", 0.0)), 4),
            "qmodel_human": round(1.0 - q_p, 4),
            "ymodel_human": round(1.0 - y_p, 4),
            "qmodel_bot_prob": round(q_p, 4),
            "ymodel_bot_prob": round(y_p, 4),
        })

    q_hold_mean = float(np.mean([r["qmodel_human"] for r in holdout])) if holdout else 0.0
    y_hold_mean = float(np.mean([r["ymodel_human"] for r in holdout])) if holdout else 0.0
    q_hold_pass = sum(1 for r in holdout if r["qmodel_human"] >= 0.70)
    y_hold_pass = sum(1 for r in holdout if r["ymodel_human"] >= 0.70)

    report = {
        "qmodel": q_latest,
        "ymodel": y_latest,
        "note": "ML-only (1 - predict_proba bot). Heuristics/403s are not applied.",
        "live_winner_fingerprints": {
            "count": len(winner_rows),
            "qmodel_still_under_0.30_botprob": q_beats,
            "ymodel_still_under_0.30_botprob": y_beats,
            "rows": winner_rows,
        },
        "fresh_family_holdout": {
            "episodes": len(holdout),
            "qmodel_mean_human": round(q_hold_mean, 4),
            "ymodel_mean_human": round(y_hold_mean, 4),
            "qmodel_ge_70_human": q_hold_pass,
            "ymodel_ge_70_human": y_hold_pass,
            "rows": holdout,
        },
    }
    Path(OUT_PATH).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "fresh_family_holdout"}, indent=2))
    print("holdout", json.dumps(report["fresh_family_holdout"] | {"rows": f"{len(holdout)} omitted"}, indent=2))
    print(f"COMPARE_COMPLETE {OUT_PATH}")


if __name__ == "__main__":
    main()
