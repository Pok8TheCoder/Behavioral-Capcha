"""One classifier replica per bot family so arena attacks can run in parallel."""

from __future__ import annotations

import copy
import os
import secrets
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from backend.active_window import (
    DEFAULT_ACTIVE_WINDOW_CONFIG,
    _split_on_gaps,
    aggregate_window_evaluations,
    segment_episode_into_windows,
)
from backend.hz60 import downsample_events_to_hz
from backend.model import BehavioralClassifier
from bots.dashboard_bots import run_dashboard_bot
from bots.event_utils import normalize_bot_events
from challenge_server.generator import generate_challenge, validate_completion

RECORD_HZ = 60.0


def load_replica(model_id: str) -> BehavioralClassifier:
    clf = BehavioralClassifier(autoload=False)
    if not clf.load_model(model_id):
        raise RuntimeError(f"Failed to load model replica '{model_id}'")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return clf


def clone_replica(src: BehavioralClassifier) -> BehavioralClassifier:
    dst = BehavioralClassifier(autoload=False)
    dst.model = copy.deepcopy(src.model)
    dst.model_type = src.model_type
    dst.display_name = src.display_name
    dst.feature_names = list(src.feature_names or [])
    dst.metadata = copy.deepcopy(src.metadata) if isinstance(src.metadata, dict) else {}
    dst.is_trained = src.is_trained
    dst.evasion_rules_path = src.evasion_rules_path
    dst.use_procedural_mimic_heuristic = src.use_procedural_mimic_heuristic
    dst.feature_schema = src.feature_schema
    dst.scaler = copy.deepcopy(src.scaler) if src.scaler is not None else None
    dst.use_scaler = src.use_scaler
    dst._z_events = None
    dst._z_device = getattr(src, "_z_device", None)
    model = dst.model
    if model is not None and hasattr(model, "to") and dst._z_device is not None:
        model.to(dst._z_device)
        model.eval()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return dst


def reload_replica(clf: BehavioralClassifier, model_id: str) -> None:
    if not clf.load_model(model_id):
        raise RuntimeError(f"Failed to reload model replica '{model_id}'")


def _scored_windows(
    classifier: BehavioralClassifier,
    events: List[Dict[str, Any]],
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    cfg = DEFAULT_ACTIVE_WINDOW_CONFIG
    meta = classifier.metadata if isinstance(getattr(classifier, "metadata", None), dict) else {}
    hz = meta.get("target_hz")
    scored = downsample_events_to_hz(events, float(hz)) if hz else events
    sorted_events = sorted(scored, key=lambda e: float(e["t"]))
    episodes = _split_on_gaps(sorted_events, cfg.afk_gap_ms)
    out: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for episode in episodes:
        for window in segment_episode_into_windows(episode, cfg):
            out.append((window, classifier.evaluate_window(window)))
    if not out and sorted_events:
        out.append((sorted_events, classifier.evaluate_window(sorted_events)))
    return out


def evaluate_event_windows(
    classifier: BehavioralClassifier,
    events: List[Dict[str, Any]],
    win_hc: float = 0.65,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, (window, evaluation) in enumerate(_scored_windows(classifier, events)):
        bot_prob = evaluation.get("bot_probability")
        hc = round(1.0 - float(bot_prob), 4) if bot_prob is not None else 0.0
        t0 = float(window[0]["t"]) if window else 0.0
        t1 = float(window[-1]["t"]) if window else 0.0
        rows.append({
            "index": index,
            "human_confidence": hc,
            "bot_probability": None if bot_prob is None else round(float(bot_prob), 4),
            "passed": hc >= win_hc,
            "event_count": len(window),
            "duration_ms": round(max(0.0, t1 - t0), 1),
            "events": window,
        })
    return rows


def evaluate_events(classifier: BehavioralClassifier, events: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    cfg = DEFAULT_ACTIVE_WINDOW_CONFIG
    meta = classifier.metadata if isinstance(getattr(classifier, "metadata", None), dict) else {}
    hz = meta.get("target_hz")
    scored = downsample_events_to_hz(events, float(hz)) if hz else events
    sorted_events = sorted(scored, key=lambda e: float(e["t"]))
    episodes = _split_on_gaps(sorted_events, cfg.afk_gap_ms)

    evaluations: List[Dict[str, Any]] = []
    primary_features: Dict[str, Any] = {}

    for episode in episodes:
        for window in segment_episode_into_windows(episode, cfg):
            features = classifier.features_from_events(window)
            evaluation = classifier.evaluate_window(window)
            evaluations.append(evaluation)
            if not primary_features or evaluation.get("risk_score", 0) >= primary_features.get("_risk", 0):
                primary_features = {**features, "_risk": evaluation.get("risk_score", 0)}

    if not evaluations:
        features = classifier.features_from_events(sorted_events)
        return classifier.evaluate_window(sorted_events), features, 1

    primary_features.pop("_risk", None)
    aggregated = aggregate_window_evaluations(evaluations, strategy="max_risk")
    return aggregated, primary_features, len(evaluations)


def run_local_attempt(
    classifier: BehavioralClassifier,
    bot_id: str,
    challenge_type: str = "random",
    record_hz: float = RECORD_HZ,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    chosen = None if not challenge_type or challenge_type == "random" else challenge_type
    seed = secrets.randbelow(2_000_000_000) + 1
    challenge = generate_challenge(seed=seed, challenge_type=chosen)
    stamped = run_dashboard_bot(challenge, bot_id)
    events = downsample_events_to_hz(normalize_bot_events(stamped), float(record_hz))
    evaluation, _features, window_count = evaluate_events(classifier, events)
    task_result = validate_completion(challenge, events)
    bot_prob = evaluation.get("bot_probability")
    hc = round(1.0 - float(bot_prob), 4) if bot_prob is not None else 0.0
    result = {
        "bot": bot_id,
        "challenge_type": challenge.get("challenge_type"),
        "seed": challenge.get("seed"),
        "human_confidence": hc,
        "task_passed": bool(task_result.get("passed")),
        "event_count": len(events),
        "windows": window_count,
        "win": False,
        "mark": False,
    }
    return result, events
