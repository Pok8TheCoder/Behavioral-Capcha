"""Shared local evaluation helpers for adversarial red/blue testing."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.active_window import (
    DEFAULT_ACTIVE_WINDOW_CONFIG,
    aggregate_window_evaluations,
    segment_episode_into_windows,
    _split_on_gaps,
)
from backend.feature_extractor import extract_features
from backend.model import BehavioralClassifier
from bots.event_utils import normalize_bot_events


def load_classifier(model_name: str = "random_forest_challenge") -> BehavioralClassifier:
    clf = BehavioralClassifier()
    if not clf.load_model(model_name):
        raise RuntimeError(f"Unable to load model: {model_name}")
    return clf


def evaluate_events_dynamic(
    events: List[Dict[str, Any]],
    classifier: BehavioralClassifier,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    cfg = DEFAULT_ACTIVE_WINDOW_CONFIG
    normalized = normalize_bot_events(events)
    meta = classifier.metadata if isinstance(getattr(classifier, "metadata", None), dict) else {}
    hz = meta.get("target_hz")
    if hz:
        from backend.hz60 import downsample_events_to_hz
        normalized = downsample_events_to_hz(normalized, float(hz))
    sorted_events = sorted(normalized, key=lambda e: float(e["t"]))
    episodes = _split_on_gaps(sorted_events, cfg.afk_gap_ms)

    evaluations: List[Dict[str, Any]] = []
    primary_features: Dict[str, Any] = {}

    for episode in episodes:
        for window in segment_episode_into_windows(episode, cfg):
            features = extract_features(window)
            evaluation = classifier.evaluate_window(window)
            evaluations.append(evaluation)
            if not primary_features or evaluation.get("risk_score", 0) >= primary_features.get("_risk", 0):
                primary_features = {**features, "_risk": evaluation.get("risk_score", 0)}

    if not evaluations:
        features = extract_features(sorted_events)
        return classifier.evaluate_window(sorted_events), features, 1

    primary_features.pop("_risk", None)
    aggregated = aggregate_window_evaluations(evaluations, strategy="max_risk")
    return aggregated, primary_features, len(evaluations)


def human_confidence_from_events(
    events: List[Dict[str, Any]],
    classifier: BehavioralClassifier,
) -> Tuple[float, Dict[str, Any]]:
    evaluation, features, window_count = evaluate_events_dynamic(events, classifier)
    risk = float(evaluation.get("risk_score", 0.0))
    hc = round(1.0 - risk, 4)
    return hc, {
        "human_confidence": hc,
        "evaluation": evaluation,
        "features": features,
        "active_windows_scored": window_count,
    }


def detection_rate(
    scores: List[float],
    threshold: float = 0.50,
) -> float:
    """Fraction of attempts classified as bot (human_confidence below threshold)."""
    if not scores:
        return 0.0
    detected = sum(1 for s in scores if s < threshold)
    return detected / len(scores)
