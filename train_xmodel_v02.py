"""Train Xmodel V0.2 — Ymodel-scale 60 Hz mix (all human + all bot sources)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.hz60 import downsample_events_to_hz
from backend.xmodel_registry import XMODEL_DIR, register_new_version
from train_isolation_forest import FEATURE_NAMES
from train_qmodel_v081g import XGB_PARAMS, load_human_os_and_dashboard
from train_random_forest_balanced import (
    BOT_DIRS_ALL,
    evaluate_bot_styles,
    load_labeled_windows,
    session_split,
)
import train_random_forest_balanced as trf

VERSION_ID = os.getenv("XMODEL_OUTPUT_ID", "xmodel_v0.2")
TARGET_HZ = float(os.getenv("XMODEL_TARGET_HZ", "60"))
MODEL_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}.joblib")
METADATA_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}_metadata.json")
FAMILY_DIR = os.path.join(ROOT, "datasets", "bot_families")
PREV_RULES = os.path.join(XMODEL_DIR, "xmodel_v0.1_evasion_rules.json")
FAMILY_CFG = os.path.join(FAMILY_DIR, "ymodel_v03_playwright_live_cfg.json")


def main() -> None:
    print("=" * 72)
    print(f"Train Xmodel V0.2 — GPU XGBoost, Ymodel-scale mix @ {TARGET_HZ:.0f} Hz")
    print("=" * 72)

    xgb_params = dict(XGB_PARAMS)
    xgb_params["device"] = "cuda"
    xgb_params["tree_method"] = "hist"
    print(f"[Train] XGBoost device={xgb_params['device']} target_hz={TARGET_HZ}", flush=True)

    family_export = {}
    if os.path.exists(FAMILY_CFG):
        with open(FAMILY_CFG, encoding="utf-8") as f:
            family_export = json.load(f)

    human_X, human_groups, human_summaries, human_counts = load_human_os_and_dashboard(
        target_hz=TARGET_HZ
    )
    bot_X: List[List[float]] = []
    bot_groups: List[str] = []
    bot_summaries: List[Dict[str, Any]] = []
    bot_windows_by_source: Dict[str, int] = {}
    for prefix, bot_dir in BOT_DIRS_ALL + [("bot_families", FAMILY_DIR)]:
        vectors, groups, summaries = load_labeled_windows(bot_dir, prefix, target_hz=TARGET_HZ)
        bot_X.extend(vectors)
        bot_groups.extend(groups)
        bot_summaries.extend(summaries)
        bot_windows_by_source[prefix] = len(vectors)

    if not human_X or not bot_X:
        raise SystemExit("Need both human and bot windows.")

    X = np.array(human_X + bot_X, dtype=np.float32)
    y = np.array([0] * len(human_X) + [1] * len(bot_X), dtype=np.int32)
    groups = np.array(human_groups + bot_groups)
    X_train, X_test, y_train, y_test = session_split(X, y, groups)

    print(f"[Train] Human windows: {len(human_X):,}  {human_counts}")
    print(f"[Train] Bot windows:   {len(bot_X):,}  {bot_windows_by_source}")
    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"[Train] Human fraction: {float(np.mean(y == 0)):.4f}  Bot fraction: {float(np.mean(y == 1)):.4f}")

    clf = XGBClassifier(**xgb_params)
    started = time.perf_counter()
    device_used = "cuda"
    try:
        clf.fit(X_train, y_train, verbose=False)
    except Exception as exc:
        print(f"[Train] CUDA fit failed ({exc}); retrying on CPU.")
        cpu_params = dict(xgb_params)
        cpu_params["device"] = "cpu"
        clf = XGBClassifier(**cpu_params)
        xgb_params = cpu_params
        device_used = "cpu"
        started = time.perf_counter()
        clf.fit(X_train, y_train, verbose=False)
    train_sec = time.perf_counter() - started
    print(f"[Train] Fit time: {train_sec:.2f}s on {device_used}", flush=True)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = float(roc_auc_score(y_test, y_proba))
    tn, fp, fn, tp = cm.ravel()
    acc = float((tn + tp) / len(y_test))
    print("=" * 72)
    print("Xmodel V0.2")
    print(f"Device: {device_used}  target_hz={TARGET_HZ:.0f}  ROC-AUC: {auc:.6f}  Accuracy: {acc:.6f}")
    print("Confusion [[TN, FP], [FN, TP]]:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"], zero_division=0))

    importances = sorted(
        zip(FEATURE_NAMES, [float(v) for v in clf.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )

    orig_extract = trf.extract_features

    def extract_60hz(events):
        return orig_extract(downsample_events_to_hz(events, TARGET_HZ))

    trf.extract_features = extract_60hz
    try:
        style_eval = evaluate_bot_styles(clf, episodes_per_style=10)
    finally:
        trf.extract_features = orig_extract

    os.makedirs(XMODEL_DIR, exist_ok=True)
    rules_name = f"{VERSION_ID}_evasion_rules.json"
    rules_path = os.path.join(XMODEL_DIR, rules_name)
    if os.path.exists(PREV_RULES) and not os.path.exists(rules_path):
        shutil.copy2(PREV_RULES, rules_path)

    metadata = {
        "model_type": "xgboost_gpu",
        "xmodel_id": VERSION_ID,
        "xmodel_label": "Xmodel V0.2",
        "xmodel_version": "0.2",
        "target_hz": TARGET_HZ,
        "trainer": "xgboost",
        "device": device_used,
        "balanced": False,
        "human_filter": "all_sources_downsampled_60hz",
        "dataset_mix": "ymodel_v03m_equivalent",
        "human_sources": human_counts,
        "family_export": family_export,
        "train_seconds": round(train_sec, 3),
        "xgb_params": xgb_params,
        "feature_names": FEATURE_NAMES,
        "human_windows_loaded": len(human_X),
        "bot_windows_loaded": len(bot_X),
        "bot_windows_by_source": bot_windows_by_source,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "accuracy": acc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Human", "Bot"], zero_division=0,
        ),
        "feature_importances": [{"feature": n, "importance": v} for n, v in importances],
        "bot_style_eval": style_eval,
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "metadata": metadata}, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    register_new_version(
        version_id=VERSION_ID,
        label="Xmodel V0.2",
        version="0.2",
        description=(
            "Ymodel V0.3M-scale training mix: all OS+dashboard human CSVs and all bot "
            "sources (challenge, tiers, evasion, families) with every window downsampled "
            "to 60 Hz for remote/Tailscale parity."
        ),
        patch_cycle=2,
        evasion_rules_filename=rules_name if os.path.exists(rules_path) else None,
        human_filter="all_sources_downsampled_60hz",
        make_latest=True,
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"[Save] {METADATA_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID} device={device_used} target_hz={TARGET_HZ:.0f}")


if __name__ == "__main__":
    main()
