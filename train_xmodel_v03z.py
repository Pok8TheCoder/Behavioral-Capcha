"""Train Xmodel V0.3z — same V0.3 dataset, Zmodel-style 1D CNN on relative sequences."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.active_window import DEFAULT_ACTIVE_WINDOW_CONFIG, segment_episode_into_windows
from backend.feature_extractor import extract_features, feature_dict_to_vector
from backend.xmodel_registry import (
    XMODEL_DIR,
    parse_ea_cycle,
    register_new_version,
    xmodel_display_label,
    xmodel_version_token,
)
from backend.zmodel_net import ZNetV01
from backend.zmodel_seq import SEQ_LEN, events_to_relative_sequence
from train_isolation_forest import FEATURE_NAMES, MIN_EVENTS_PER_WINDOW, WindowSample
import train_isolation_forest as tif

PARENT_ID = os.getenv("XMODEL_PARENT_ID", "xmodel_v0.3z").strip() or "xmodel_v0.3z"
VERSION_ID = os.getenv("XMODEL_OUTPUT_ID", "").strip() or PARENT_ID
TARGET_HZ = float(os.getenv("XMODEL_TARGET_HZ", "60"))
BATCH_SIZE = int(os.getenv("XMODEL_Z_BATCH_SIZE", "256"))
EPOCHS = int(os.getenv("XMODEL_Z_EPOCHS", "8"))
LR = float(os.getenv("XMODEL_Z_LR", "1e-3"))
HUMAN_DIR = os.path.join(ROOT, "datasets", "human")
BOT_ROOT = os.path.join(ROOT, "datasets", "bot_xmodel_60hz")
ATTACK_WINNERS_DIR = os.path.join(BOT_ROOT, "attack_winners")
LOOP_WINNERS_DIR = os.path.join(BOT_ROOT, "loop_winners")
ATTACK_CFG = os.path.join(ATTACK_WINNERS_DIR, "xmodel_v02s_attack_winners_cfg.json")
MODEL_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}.pt")
METADATA_PATH = os.path.join(XMODEL_DIR, f"{VERSION_ID}_metadata.json")

BOT_DIRS_XMODEL = [
    ("bot_xmodel_challenge", os.path.join(BOT_ROOT, "challenge")),
    ("bot_xmodel_tiers", os.path.join(BOT_ROOT, "tiers")),
    ("bot_xmodel_evasion", os.path.join(BOT_ROOT, "evasion")),
    ("bot_xmodel_families", os.path.join(BOT_ROOT, "families")),
    ("bot_xmodel_legacy", os.path.join(BOT_ROOT, "legacy")),
    ("bot_xmodel_attack_winners", ATTACK_WINNERS_DIR),
    ("bot_xmodel_loop_winners", LOOP_WINNERS_DIR),
    ("bot_xmodel_probe_fails", os.path.join(BOT_ROOT, "probe_fails")),
]


def _z_window_episode(events, source_file, target_hz=None):
    if len(events) < MIN_EVENTS_PER_WINDOW:
        return []
    samples: List[WindowSample] = []
    for window in segment_episode_into_windows(events, DEFAULT_ACTIVE_WINDOW_CONFIG):
        if len(window) < 8:
            continue
        samples.append(
            WindowSample(
                feature_dict_to_vector(extract_features(window)),
                source_file,
                [dict(event) for event in window],
            )
        )
    return samples


def _pack_samples(samples: List[WindowSample], group_prefix: str):
    seqs, masks, tabs, groups = [], [], [], []
    for sample in samples:
        seq, mask = events_to_relative_sequence(sample.events or [])
        if float(mask.sum()) < 8:
            continue
        seqs.append(seq)
        masks.append(mask)
        tabs.append(np.asarray(sample.vector, dtype=np.float32))
        groups.append(f"{group_prefix}::{sample.source_file}")
    if not seqs:
        return None
    return np.stack(seqs), np.stack(masks), np.stack(tabs), groups


def load_human_60hz_native() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], List[Dict[str, Any]], Dict[str, int]]:
    seq_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    tab_parts: List[np.ndarray] = []
    groups: List[str] = []
    summaries: List[Dict[str, Any]] = []
    counts = {"os_recordings_60hz_native": 0}

    csvs = sorted(
        p for p in glob.glob(os.path.join(HUMAN_DIR, "*.csv"))
        if "_60hz" in os.path.basename(p).lower()
    )
    print(f"[Data] Native 60 Hz human recordings: {len(csvs)} CSV files")
    for path in csvs:
        samples, summary = tif.extract_active_windows_from_csv(path, target_hz=None)
        packed = _pack_samples(samples, f"human_60hz_native::{os.path.basename(path)}")
        if not packed:
            continue
        s, m, t, g = packed
        seq_parts.append(s)
        mask_parts.append(m)
        tab_parts.append(t)
        groups.extend(g)
        counts["os_recordings_60hz_native"] += len(g)
        summaries.append(summary)
        print(f"[Data] human {os.path.basename(path)}: {len(g):,} windows")

    return seq_parts, mask_parts, tab_parts, groups, summaries, counts


def load_bot_60hz_native() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], List[Dict[str, Any]], Dict[str, int]]:
    seq_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    tab_parts: List[np.ndarray] = []
    groups: List[str] = []
    summaries: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for prefix, bot_dir in BOT_DIRS_XMODEL:
        if not os.path.isdir(bot_dir):
            print(f"[Data] Skipping missing bot dir: {bot_dir}")
            continue
        csvs = sorted(glob.glob(os.path.join(bot_dir, "**", "*.csv"), recursive=True))
        dir_windows = 0
        for path in csvs:
            rel = os.path.relpath(path, bot_dir).replace("\\", "/")
            samples, summary = tif.extract_active_windows_from_csv(
                path, max_windows=100_000, target_hz=None
            )
            packed = _pack_samples(samples, f"{prefix}::{rel}")
            if not packed:
                continue
            s, m, t, g = packed
            seq_parts.append(s)
            mask_parts.append(m)
            tab_parts.append(t)
            groups.extend(g)
            dir_windows += len(g)
            summaries.append(summary)
        counts[prefix] = dir_windows
        print(f"[Data] {prefix}: {dir_windows:,} windows from {len(csvs)} files")

    return seq_parts, mask_parts, tab_parts, groups, summaries, counts


def _metrics(name: str, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else 0.0
    acc = float((tn + tp) / len(y_true))
    print("=" * 72)
    print(name)
    print(f"ROC-AUC {auc:.4f}  Acc {acc:.4f}")
    print(cm)
    print(classification_report(y_true, pred, target_names=["Human", "Bot"], zero_division=0))
    return {
        "name": name,
        "roc_auc": auc,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Train] WARNING: CUDA not available; training on CPU will be slow.")

    print("=" * 72)
    print(f"Train {VERSION_ID} from {PARENT_ID} — ZNet 1D CNN + loop winners")
    print("=" * 72)

    if not os.path.isdir(ATTACK_WINNERS_DIR):
        raise SystemExit(
            f"Missing {ATTACK_WINNERS_DIR}. Run: python export_xmodel_attack_winners.py"
        )

    tif._window_episode = _z_window_episode

    h_seq, h_mask, h_tab, h_groups, human_summaries, human_counts = load_human_60hz_native()
    b_seq, b_mask, b_tab, b_groups, bot_summaries, bot_counts = load_bot_60hz_native()

    if not h_groups or not b_groups:
        raise SystemExit("Need both human and bot windows.")

    seq = np.concatenate(h_seq + b_seq, axis=0)
    mask = np.concatenate(h_mask + b_mask, axis=0)
    tab = np.concatenate(h_tab + b_tab, axis=0)
    groups = np.array(h_groups + b_groups)
    y = np.array([0] * len(h_groups) + [1] * len(b_groups), dtype=np.int64)

    print(f"[Train] Human windows: {len(h_groups):,}  {human_counts}")
    print(f"[Train] Bot windows:   {len(b_groups):,}  {bot_counts}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(tab, y, groups))
    print(f"[Train] Train: {len(train_idx):,}  Test: {len(test_idx):,}")

    ds = TensorDataset(
        torch.from_numpy(seq[train_idx]),
        torch.from_numpy(mask[train_idx]),
        torch.from_numpy(tab[train_idx]),
        torch.from_numpy(y[train_idx]),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    n_pos = float((y[train_idx] == 1).sum())
    n_neg = float((y[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    net = ZNetV01(tab_dim=tab.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    started = time.perf_counter()
    net.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        n = 0
        for seq_b, mask_b, tab_b, y_b in loader:
            seq_b = seq_b.to(device)
            mask_b = mask_b.to(device)
            tab_b = tab_b.to(device)
            y_b = y_b.float().to(device)
            opt.zero_grad(set_to_none=True)
            logits = net(seq_b, mask_b, tab_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y_b)
            n += len(y_b)
        print(f"[Train] epoch {epoch}/{EPOCHS} loss={total / max(n, 1):.4f}", flush=True)
    train_sec = time.perf_counter() - started

    net.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(test_idx), 1024):
            sl = test_idx[i : i + 1024]
            batch_logits = net(
                torch.from_numpy(seq[sl]).to(device),
                torch.from_numpy(mask[sl]).to(device),
                torch.from_numpy(tab[sl]).to(device),
            )
            chunks.append(torch.sigmoid(batch_logits).cpu().numpy())
    scores = np.concatenate(chunks)
    y_test = y[test_idx]
    eval_result = _metrics(xmodel_display_label(VERSION_ID), y_test, scores)

    attack_export = {}
    if os.path.exists(ATTACK_CFG):
        with open(ATTACK_CFG, encoding="utf-8") as f:
            attack_export = json.load(f)

    bot_gen_summary = {}
    bot_summary_path = os.path.join(BOT_ROOT, "xmodel_60hz_bot_summary.json")
    if os.path.exists(bot_summary_path):
        with open(bot_summary_path, encoding="utf-8") as f:
            bot_gen_summary = json.load(f)

    os.makedirs(XMODEL_DIR, exist_ok=True)
    ckpt = {
        "arch": "ZNetV01",
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "seq_len": SEQ_LEN,
        "target_hz": TARGET_HZ,
        "tab_dim": int(tab.shape[1]),
        "feature_names": FEATURE_NAMES,
        "feature_schema": "zmodel_seq",
    }
    torch.save(ckpt, MODEL_PATH)

    is_ea = ".EA." in VERSION_ID.upper()
    label = xmodel_display_label(VERSION_ID)
    version = xmodel_version_token(VERSION_ID)
    metadata = {
        "model_type": "znet_v01",
        "xmodel_id": VERSION_ID,
        "xmodel_label": label,
        "xmodel_version": version,
        "parent_model": PARENT_ID,
        "target_hz": TARGET_HZ,
        "seq_len": SEQ_LEN,
        "feature_schema": "zmodel_seq",
        "trainer": "znet_cnn",
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "balanced": False,
        "human_filter": "60hz_native_only",
        "bot_filter": "native_60hz_plus_loop_winners" if is_ea else "native_60hz_regenerated_plus_attack_winners",
        "dataset_mix": "evasion_loop" if is_ea else "v03_znet",
        "attack_export": attack_export,
        "human_sources": human_counts,
        "bot_data_root": BOT_ROOT,
        "bot_generation_summary": bot_gen_summary,
        "train_seconds": round(train_sec, 3),
        "human_windows_loaded": int(len(h_groups)),
        "bot_windows_loaded": int(len(b_groups)),
        "bot_windows_by_source": bot_counts,
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "roc_auc": eval_result["roc_auc"],
        "accuracy": eval_result["accuracy"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "human_files": human_summaries,
        "bot_files": bot_summaries,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    register_new_version(
        version_id=VERSION_ID,
        label=label,
        version=version,
        description=(
            f"Evasion-arena child of {PARENT_ID}: ZNet V0.3z retrained on native 60 Hz mix "
            "plus dashboard-bot episodes that scored >=65% human confidence."
            if is_ea else
            "V0.3 native 60 Hz dataset trained with Zmodel-style relative dx,dy,dt 1D CNN "
            "fused with 12 kinematic tabular features (ZNetV01)."
        ),
        patch_cycle=parse_ea_cycle(VERSION_ID) if is_ea else 3,
        evasion_rules_filename=None,
        human_filter="60hz_native_only",
        make_latest=is_ea,
        artifact=f"{VERSION_ID}.pt",
        feature_schema="zmodel_seq",
    )
    print(f"[Save] {MODEL_PATH}")
    print(f"[Save] {METADATA_PATH}")
    print(f"TRAIN_COMPLETE {VERSION_ID} device={device}")


if __name__ == "__main__":
    main()
