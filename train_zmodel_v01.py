"""Train Zmodel V0.1 — 60 Hz relative (Δx, Δy, Δt) 1D CNN fused with 12 kinematic features."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.active_window import DEFAULT_ACTIVE_WINDOW_CONFIG, segment_episode_into_windows
from backend.feature_extractor import extract_features, feature_dict_to_vector
from backend.hz60 import downsample_events_to_hz
from backend.zmodel_net import ZNetV01
from backend.zmodel_registry import ZMODEL_DIR, register_new_version
from backend.zmodel_seq import SEQ_LEN, events_to_relative_sequence
from train_isolation_forest import FEATURE_NAMES, HUMAN_DIR, MIN_EVENTS_PER_WINDOW, WindowSample
from train_qmodel_v081g import CHALLENGE_DIR
from train_random_forest_balanced import BOT_DIRS_ALL
import train_isolation_forest as tif

VERSION_ID = "zmodel_v0.1"
TARGET_HZ = 60.0
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
FAMILY_DIR = os.path.join(ROOT, "datasets", "bot_families")
Q_BENCH_ID = "qmodel_v0.16"
Y_BENCH_ID = "ymodel_v0.3.2M"


def _z_window_episode(events, source_file, target_hz=None):
    if target_hz:
        events = downsample_events_to_hz(events, float(target_hz))
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


def load_all_z_arrays():
    tif._window_episode = _z_window_episode
    seqs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    tabs: List[np.ndarray] = []
    groups: List[str] = []
    labels: List[int] = []

    os_csvs = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.csv")))
    print(f"[Z] OS recordings: {len(os_csvs)}")
    for path in os_csvs:
        samples, _ = tif.extract_active_windows_from_csv(path, target_hz=TARGET_HZ)
        packed = _pack_samples(samples, f"human_os::{os.path.basename(path)}")
        if not packed:
            continue
        s, m, t, g = packed
        seqs.append(s)
        masks.append(m)
        tabs.append(t)
        groups.extend(g)
        labels.extend([0] * len(g))
        print(f"[Z] human_os {os.path.basename(path)}: {len(g):,} windows")

    dash = sorted(glob.glob(os.path.join(CHALLENGE_DIR, "**", "*.csv"), recursive=True))
    dash = [p for p in dash if "_archive_no_spec" not in os.path.normpath(p).split(os.sep)]
    print(f"[Z] Dashboard CSVs: {len(dash)}")
    for path in dash:
        rel = os.path.relpath(path, CHALLENGE_DIR).replace("\\", "/")
        samples, _ = tif.extract_active_windows_from_csv(path, target_hz=TARGET_HZ)
        packed = _pack_samples(samples, f"human_dashboard::{rel}")
        if not packed:
            continue
        s, m, t, g = packed
        seqs.append(s)
        masks.append(m)
        tabs.append(t)
        groups.extend(g)
        labels.extend([0] * len(g))

    for prefix, bot_dir in BOT_DIRS_ALL + [("bot_families", FAMILY_DIR)]:
        if not os.path.isdir(bot_dir):
            continue
        csvs = sorted(glob.glob(os.path.join(bot_dir, "**", "*.csv"), recursive=True))
        print(f"[Z] {prefix}: {len(csvs)} files")
        for path in csvs:
            rel = os.path.relpath(path, bot_dir).replace("\\", "/")
            samples, _ = tif.extract_active_windows_from_csv(path, max_windows=100_000, target_hz=TARGET_HZ)
            packed = _pack_samples(samples, f"{prefix}::{rel}")
            if not packed:
                continue
            s, m, t, g = packed
            seqs.append(s)
            masks.append(m)
            tabs.append(t)
            groups.extend(g)
            labels.extend([1] * len(g))
            print(f"[Z] {prefix} {rel}: {len(g):,} windows")

    seq = np.concatenate(seqs, axis=0)
    mask = np.concatenate(masks, axis=0)
    tab = np.concatenate(tabs, axis=0)
    y = np.array(labels, dtype=np.int64)
    groups_arr = np.array(groups)
    print(f"[Z] Packed human={(y == 0).sum():,} bot={(y == 1).sum():,}")
    return seq, mask, tab, y, groups_arr


def _metrics(name: str, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else 0.0
    acc = float((tn + tp) / len(y_true))
    print("=" * 72)
    print(name)
    print(f"ROC-AUC {auc:.4f}  Acc {acc:.4f}")
    print("Confusion [[TN, FP], [FN, TP]]")
    print(cm)
    print(classification_report(y_true, pred, target_names=["Human", "Bot"], zero_division=0))
    print(
        f"ML-human humans={1.0 - float(np.mean(scores[y_true == 0])):.3f} "
        f"bots={1.0 - float(np.mean(scores[y_true == 1])):.3f}"
    )
    return {
        "name": name,
        "roc_auc": auc,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "ml_human_on_humans": float(1.0 - np.mean(scores[y_true == 0])),
        "ml_human_on_bots": float(1.0 - np.mean(scores[y_true == 1])),
    }


def _score_xgb(artifact: str, tab: np.ndarray) -> np.ndarray:
    pkg = joblib.load(artifact)
    model = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
    return np.asarray(model.predict_proba(tab)[:, 1], dtype=np.float64)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Zmodel V0.1 requires CUDA.")
    device = torch.device("cuda")
    print("=" * 72)
    print("Train Zmodel V0.1 — relative dx,dy,dt 1D CNN + 12 features, 60 Hz")
    print("=" * 72)

    seq, mask, tab, y, groups = load_all_z_arrays()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(tab, y, groups))
    print(f"[Z] Train {len(train_idx):,}  Test {len(test_idx):,}")

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
        print(f"[Z] epoch {epoch}/{EPOCHS} loss={total / max(n, 1):.4f}", flush=True)
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
    z_scores = np.concatenate(chunks)

    y_test = y[test_idx]
    tab_test = tab[test_idx]
    results = [_metrics("Zmodel V0.1", y_test, z_scores)]

    q_path = os.path.join(ROOT, "backend", "qmodels", f"{Q_BENCH_ID}.joblib")
    y_path = os.path.join(ROOT, "backend", "ymodels", f"{Y_BENCH_ID}.joblib")
    if os.path.exists(q_path):
        results.append(_metrics("Qmodel V0.16 (same windows)", y_test, _score_xgb(q_path, tab_test)))
    if os.path.exists(y_path):
        results.append(_metrics("Ymodel V0.3.2M (same windows)", y_test, _score_xgb(y_path, tab_test)))

    os.makedirs(ZMODEL_DIR, exist_ok=True)
    ckpt = {
        "arch": "ZNetV01",
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "seq_len": SEQ_LEN,
        "target_hz": TARGET_HZ,
        "tab_dim": int(tab.shape[1]),
        "feature_names": FEATURE_NAMES,
        "feature_schema": "zmodel_seq",
    }
    pt_path = os.path.join(ZMODEL_DIR, f"{VERSION_ID}.pt")
    torch.save(ckpt, pt_path)

    metadata = {
        "model_type": "znet_v01",
        "zmodel_id": VERSION_ID,
        "zmodel_label": "Zmodel V0.1",
        "target_hz": TARGET_HZ,
        "seq_len": SEQ_LEN,
        "feature_schema": "zmodel_seq",
        "train_seconds": round(train_sec, 3),
        "device": "cuda",
        "epochs": EPOCHS,
        "human_windows": int((y == 0).sum()),
        "bot_windows": int((y == 1).sum()),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "bench": results,
    }
    meta_path = os.path.join(ZMODEL_DIR, f"{VERSION_ID}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    register_new_version(
        version_id=VERSION_ID,
        label="Zmodel V0.1",
        version="0.1",
        description=(
            "60 Hz relative dx,dy,dt 1D CNN fused with the 12 kinematic features, "
            "same dynamic windows as Q/Y."
        ),
        patch_cycle=1,
        make_latest=True,
    )
    print(f"[Save] {pt_path}")
    print("TRAIN_COMPLETE", VERSION_ID)


if __name__ == "__main__":
    main()
