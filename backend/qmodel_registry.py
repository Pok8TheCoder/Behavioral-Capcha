"""Qmodel version registry — paths, labels, and patch metadata."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(__file__)
QMODEL_DIR = os.path.join(ROOT, "qmodels")
REGISTRY_PATH = os.path.join(ROOT, "qmodel_registry.json")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"versions": [], "latest": None, "legacy_models": []}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(doc: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def list_qmodel_versions() -> List[Dict[str, Any]]:
    doc = _load_registry()
    versions = doc.get("versions", [])
    out: List[Dict[str, Any]] = []
    for entry in versions:
        artifact_path = os.path.join(QMODEL_DIR, entry["artifact"])
        out.append({
            **entry,
            "available": os.path.exists(artifact_path),
            "path": artifact_path,
        })
    return out


def get_qmodel_entry(model_id: str) -> Optional[Dict[str, Any]]:
    normalized = model_id.strip().lower()
    for entry in list_qmodel_versions():
        if entry["id"].lower() == normalized:
            return entry
    return None


def get_qmodel_artifact_path(model_id: str) -> Optional[str]:
    entry = get_qmodel_entry(model_id)
    if not entry:
        return None
    return entry["path"]


def get_evasion_rules_path(model_id: str) -> Optional[str]:
    entry = get_qmodel_entry(model_id)
    if not entry or not entry.get("evasion_rules"):
        return None
    return os.path.join(QMODEL_DIR, entry["evasion_rules"])


def procedural_mimic_enabled(model_id: str) -> bool:
    entry = get_qmodel_entry(model_id)
    if not entry:
        return True
    return bool(entry.get("procedural_mimic_heuristic", True))


def catalog_for_api() -> Dict[str, Any]:
    doc = _load_registry()
    versions = list_qmodel_versions()
    return {
        "product": doc.get("product", "Qmodel"),
        "latest": doc.get("latest"),
        "stable": doc.get("stable"),
        "versions": versions,
        "legacy_models": doc.get("legacy_models", []),
    }


def next_patch_version() -> str:
    """Next id as qmodel_v0.10, v0.11, ... (integer minor, not float 0.9+0.1)."""
    doc = _load_registry()
    versions = doc.get("versions", [])
    minors: List[int] = []
    for entry in versions:
        ver = str(entry.get("version", ""))
        if ver in {"1.0", "1"}:
            continue
        for src in (ver, str(entry.get("id", ""))):
            m = re.search(r"0\.(\d+)", src)
            if m:
                minors.append(int(m.group(1)))
    nxt = max(minors) + 1 if minors else 2
    return f"qmodel_v0.{nxt}"


def register_new_version(
    version_id: str,
    label: str,
    version: str,
    description: str,
    patch_cycle: int,
    evasion_rules_filename: Optional[str],
    bot_evasion_in_training: bool = True,
) -> Dict[str, Any]:
    doc = _load_registry()
    entry = {
        "id": version_id,
        "label": label,
        "version": version,
        "status": "latest",
        "description": description,
        "artifact": f"{version_id}.joblib",
        "metadata": f"{version_id}_metadata.json",
        "evasion_rules": evasion_rules_filename,
        "procedural_mimic_heuristic": True,
        "trained_at": __import__("datetime").date.today().isoformat(),
        "patch_cycle": patch_cycle,
        "bot_evasion_in_training": bot_evasion_in_training,
    }
    for v in doc.get("versions", []):
        if v.get("status") == "latest":
            v["status"] = "archived"
    doc.setdefault("versions", []).append(entry)
    doc["latest"] = version_id
    save_registry(doc)
    return entry
