"""Ymodel version registry — family-trained detectors, separate from Qmodel latest."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(__file__)
YMODEL_DIR = os.path.join(ROOT, "ymodels")
REGISTRY_PATH = os.path.join(ROOT, "ymodel_registry.json")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"product": "Ymodel", "versions": [], "latest": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(doc: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def list_ymodel_versions() -> List[Dict[str, Any]]:
    doc = _load_registry()
    out: List[Dict[str, Any]] = []
    for entry in doc.get("versions", []):
        artifact_path = os.path.join(YMODEL_DIR, entry["artifact"])
        out.append({
            **entry,
            "available": os.path.exists(artifact_path),
            "path": artifact_path,
        })
    return out


def get_ymodel_entry(model_id: str) -> Optional[Dict[str, Any]]:
    normalized = model_id.strip().lower()
    for entry in list_ymodel_versions():
        if entry["id"].lower() == normalized:
            return entry
    return None


def get_ymodel_artifact_path(model_id: str) -> Optional[str]:
    entry = get_ymodel_entry(model_id)
    if not entry:
        return None
    return entry["path"]


def get_ymodel_evasion_rules_path(model_id: str) -> Optional[str]:
    entry = get_ymodel_entry(model_id)
    if not entry or not entry.get("evasion_rules"):
        return None
    return os.path.join(YMODEL_DIR, entry["evasion_rules"])


def catalog_for_api() -> Dict[str, Any]:
    doc = _load_registry()
    return {
        "product": doc.get("product", "Ymodel"),
        "latest": doc.get("latest"),
        "versions": list_ymodel_versions(),
    }


def register_new_version(
    version_id: str,
    label: str,
    version: str,
    description: str,
    patch_cycle: int,
    evasion_rules_filename: Optional[str],
    bot_evasion_in_training: bool = True,
    make_latest: bool = True,
) -> Dict[str, Any]:
    os.makedirs(YMODEL_DIR, exist_ok=True)
    doc = _load_registry()
    entry = {
        "id": version_id,
        "label": label,
        "version": version,
        "status": "latest" if make_latest else "available",
        "description": description,
        "artifact": f"{version_id}.joblib",
        "metadata": f"{version_id}_metadata.json",
        "evasion_rules": evasion_rules_filename,
        "procedural_mimic_heuristic": True,
        "trained_at": date.today().isoformat(),
        "patch_cycle": patch_cycle,
        "bot_evasion_in_training": bot_evasion_in_training,
        "family_sweep_in_training": True,
    }
    existing = [v for v in doc.get("versions", []) if v.get("id") == version_id]
    if existing:
        existing[0].update(entry)
        if make_latest:
            for v in doc.get("versions", []):
                if v.get("id") != version_id and v.get("status") == "latest":
                    v["status"] = "archived"
            existing[0]["status"] = "latest"
            doc["latest"] = version_id
        save_registry(doc)
        return existing[0]
    if make_latest:
        for v in doc.get("versions", []):
            if v.get("status") == "latest":
                v["status"] = "archived"
        entry["status"] = "latest"
        doc["latest"] = version_id
    else:
        entry["status"] = "available"
    doc.setdefault("versions", []).append(entry)
    doc["product"] = "Ymodel"
    save_registry(doc)
    return entry
