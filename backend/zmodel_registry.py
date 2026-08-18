"""Zmodel version registry — sequence detectors, separate from Qmodel/Ymodel latest."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(__file__)
ZMODEL_DIR = os.path.join(ROOT, "zmodels")
REGISTRY_PATH = os.path.join(ROOT, "zmodel_registry.json")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"product": "Zmodel", "versions": [], "latest": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(doc: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def list_zmodel_versions() -> List[Dict[str, Any]]:
    doc = _load_registry()
    out: List[Dict[str, Any]] = []
    for entry in doc.get("versions", []):
        artifact_path = os.path.join(ZMODEL_DIR, entry["artifact"])
        out.append({
            **entry,
            "available": os.path.exists(artifact_path),
            "path": artifact_path,
        })
    return out


def get_zmodel_entry(model_id: str) -> Optional[Dict[str, Any]]:
    normalized = model_id.strip().lower()
    for entry in list_zmodel_versions():
        if entry["id"].lower() == normalized:
            return entry
    return None


def get_zmodel_artifact_path(model_id: str) -> Optional[str]:
    entry = get_zmodel_entry(model_id)
    if not entry:
        return None
    return entry["path"]


def catalog_for_api() -> Dict[str, Any]:
    doc = _load_registry()
    return {
        "product": doc.get("product", "Zmodel"),
        "latest": doc.get("latest"),
        "versions": list_zmodel_versions(),
    }


def register_new_version(
    version_id: str,
    label: str,
    version: str,
    description: str,
    patch_cycle: int,
    make_latest: bool = True,
) -> Dict[str, Any]:
    os.makedirs(ZMODEL_DIR, exist_ok=True)
    doc = _load_registry()
    entry = {
        "id": version_id,
        "label": label,
        "version": version,
        "status": "latest" if make_latest else "available",
        "description": description,
        "artifact": f"{version_id}.pt",
        "metadata": f"{version_id}_metadata.json",
        "procedural_mimic_heuristic": True,
        "trained_at": date.today().isoformat(),
        "patch_cycle": patch_cycle,
        "feature_schema": "zmodel_seq",
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
    doc["product"] = "Zmodel"
    save_registry(doc)
    return entry
