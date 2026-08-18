"""Registry for user-imported joblib detectors (not Qmodel/Ymodel latest)."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(__file__)
IMPORTED_DIR = os.path.join(ROOT, "imported")
REGISTRY_PATH = os.path.join(ROOT, "imported_registry.json")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"product": "Imported", "versions": []}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(doc: Dict[str, Any]) -> None:
    os.makedirs(ROOT, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def list_imported_versions() -> List[Dict[str, Any]]:
    doc = _load_registry()
    out: List[Dict[str, Any]] = []
    for entry in doc.get("versions", []):
        artifact_path = os.path.join(IMPORTED_DIR, entry["artifact"])
        out.append({
            **entry,
            "available": os.path.exists(artifact_path),
            "path": artifact_path,
        })
    return out


def get_imported_entry(model_id: str) -> Optional[Dict[str, Any]]:
    normalized = model_id.strip().lower()
    for entry in list_imported_versions():
        if entry["id"].lower() == normalized:
            return entry
    return None


def register_imported(
    version_id: str,
    label: str,
    original_filename: str,
    n_features: int,
) -> Dict[str, Any]:
    os.makedirs(IMPORTED_DIR, exist_ok=True)
    doc = _load_registry()
    entry = {
        "id": version_id,
        "label": label,
        "version": "imported",
        "status": "imported",
        "description": f"Imported joblib from {original_filename}",
        "artifact": f"{version_id}.joblib",
        "original_filename": original_filename,
        "n_features": n_features,
        "trained_at": date.today().isoformat(),
        "family": "imported",
    }
    existing = [v for v in doc.get("versions", []) if v.get("id") == version_id]
    if existing:
        existing[0].update(entry)
        save_registry(doc)
        return existing[0]
    doc.setdefault("versions", []).append(entry)
    doc["product"] = "Imported"
    save_registry(doc)
    return entry
