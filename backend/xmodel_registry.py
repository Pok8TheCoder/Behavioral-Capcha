"""Xmodel version registry — 60 Hz native human+bot detectors."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

EA_ID_RE = re.compile(r"^(?P<base>.+)\.EA\.(?P<n>\d+)$", re.IGNORECASE)

ROOT = os.path.dirname(__file__)
XMODEL_DIR = os.path.join(ROOT, "xmodels")
REGISTRY_PATH = os.path.join(ROOT, "xmodel_registry.json")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"product": "Xmodel", "versions": [], "latest": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(doc: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def list_xmodel_versions() -> List[Dict[str, Any]]:
    doc = _load_registry()
    out: List[Dict[str, Any]] = []
    for entry in doc.get("versions", []):
        artifact_path = os.path.join(XMODEL_DIR, entry["artifact"])
        out.append({
            **entry,
            "available": os.path.exists(artifact_path),
            "path": artifact_path,
        })
    return out


def get_xmodel_entry(model_id: str) -> Optional[Dict[str, Any]]:
    normalized = model_id.strip().lower()
    for entry in list_xmodel_versions():
        if entry["id"].lower() == normalized:
            return entry
    return None


def get_xmodel_artifact_path(model_id: str) -> Optional[str]:
    entry = get_xmodel_entry(model_id)
    if not entry:
        return None
    return entry["path"]


def get_xmodel_evasion_rules_path(model_id: str) -> Optional[str]:
    entry = get_xmodel_entry(model_id)
    if not entry or not entry.get("evasion_rules"):
        return None
    return os.path.join(XMODEL_DIR, entry["evasion_rules"])


def catalog_for_api() -> Dict[str, Any]:
    doc = _load_registry()
    return {
        "product": doc.get("product", "Xmodel"),
        "latest": doc.get("latest"),
        "versions": list_xmodel_versions(),
    }


def next_ea_xmodel_id(parent_id: str, existing_ids: Optional[List[str]] = None) -> str:
    """Name the next evasion-arena model: `{parent}.EA.01`, or `{base}.EA.{n+1}` if parent is already EA."""
    parent = (parent_id or "").strip()
    if not parent:
        raise ValueError("parent_id is required to name an evasion-arena model")
    if existing_ids is None:
        existing_ids = [entry["id"] for entry in list_xmodel_versions()]
    taken = {item.lower() for item in existing_ids}
    match = EA_ID_RE.match(parent)
    if match:
        base = match.group("base")
        n = int(match.group("n")) + 1
    else:
        base = parent
        n = 1
    while True:
        candidate = f"{base}.EA.{n:02d}"
        if candidate.lower() not in taken:
            return candidate
        n += 1


def parse_ea_cycle(version_id: str) -> int:
    match = EA_ID_RE.match((version_id or "").strip())
    return int(match.group("n")) if match else 1


def xmodel_display_label(version_id: str) -> str:
    rest = (version_id or "").strip()
    if rest.lower().startswith("xmodel_"):
        rest = rest[7:]
    if rest[:1] in ("v", "V"):
        rest = rest[1:]
    return f"Xmodel V{rest}" if rest else version_id


def xmodel_version_token(version_id: str) -> str:
    rest = (version_id or "").strip()
    if rest.lower().startswith("xmodel_"):
        rest = rest[7:]
    if rest[:1] in ("v", "V"):
        rest = rest[1:]
    return rest or version_id


def register_new_version(
    version_id: str,
    label: str,
    version: str,
    description: str,
    patch_cycle: int = 1,
    evasion_rules_filename: Optional[str] = None,
    human_filter: str = "60hz_filename_only",
    make_latest: bool = True,
    artifact: Optional[str] = None,
    feature_schema: str = "native12",
) -> Dict[str, Any]:
    os.makedirs(XMODEL_DIR, exist_ok=True)
    doc = _load_registry()
    artifact_name = artifact or (
        f"{version_id}.pt" if feature_schema == "zmodel_seq" else f"{version_id}.joblib"
    )
    entry = {
        "id": version_id,
        "label": label,
        "version": version,
        "status": "latest" if make_latest else "available",
        "description": description,
        "artifact": artifact_name,
        "metadata": f"{version_id}_metadata.json",
        "evasion_rules": evasion_rules_filename,
        "procedural_mimic_heuristic": False,
        "trained_at": date.today().isoformat(),
        "patch_cycle": patch_cycle,
        "target_hz": 60,
        "human_filter": human_filter,
        "feature_schema": feature_schema,
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
    doc["product"] = "Xmodel"
    save_registry(doc)
    return entry
