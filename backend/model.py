import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.feature_extractor import extract_features, feature_dict_to_vector
from backend.alex_features import alex_dict_to_vector, extract_alex_features
from backend.evasion_heuristics import load_rules, score_dynamic_rules
from backend.qmodel_registry import (
    catalog_for_api,
    get_evasion_rules_path,
    get_qmodel_artifact_path,
    get_qmodel_entry,
    list_qmodel_versions,
    procedural_mimic_enabled as qmodel_procedural_mimic,
)
from backend.imported_registry import get_imported_entry, list_imported_versions
from backend.ymodel_registry import (
    get_ymodel_entry,
    get_ymodel_evasion_rules_path,
    list_ymodel_versions,
)
from backend.zmodel_registry import get_zmodel_entry, list_zmodel_versions

IFOREST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_iforest_model.joblib")
RF_CHALLENGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_challenge_model.joblib")
RF_HUMANLIKE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_humanlike_model.joblib")
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_model.joblib")

LEGACY_MODEL_REGISTRY = {
    "random_forest_challenge": RF_CHALLENGE_MODEL_PATH,
    "random_forest_humanlike": RF_HUMANLIKE_MODEL_PATH,
    "isolation_forest": IFOREST_MODEL_PATH,
    "random_forest": RF_MODEL_PATH,
}


def _build_model_registry() -> dict:
    registry = dict(LEGACY_MODEL_REGISTRY)
    for entry in list_qmodel_versions():
        registry[entry["id"]] = entry["path"]
    for entry in list_ymodel_versions():
        registry[entry["id"]] = entry["path"]
    for entry in list_imported_versions():
        registry[entry["id"]] = entry["path"]
    for entry in list_zmodel_versions():
        registry[entry["id"]] = entry["path"]
    return registry


def _canonical_model_id(model_name: str, registry: dict) -> str | None:
    """Match registry keys case-insensitively (e.g. qmodel_v0.8.1G)."""
    requested = model_name.strip()
    if not requested:
        return None
    if requested in registry:
        return requested
    lowered = requested.lower()
    for key in registry:
        if key.lower() == lowered:
            return key
    return None


MODEL_REGISTRY = _build_model_registry()

class BehavioralClassifier:
    def __init__(self):
        self.model = None
        self.model_type = "unloaded"
        self.display_name = "unloaded"
        self.feature_names = []
        self.metadata = {}
        self.is_trained = False
        self.evasion_rules_path = None
        self.use_procedural_mimic_heuristic = True
        self.feature_schema = "native12"
        self.scaler = None
        self.use_scaler = False
        requested_model = os.getenv("BEHAVIORAL_MODEL", "").strip()
        if not requested_model:
            catalog = catalog_for_api()
            requested_model = catalog.get("latest") or "random_forest_challenge"
        if not self.load_model(requested_model):
            print(f"[Model] Could not load requested model '{requested_model}'; trying fallbacks.")
            for model_name in self._fallback_order():
                if self.load_model(model_name):
                    return
            self.train_synthetic_model()

    def _fallback_order(self) -> tuple:
        catalog = catalog_for_api()
        latest = catalog.get("latest")
        order = []
        if latest:
            order.append(latest)
        for entry in reversed(catalog.get("versions", [])):
            if entry["id"] != latest:
                order.append(entry["id"])
        order.extend(["random_forest_challenge", "random_forest_humanlike", "isolation_forest", "random_forest"])
        return tuple(order)

    def available_models(self) -> dict:
        """Selectable artifacts: Qmodels first, then legacy."""
        out: dict = {}
        for entry in list_qmodel_versions():
            out[entry["id"]] = {
                "available": entry.get("available", False),
                "path": entry.get("path"),
                "label": entry.get("label"),
                "version": entry.get("version"),
                "status": entry.get("status"),
                "family": "qmodel",
            }
        for entry in list_ymodel_versions():
            out[entry["id"]] = {
                "available": entry.get("available", False),
                "path": entry.get("path"),
                "label": entry.get("label"),
                "version": entry.get("version"),
                "status": entry.get("status"),
                "family": "ymodel",
            }
        for entry in list_imported_versions():
            out[entry["id"]] = {
                "available": entry.get("available", False),
                "path": entry.get("path"),
                "label": entry.get("label"),
                "version": entry.get("version"),
                "status": entry.get("status"),
                "family": "imported",
            }
        for entry in list_zmodel_versions():
            out[entry["id"]] = {
                "available": entry.get("available", False),
                "path": entry.get("path"),
                "label": entry.get("label"),
                "version": entry.get("version"),
                "status": entry.get("status"),
                "family": "zmodel",
            }
        for model_name, model_path in LEGACY_MODEL_REGISTRY.items():
            out[model_name] = {
                "available": os.path.exists(model_path),
                "path": model_path,
                "label": model_name.replace("_", " "),
                "family": "legacy",
            }
        return out

    def qmodel_catalog(self) -> dict:
        return catalog_for_api()

    def load_model(self, model_name: str) -> bool:
        """Loads one registered artifact (Qmodel or legacy)."""
        requested = model_name.strip()
        if requested.lower() == "random_forest_challenge":
            catalog = catalog_for_api()
            latest = catalog.get("latest")
            if latest and get_qmodel_entry(latest):
                requested = latest

        global MODEL_REGISTRY
        MODEL_REGISTRY = _build_model_registry()
        canonical_name = _canonical_model_id(requested, MODEL_REGISTRY)
        model_path = MODEL_REGISTRY.get(canonical_name) if canonical_name else None
        if not model_path:
            print(f"[Model] Unknown model requested: {model_name}")
            return False
        if not os.path.exists(model_path):
            print(f"[Model] Model artifact is missing: {model_path}")
            return False

        self.feature_schema = "native12"
        self.scaler = None
        self.use_scaler = False
        self._z_events = None
        z_entry = get_zmodel_entry(canonical_name) if canonical_name else None
        if z_entry:
            return self._load_zmodel(canonical_name, model_path, z_entry)
        try:
            package = joblib.load(model_path)
            self.model = package["model"] if isinstance(package, dict) and "model" in package else package
            self.feature_names = package.get("feature_names", []) if isinstance(package, dict) else []
            self.metadata = package.get("metadata", {}) if isinstance(package, dict) else {}
            self.scaler = package.get("scaler") if isinstance(package, dict) else None
            self.feature_schema = (
                (package.get("feature_schema") if isinstance(package, dict) else None)
                or self.metadata.get("feature_schema")
                or "native12"
            )
            self.use_scaler = bool(
                (package.get("use_scaler") if isinstance(package, dict) else False)
                or self.metadata.get("use_scaler")
            )
            self.model_type = canonical_name
            entry = (
                get_qmodel_entry(canonical_name)
                or get_ymodel_entry(canonical_name)
                or get_imported_entry(canonical_name)
                or get_zmodel_entry(canonical_name)
            )
            if entry:
                self.display_name = entry.get("label", canonical_name)
                if get_qmodel_entry(canonical_name):
                    self.evasion_rules_path = get_evasion_rules_path(canonical_name)
                    self.use_procedural_mimic_heuristic = qmodel_procedural_mimic(canonical_name)
                elif get_imported_entry(canonical_name):
                    self.evasion_rules_path = None
                    self.use_procedural_mimic_heuristic = self.feature_schema != "alex11"
                else:
                    self.evasion_rules_path = get_ymodel_evasion_rules_path(canonical_name)
                    self.use_procedural_mimic_heuristic = bool(entry.get("procedural_mimic_heuristic", True))
            else:
                self.display_name = self.metadata.get("qmodel_label") or canonical_name.replace("_", " ")
                self.evasion_rules_path = None
                self.use_procedural_mimic_heuristic = True
            self.is_trained = True
            print(f"[Model] Loaded {self.display_name} ({canonical_name}) from {model_path}")
            return True
        except Exception as e:
            print(f"[Model] Error loading {canonical_name}: {e}")
            return False

    def _load_zmodel(self, canonical_name: str, model_path: str, entry: dict) -> bool:
        try:
            import torch
            from backend.zmodel_net import ZNetV01
            from backend.zmodel_seq import SEQ_LEN
        except ImportError as exc:
            print(f"[Model] Zmodel requires PyTorch: {exc}")
            return False
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            tab_dim = int(ckpt.get("tab_dim", 12))
            net = ZNetV01(tab_dim=tab_dim)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(device)
            self.model = net
            self._z_device = device
            self.feature_schema = ckpt.get("feature_schema") or "zmodel_seq"
            self.feature_names = list(ckpt.get("feature_names") or [])
            meta_name = entry.get("metadata") or f"{canonical_name}_metadata.json"
            meta_path = os.path.join(os.path.dirname(model_path), meta_name)
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            metadata["target_hz"] = ckpt.get("target_hz") or metadata.get("target_hz") or 60.0
            metadata["seq_len"] = ckpt.get("seq_len") or SEQ_LEN
            self.metadata = metadata
            self.model_type = canonical_name
            self.display_name = entry.get("label", canonical_name)
            self.evasion_rules_path = None
            self.use_procedural_mimic_heuristic = bool(entry.get("procedural_mimic_heuristic", True))
            self.is_trained = True
            print(f"[Model] Loaded {self.display_name} ({canonical_name}) from {model_path} on {device}")
            return True
        except Exception as exc:
            print(f"[Model] Error loading Zmodel {canonical_name}: {exc}")
            return False

    def features_from_events(self, events: list) -> dict:
        if self.feature_schema == "alex11":
            return extract_alex_features(events)
        return extract_features(events)

    def _ml_vector(self, features: dict):
        if self.feature_schema == "alex11":
            vec = np.array([alex_dict_to_vector(features)], dtype=float)
            if self.use_scaler and self.scaler is not None:
                vec = self.scaler.transform(vec)
            return vec
        return [feature_dict_to_vector(features)]

    def _ml_bot_probability(self, features: dict) -> float:
        if not (self.is_trained and self.model):
            return 0.5
        if self.feature_schema == "zmodel_seq":
            return self._z_bot_probability(features)
        vec = self._ml_vector(features)
        if hasattr(self.model, "predict_proba"):
            try:
                return float(self.model.predict_proba(vec)[0][1])
            except Exception:
                pass
        if hasattr(self.model, "decision_function"):
            score = float(np.asarray(self.model.decision_function(vec)).ravel()[0])
            return float(1.0 / (1.0 + np.exp(-score)))
        return float(self.model.predict(vec)[0])

    def _z_bot_probability(self, features: dict) -> float:
        import torch
        from backend.zmodel_seq import events_to_relative_sequence

        events = getattr(self, "_z_events", None) or []
        seq, mask = events_to_relative_sequence(events)
        tab = np.asarray(feature_dict_to_vector(features), dtype=np.float32)
        device = getattr(self, "_z_device", torch.device("cpu"))
        self.model.eval()
        with torch.no_grad():
            logit = self.model(
                torch.from_numpy(seq[None, ...]).to(device),
                torch.from_numpy(mask[None, ...]).to(device),
                torch.from_numpy(tab[None, ...]).to(device),
            )
            return float(torch.sigmoid(logit).detach().cpu().item())

    def evaluate_window(self, events: list) -> dict:
        self._z_events = events
        try:
            return self.evaluate(self.features_from_events(events))
        finally:
            self._z_events = None

    def import_joblib_bytes(self, raw: bytes, filename: str, activate: bool = True) -> dict:
        """Save a third-party joblib package into the imported-model catalog."""
        import io
        import re

        from backend.imported_registry import IMPORTED_DIR, register_imported
        from train_isolation_forest import FEATURE_NAMES

        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", os.path.splitext(os.path.basename(filename))[0])
        stem = stem.strip("._") or "model"
        version_id = f"imported_{stem}"[:80]
        try:
            package = joblib.load(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(f"Could not unpickle joblib: {exc}") from exc

        if isinstance(package, dict) and "model" in package:
            estimator = package["model"]
            feature_names = list(package.get("feature_names") or FEATURE_NAMES)
            metadata = dict(package.get("metadata") or {})
        else:
            estimator = package
            feature_names = list(FEATURE_NAMES)
            metadata = {"imported_from": filename}

        if not any(hasattr(estimator, name) for name in ("predict_proba", "predict", "decision_function")):
            raise ValueError("Joblib payload has no predict / predict_proba / decision_function.")

        metadata["imported_from"] = os.path.basename(filename)
        label = (
            metadata.get("ymodel_label")
            or metadata.get("qmodel_label")
            or f"Imported {stem}"
        )
        metadata["ymodel_label"] = label
        os.makedirs(IMPORTED_DIR, exist_ok=True)
        dest = os.path.join(IMPORTED_DIR, f"{version_id}.joblib")
        joblib.dump({"model": estimator, "feature_names": feature_names, "metadata": metadata}, dest)
        register_imported(
            version_id=version_id,
            label=label,
            original_filename=os.path.basename(filename),
            n_features=len(feature_names),
        )
        if activate and not self.load_model(version_id):
            raise ValueError(f"Imported {version_id} but failed to activate it.")
        return {
            "model_id": version_id,
            "label": label,
            "n_features": len(feature_names),
            "activated": bool(activate),
        }

    def _load_or_train_model(self):
        """Backward-compatible artifact loading entry point."""
        if os.path.exists(IFOREST_MODEL_PATH):
            try:
                package = joblib.load(IFOREST_MODEL_PATH)
                if isinstance(package, dict) and "model" in package:
                    self.model = package["model"]
                    self.feature_names = package.get("feature_names", [])
                    self.metadata = package.get("metadata", {})
                else:
                    self.model = package
                self.model_type = "isolation_forest"
                self.is_trained = True
                print(f"[Model] Loaded Isolation Forest model from {IFOREST_MODEL_PATH}")
                return
            except Exception as e:
                print(f"[Model] Error loading Isolation Forest model: {e}")

        if os.path.exists(RF_MODEL_PATH):
            try:
                self.model = joblib.load(RF_MODEL_PATH)
                self.model_type = "random_forest"
                self.is_trained = True
                return
            except Exception as e:
                print(f"[Model] Error loading Random Forest fallback model: {e}")

        # Train baseline model on synthetic dataset
        self.train_synthetic_model()

    def train_synthetic_model(self, n_samples: int = 1200):
        """Generates synthetic human & bot biometric feature vectors and trains Random Forest."""
        X, y = [], []

        np.random.seed(42)

        # 1. Generate Synthetic Humans (Class 0)
        for _ in range(n_samples // 2):
            straightness = np.random.uniform(0.65, 0.93)
            vel_mean = np.random.uniform(0.3, 1.8)
            vel_std = np.random.uniform(0.2, 1.2) * vel_mean
            vel_max = vel_mean + np.random.uniform(1.5, 4.0) * vel_std
            acc_mean = np.random.uniform(0.001, 0.02)
            acc_std = np.random.uniform(0.001, 0.03)
            jerk_mean = np.random.uniform(0.00001, 0.0005)
            angle_mean = np.random.uniform(0.15, 0.6)
            angle_std = np.random.uniform(0.2, 0.8)
            jitter_ratio = np.random.uniform(0.02, 0.25)
            idle_ratio = np.random.uniform(0.05, 0.35)
            click_dwell = np.random.uniform(50.0, 180.0)

            X.append([
                straightness, vel_mean, vel_std, vel_max, acc_mean,
                acc_std, jerk_mean, angle_mean, angle_std, jitter_ratio,
                idle_ratio, click_dwell
            ])
            y.append(0)  # Human

        # 2. Generate Synthetic Bots (Class 1)
        for _ in range(n_samples // 2):
            bot_type = np.random.choice(["linear", "bezier", "teleport", "jitter_bot"])
            if bot_type == "linear":
                straightness = np.random.uniform(0.985, 1.0)
                vel_mean = np.random.uniform(0.5, 3.0)
                vel_std = np.random.uniform(0.0, 0.03)  # Almost constant velocity
                vel_max = vel_mean + 0.05
                acc_mean = np.random.uniform(0.0, 0.0001)
                acc_std = 0.0
                jerk_mean = 0.0
                angle_mean = np.random.uniform(0.0, 0.02)
                angle_std = 0.0
                jitter_ratio = 0.0
                idle_ratio = np.random.choice([0.0, 0.5])
                click_dwell = np.random.uniform(0.5, 8.0) # Robot instant click

            elif bot_type == "bezier":
                straightness = np.random.uniform(0.92, 0.97)
                vel_mean = np.random.uniform(0.8, 2.5)
                vel_std = np.random.uniform(0.01, 0.08)  # Mathematical curve uniform velocity
                vel_max = vel_mean + 0.1
                acc_mean = np.random.uniform(0.0001, 0.001)
                acc_std = 0.0001
                jerk_mean = 0.000001
                angle_mean = np.random.uniform(0.01, 0.05)
                angle_std = 0.01
                jitter_ratio = 0.0  # Zero micro-tremors
                idle_ratio = 0.0
                click_dwell = np.random.uniform(1.0, 15.0)

            elif bot_type == "teleport":
                straightness = 1.0
                vel_mean = np.random.uniform(15.0, 100.0) # Huge velocity jump
                vel_std = np.random.uniform(10.0, 50.0)
                vel_max = vel_mean * 2
                acc_mean = np.random.uniform(0.1, 1.0)
                acc_std = 0.5
                jerk_mean = 0.01
                angle_mean = 0.0
                angle_std = 0.0
                jitter_ratio = 0.0
                idle_ratio = 0.0
                click_dwell = np.random.uniform(0.0, 2.0)

            else:  # jitter_bot (artificial white noise added to straight line)
                straightness = np.random.uniform(0.95, 0.99)
                vel_mean = np.random.uniform(0.5, 2.0)
                vel_std = np.random.uniform(0.05, 0.2)
                vel_max = vel_mean + 0.3
                acc_mean = np.random.uniform(0.002, 0.01)
                acc_std = 0.005
                jerk_mean = np.random.uniform(0.001, 0.01) # High artificial jerk
                angle_mean = np.random.uniform(0.8, 1.5)  # Chaotic unnatural angle shifts
                angle_std = np.random.uniform(1.0, 2.0)
                jitter_ratio = np.random.uniform(0.4, 0.9) # Excessive artificial jitter
                idle_ratio = 0.0
                click_dwell = np.random.uniform(2.0, 20.0)

            X.append([
                straightness, vel_mean, vel_std, vel_max, acc_mean,
                acc_std, jerk_mean, angle_mean, angle_std, jitter_ratio,
                idle_ratio, click_dwell
            ])
            y.append(1)  # Bot

        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X, y)
        self.model = clf
        self.model_type = "random_forest_synthetic_fallback"
        self.is_trained = True

        try:
            joblib.dump(clf, RF_MODEL_PATH)
        except Exception as e:
            print(f"[Model] Warning: Failed saving trained model: {e}")

    def _evaluate_isolation_forest(self, features: dict, flags: list, heuristic_bot_score: float) -> dict:
        vec = [feature_dict_to_vector(features)]
        prediction = int(self.model.predict(vec)[0])
        decision_score = float(self.model.decision_function(vec)[0])
        anomaly_score = -decision_score

        if prediction == -1:
            ml_bot_prob = min(1.0, 0.55 + max(0.0, anomaly_score) * 6.0)
        else:
            ml_bot_prob = max(0.0, 0.45 - max(0.0, decision_score) * 6.0)

        combined_score = min(1.0, max(min(1.0, heuristic_bot_score), ml_bot_prob))
        is_bot = combined_score >= 0.5
        decision = "BLOCKED" if is_bot else "PASSED"

        if prediction == -1:
            flags.append(f"Isolation Forest marked telemetry as anomalous (score: {decision_score:.4f})")

        return {
            "decision": decision,
            "is_bot": is_bot,
            "risk_score": round(combined_score, 4),
            "bot_probability": round(ml_bot_prob, 4),
            "anomaly_score": round(anomaly_score, 6),
            "isolation_forest_score": round(decision_score, 6),
            "heuristic_score": round(min(1.0, heuristic_bot_score), 4),
            "model_type": self.model_type,
            "qmodel_label": self.display_name if (get_qmodel_entry(self.model_type) or get_ymodel_entry(self.model_type)) else None,
            "flags": flags,
            "biometric_summary": {
                "straightness": features.get("straightness_index", 0.0),
                "velocity_mean": features.get("velocity_mean", 0.0),
                "velocity_variance": features.get("velocity_std", 0.0),
                "jitter_ratio": features.get("jitter_ratio", 0.0),
                "click_dwell_ms": features.get("avg_click_dwell_ms", -1.0)
            }
        }

    def evaluate(self, features: dict) -> dict:
        """
        Evaluates biometric features using heuristic checks plus the loaded ML model.
        Returns risk assessment dictionary.
        """
        if self.feature_schema == "alex11":
            ml_bot_prob = self._ml_bot_probability(features)
            is_bot = ml_bot_prob >= 0.5
            return {
                "decision": "BLOCKED" if is_bot else "PASSED",
                "is_bot": is_bot,
                "risk_score": round(ml_bot_prob, 4),
                "bot_probability": round(ml_bot_prob, 4),
                "heuristic_score": 0.0,
                "model_type": self.model_type,
                "qmodel_label": self.display_name,
                "flags": ["Alex 11-feature schema (native heuristics skipped)"],
                "biometric_summary": {
                    "straightness": features.get("straightness_index", 0.0),
                    "velocity_mean": features.get("v_mean", features.get("velocity_mean", 0.0)),
                    "velocity_variance": features.get("v_std", 0.0),
                    "jitter_ratio": features.get("micro_jitter_ratio", 0.0),
                    "click_dwell_ms": features.get("click_rate", 0.0),
                },
            }

        flags = []
        heuristic_bot_score = 0.0

        event_count = features.get("event_count", 0)
        straightness = features.get("straightness_index", 0.0)
        vel_std = features.get("velocity_std", 0.0)
        vel_mean = features.get("velocity_mean", 0.0)
        vel_max = features.get("velocity_max", 0.0)
        jitter_ratio = features.get("jitter_ratio", 0.0)
        click_dwell = features.get("avg_click_dwell_ms", 0.0)
        duration = features.get("duration_ms", 0.0)

        # 1. Telemetry Volume Check
        if event_count < 4:
            flags.append("Insufficient mouse telemetry data (<4 events)")
            heuristic_bot_score += 0.45

        # 2. Perfect Straight Line Check
        if straightness > 0.985 and event_count > 8:
            flags.append(f"Near-perfect linear trajectory (Straightness: {straightness:.3f})")
            heuristic_bot_score += 0.40

        # 3. Zero Velocity Variance Check (Constant speed bot)
        if vel_std < 0.015 and vel_mean > 0.1 and event_count > 5:
            flags.append("Constant speed without natural acceleration/deceleration")
            heuristic_bot_score += 0.35

        # 4. Instant Teleportation / Abnormally high velocity
        if vel_max > 15.0:
            flags.append(f"Abnormal velocity spike ({vel_max:.1f} px/ms)")
            heuristic_bot_score += 0.40

        # 5. Superhuman Click Speed
        if click_dwell >= 0.0 and click_dwell < 15.0 and event_count > 4:
            flags.append(f"Superhuman click dwell duration ({click_dwell:.1f}ms)")
            heuristic_bot_score += 0.30

        # 6. Absence of micro-tremors on long paths
        if features.get("total_distance_px", 0) > 200 and jitter_ratio == 0.0:
            flags.append("Complete absence of organic micro-tremor / sub-movements")
            heuristic_bot_score += 0.25

        # 7. Very fast completion over a meaningful distance
        if duration < 1000.0 and features.get("total_distance_px", 0) > 250:
            flags.append(f"Very fast movement sequence ({duration:.0f}ms over {features.get('total_distance_px', 0):.0f}px)")
            heuristic_bot_score += 0.25

        # 8. Procedural human-mimic (sharepad evasion signature) — Qmodel V0.1 disables this
        if self.use_procedural_mimic_heuristic and (
            straightness < 0.22
            and jitter_ratio > 0.28
            and 0.28 < vel_mean < 0.58
            and vel_max < 2.5
            and click_dwell >= 90.0
            and event_count > 15
            and duration > 4000.0
        ):
            flags.append("Procedural human-mimic movement signature")
            heuristic_bot_score += 0.32

        dynamic_boost, dynamic_flags = score_dynamic_rules(
            features, load_rules(self.evasion_rules_path),
        )
        if dynamic_boost > 0:
            flags.extend(dynamic_flags)
            heuristic_bot_score += dynamic_boost

        # 9. Wobble-linear API evasion (issue 001 — low jitter, elevated velocity, button_sequence timing)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio < 0.13
            and 0.60 < vel_mean < 1.08
            and 3.2 < vel_max < 10.0
            and 60.0 <= click_dwell < 115.0
            and 180 < event_count < 540
            and 3400.0 < duration < 9500.0
            and straightness < 0.56
        ):
            flags.append("Wobble-linear evasion signature (issue 001)")
            heuristic_bot_score += 0.40

        # 10. Low-straightness fast variant from same wobble generator
        if (
            self.use_procedural_mimic_heuristic
            and straightness < 0.22
            and jitter_ratio < 0.14
            and vel_mean > 0.68
            and event_count > 175
            and duration > 3200.0
            and 55.0 <= click_dwell < 115.0
        ):
            flags.append("Low-straightness fast evasion (issue 001 variant)")
            heuristic_bot_score += 0.34

        jerk_mean = features.get("jerk_mean", 0.0)
        acc_mean = features.get("acceleration_mean", 0.0)

        # 11. Playwright hop+jiggle inflation (issue 002 — high jitter, low jerk/acceleration)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio > 0.48
            and jerk_mean < 0.18
            and acc_mean < 0.28
            and 0.42 < vel_mean < 0.88
            and 85 < event_count < 320
            and straightness < 0.52
            and duration > 2000.0
        ):
            flags.append("Playwright micro-jiggle inflation (issue 002)")
            heuristic_bot_score += 0.48

        # 12. Broad automation kinematic mismatch (high apparent jitter, flat dynamics)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio > 0.38
            and jerk_mean < 0.22
            and acc_mean < 0.35
            and vel_mean < 0.92
            and 70 < event_count < 380
            and duration > 1800.0
        ):
            flags.append("Automation kinematic mismatch (inflated jitter, low dynamics)")
            heuristic_bot_score += 0.36

        # 13. Inflated jitter + sparse Playwright hops (issue 002 live fingerprint)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio > 0.50
            and vel_mean < 0.88
            and event_count < 280
            and straightness < 0.55
        ):
            flags.append("Inflated jitter sparse-hop automation (issue 002)")
            heuristic_bot_score += 0.40

        # 14. Extreme jitter inflation from micro-jiggle automation (live + synthetic variants)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio > 0.62
            and event_count < 300
            and straightness < 0.55
        ):
            flags.append("Extreme jitter inflation (micro-jiggle automation)")
            heuristic_bot_score += 0.52

        # 15. Fast-hop Playwright flat dynamics (issue 003 hi_dyn)
        if (
            self.use_procedural_mimic_heuristic
            and jerk_mean < 0.10
            and acc_mean < 0.22
            and 0.28 < jitter_ratio < 0.52
            and 0.82 < vel_mean < 1.50
            and 85 < event_count < 230
            and straightness < 0.55
        ):
            flags.append("Fast-hop Playwright flat dynamics (issue 003)")
            heuristic_bot_score += 0.50

        # 16. Flat dynamics automation (any jitter band, elevated velocity)
        if (
            self.use_procedural_mimic_heuristic
            and jerk_mean < 0.12
            and acc_mean < 0.25
            and vel_mean > 0.80
            and 70 < event_count < 260
            and duration > 1200.0
        ):
            flags.append("Flat-dynamics automation signature")
            heuristic_bot_score += 0.38

        # 17. mid_dense linear dense-hop (issue 004 winner)
        if (
            self.use_procedural_mimic_heuristic
            and 0.08 < jitter_ratio < 0.22
            and 0.12 < jerk_mean < 0.32
            and 0.22 < acc_mean < 0.45
            and 0.82 < vel_mean < 1.18
            and 115 < event_count < 225
            and straightness < 0.38
            and duration > 1500.0
        ):
            flags.append("Dense-hop Playwright mid_dense (issue 004)")
            heuristic_bot_score += 0.50

        # 18. Playwright dense-hop family (whole family — low jitter, moderate dynamics)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio < 0.24
            and jerk_mean > 0.08
            and acc_mean > 0.18
            and 0.45 < vel_mean < 1.28
            and 50 < event_count < 280
            and straightness < 0.50
            and duration > 1000.0
        ):
            flags.append("Playwright dense-hop family automation")
            heuristic_bot_score += 0.44

        # 19. Strengthen #10 corridor for low-jitter low-straightness (attempt 03 was 66%)
        if (
            self.use_procedural_mimic_heuristic
            and straightness < 0.24
            and jitter_ratio < 0.16
            and vel_mean > 0.65
            and event_count > 130
            and duration > 2000.0
        ):
            flags.append("Low-straightness low-jitter Playwright hop")
            heuristic_bot_score += 0.40

        # 20. Dart-hop high-jitter fast_direct (issue 005 winner)
        if (
            self.use_procedural_mimic_heuristic
            and 0.38 < jitter_ratio < 0.62
            and 1.05 < vel_mean < 1.55
            and 70 < event_count < 180
            and straightness < 0.35
            and 0.08 < jerk_mean < 0.28
            and 0.18 < acc_mean < 0.50
            and duration > 800.0
        ):
            flags.append("Playwright dart-hop high-jitter (issue 005)")
            heuristic_bot_score += 0.50

        # 21. Dart-hop family (lateral darts, elevated velocity, sparse events)
        if (
            self.use_procedural_mimic_heuristic
            and jitter_ratio > 0.42
            and vel_mean > 1.00
            and event_count < 220
            and straightness < 0.45
            and jerk_mean < 0.35
            and acc_mean > 0.16
        ):
            flags.append("Playwright dart-hop family automation")
            heuristic_bot_score += 0.44

        # 22. Min-jerk dense overflow (issue 006 — events above #18 cap of 280)
        if (
            self.use_procedural_mimic_heuristic
            and 0.10 < jitter_ratio < 0.22
            and 0.10 < jerk_mean < 0.28
            and 0.16 < acc_mean < 0.35
            and 0.65 < vel_mean < 1.05
            and event_count > 270
            and straightness < 0.28
        ):
            flags.append("Min-jerk dense Playwright overflow (issue 006)")
            heuristic_bot_score += 0.50

        # 23. High event-count low-jitter automation
        if (
            self.use_procedural_mimic_heuristic
            and event_count >= 280
            and jitter_ratio < 0.24
            and straightness < 0.40
            and 0.40 < vel_mean < 1.10
        ):
            flags.append("High event-count low-jitter Playwright automation")
            heuristic_bot_score += 0.42

        # 24. Sparse irregular min-jerk (issue 007 — ~60 events, high vel/acc, mid jitter)
        if (
            self.use_procedural_mimic_heuristic
            and 45 <= event_count <= 110
            and 1.15 < vel_mean < 1.85
            and 0.18 < jitter_ratio < 0.32
            and jerk_mean > 0.10
            and acc_mean > 0.28
            and straightness < 0.35
            and duration > 1200.0
        ):
            flags.append("Sparse irregular Playwright min-jerk (issue 007)")
            heuristic_bot_score += 0.50

        # 25. Mid-count irregular (issue 008 — events above #24 cap of 110)
        if (
            self.use_procedural_mimic_heuristic
            and 111 <= event_count <= 250
            and 1.05 < vel_mean < 1.70
            and 0.14 < jitter_ratio < 0.28
            and jerk_mean > 0.18
            and acc_mean > 0.32
            and straightness < 0.12
            and duration > 1800.0
        ):
            flags.append("Mid-count irregular Playwright min-jerk (issue 008)")
            heuristic_bot_score += 0.50

        # 26. High-str fast hops (issue 009 — vel above #24/#25 caps)
        if (
            self.use_procedural_mimic_heuristic
            and 45 <= event_count <= 95
            and vel_mean > 1.70
            and 0.14 < jitter_ratio < 0.28
            and jerk_mean > 0.28
            and acc_mean > 0.45
            and duration > 1200.0
        ):
            flags.append("High-str fast Playwright hops (issue 009)")
            heuristic_bot_score += 0.50

        # 27. Idle-correct long dwell (issue 010 — events above #26 cap, vel above #25)
        idle_ratio = features.get("idle_ratio", 0.0)
        if (
            self.use_procedural_mimic_heuristic
            and 120 <= event_count <= 220
            and vel_mean > 1.55
            and 0.14 < jitter_ratio < 0.24
            and jerk_mean > 0.35
            and acc_mean > 0.50
            and idle_ratio > 0.32
            and straightness < 0.18
            and duration > 3500.0
            and click_dwell >= 90.0
        ):
            flags.append("Idle-correct Playwright long-dwell (issue 010)")
            heuristic_bot_score += 0.50

        # 28. High-jitter mid-count (issue 011 — jitter above #25 cap of 0.28)
        if (
            self.use_procedural_mimic_heuristic
            and 100 <= event_count <= 145
            and 1.05 < vel_mean < 1.50
            and jitter_ratio > 0.28
            and jerk_mean > 0.18
            and acc_mean > 0.32
            and 0.18 < straightness < 0.45
            and idle_ratio < 0.28
            and duration > 2000.0
            and click_dwell < 110.0
        ):
            flags.append("High-jitter Playwright mid-count (issue 011)")
            heuristic_bot_score += 0.50

        # 29. Smooth low-jitter (issue 012 — jitter below #25 floor of 0.14)
        if (
            self.use_procedural_mimic_heuristic
            and 100 <= event_count <= 160
            and 1.05 < vel_mean < 1.50
            and jitter_ratio < 0.14
            and jerk_mean > 0.18
            and acc_mean > 0.32
            and straightness < 0.15
            and duration > 2000.0
        ):
            flags.append("Smooth low-jitter Playwright hops (issue 012)")
            heuristic_bot_score += 0.50

        # 30. High-jitter long trace (issue 013 — events above #25 cap of 250)
        if (
            self.use_procedural_mimic_heuristic
            and 220 <= event_count <= 340
            and 1.05 < vel_mean < 1.50
            and 0.24 < jitter_ratio < 0.36
            and jerk_mean > 0.18
            and acc_mean > 0.32
            and straightness < 0.12
            and duration > 4000.0
        ):
            flags.append("High-jitter long Playwright trace (issue 013)")
            heuristic_bot_score += 0.50

        if self.is_trained and self.model_type == "isolation_forest":
            return self._evaluate_isolation_forest(features, flags, heuristic_bot_score)

        # ML Model Prediction
        if self.is_trained and self.model:
            ml_bot_prob = self._ml_bot_probability(features)
        else:
            ml_bot_prob = 0.5

        # Combine Heuristic score and ML probability
        combined_score = min(1.0, max(min(1.0, heuristic_bot_score), ml_bot_prob))

        is_bot = combined_score >= 0.5
        decision = "BLOCKED" if is_bot else "PASSED"

        return {
            "decision": decision,
            "is_bot": is_bot,
            "risk_score": round(combined_score, 4),
            "bot_probability": round(ml_bot_prob, 4),
            "heuristic_score": round(min(1.0, heuristic_bot_score), 4),
            "model_type": self.model_type,
            "qmodel_label": self.display_name if (
                get_qmodel_entry(self.model_type)
                or get_ymodel_entry(self.model_type)
                or get_zmodel_entry(self.model_type)
            ) else None,
            "flags": flags,
            "biometric_summary": {
                "straightness": straightness,
                "velocity_mean": vel_mean,
                "velocity_variance": vel_std,
                "jitter_ratio": jitter_ratio,
                "click_dwell_ms": click_dwell
            }
        }
