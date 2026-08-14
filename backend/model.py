import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.feature_extractor import feature_dict_to_vector

IFOREST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_iforest_model.joblib")
RF_HUMANLIKE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_humanlike_model.joblib")
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_model.joblib")

MODEL_REGISTRY = {
    "random_forest_humanlike": RF_HUMANLIKE_MODEL_PATH,
    "isolation_forest": IFOREST_MODEL_PATH,
    "random_forest": RF_MODEL_PATH,
}

class BehavioralClassifier:
    def __init__(self):
        self.model = None
        self.model_type = "unloaded"
        self.feature_names = []
        self.metadata = {}
        self.is_trained = False
        requested_model = os.getenv("BEHAVIORAL_MODEL", "random_forest_humanlike")
        if not self.load_model(requested_model):
            print(f"[Model] Could not load requested model '{requested_model}'; trying available artifacts.")
            for model_name in ("random_forest_humanlike", "isolation_forest", "random_forest"):
                if self.load_model(model_name):
                    return
            self.train_synthetic_model()

    def available_models(self) -> dict:
        """Returns the selectable model artifacts and whether each is present on disk."""
        return {
            model_name: {
                "available": os.path.exists(model_path),
                "path": model_path,
            }
            for model_name, model_path in MODEL_REGISTRY.items()
        }

    def load_model(self, model_name: str) -> bool:
        """Loads one registered artifact without changing the API contract."""
        normalized_name = model_name.strip().lower()
        model_path = MODEL_REGISTRY.get(normalized_name)
        if not model_path:
            print(f"[Model] Unknown model requested: {model_name}")
            return False
        if not os.path.exists(model_path):
            print(f"[Model] Model artifact is missing: {model_path}")
            return False

        try:
            package = joblib.load(model_path)
            self.model = package["model"] if isinstance(package, dict) and "model" in package else package
            self.feature_names = package.get("feature_names", []) if isinstance(package, dict) else []
            self.metadata = package.get("metadata", {}) if isinstance(package, dict) else {}
            self.model_type = normalized_name
            self.is_trained = True
            print(f"[Model] Loaded {normalized_name} from {model_path}")
            return True
        except Exception as e:
            print(f"[Model] Error loading {normalized_name}: {e}")
            return False

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
        if 0 < click_dwell < 15.0:
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

        if self.is_trained and self.model_type == "isolation_forest":
            return self._evaluate_isolation_forest(features, flags, heuristic_bot_score)

        # ML Model Prediction
        vec = [feature_dict_to_vector(features)]
        if self.is_trained and self.model:
            ml_bot_prob = float(self.model.predict_proba(vec)[0][1])
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
            "flags": flags,
            "biometric_summary": {
                "straightness": straightness,
                "velocity_mean": vel_mean,
                "velocity_variance": vel_std,
                "jitter_ratio": jitter_ratio,
                "click_dwell_ms": click_dwell
            }
        }
