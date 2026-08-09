import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.feature_extractor import feature_dict_to_vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "behavioral_rf_model.joblib")

class BehavioralClassifier:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.is_trained = True
                return
            except Exception as e:
                print(f"[Model] Error loading pre-existing model: {e}")

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
        self.is_trained = True

        try:
            joblib.dump(clf, MODEL_PATH)
        except Exception as e:
            print(f"[Model] Warning: Failed saving trained model: {e}")

    def evaluate(self, features: dict) -> dict:
        """
        Evaluates biometric features using heuristic rule-based checks combined with Random Forest ML score.
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

        # ML Model Prediction
        vec = [feature_dict_to_vector(features)]
        if self.is_trained and self.model:
            ml_bot_prob = float(self.model.predict_proba(vec)[0][1])
        else:
            ml_bot_prob = 0.5

        # Combine Heuristic score and ML probability
        combined_score = min(1.0, (0.4 * min(1.0, heuristic_bot_score)) + (0.6 * ml_bot_prob))

        is_bot = combined_score >= 0.5
        decision = "BLOCKED" if is_bot else "PASSED"

        return {
            "decision": decision,
            "is_bot": is_bot,
            "risk_score": round(combined_score, 4),
            "bot_probability": round(ml_bot_prob, 4),
            "heuristic_score": round(min(1.0, heuristic_bot_score), 4),
            "flags": flags,
            "biometric_summary": {
                "straightness": straightness,
                "velocity_mean": vel_mean,
                "velocity_variance": vel_std,
                "jitter_ratio": jitter_ratio,
                "click_dwell_ms": click_dwell
            }
        }
