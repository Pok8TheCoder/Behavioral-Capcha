import time
import math
import random
from bots.base_bot import BaseBot

class HumanoidFittsBot(BaseBot):
    """
    Tier 4: Humanoid Fitts Bot (Adversarial High-Level Test v2.0 - Organic Biomechanical Curvature)
    Engineered strictly from theoretical biophysical & human motor control principles:
    1. Minimum Jerk 5th-Degree Polynomial: x(t) = 10t^3 - 15t^4 + 6t^5
    2. Arm/Wrist Joint Parabolic Curvature & S-Curve Deviation
    3. Fitts's Law Acquisition Time: T = a + b * log2(1 + D/W)
    4. Dual-Phase Movement: Primary stroke + sub-movement correction phase
    5. Physiological Motor Tremor: Superimposed 8Hz - 12Hz vibrations
    6. Log-Normal Click Dwell Time: Gaussian press-and-release distribution
    """
    def __init__(self, target_width: float = 40.0):
        super().__init__("Tier 4 - Humanoid Fitts Bot")
        self.target_width = target_width
        self.a = 0.12  # Fitts intercept
        self.b = 0.18  # Fitts slope

    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float):
        distance = math.hypot(target_x - start_x, target_y - start_y)
        if distance < 1.0:
            return

        # Fitts's Law duration estimate (in seconds)
        index_of_difficulty = math.log2(1.0 + distance / max(5.0, self.target_width))
        total_duration = self.a + self.b * index_of_difficulty + random.uniform(-0.03, 0.05)
        total_duration = max(0.20, total_duration)

        # Primary Ballistic Target (90% - 94% of total distance with overshoot offset)
        ballistic_ratio = random.uniform(0.88, 0.94)
        ballistic_x = start_x + (target_x - start_x) * ballistic_ratio + random.uniform(-5.0, 5.0)
        ballistic_y = start_y + (target_y - start_y) * ballistic_ratio + random.uniform(-5.0, 5.0)

        # Calculate Perpendicular Vector for Wrist/Elbow Parabolic Arc Curvature
        dx_dir = ballistic_x - start_x
        dy_dir = ballistic_y - start_y
        perp_x = -dy_dir / (distance + 1e-6)
        perp_y = dx_dir / (distance + 1e-6)

        # Randomize curvature style: Arc vs S-Curve vs Deep Swing
        curve_style = random.choice(["arc", "s_curve", "elbow_swing"])
        curve_amp = random.uniform(0.08, 0.22) * distance * random.choice([-1.0, 1.0])

        # Phase 1: Primary Ballistic Minimum Jerk Stroke
        phase1_duration = total_duration * 0.78
        steps_p1 = max(18, int(phase1_duration * 100))  # ~100Hz telemetry steps
        dt1 = phase1_duration / steps_p1

        # Physiological Tremor (8Hz - 12Hz frequency band)
        f1, f2 = random.uniform(7.5, 9.5), random.uniform(10.5, 12.5)
        phi1, phi2 = random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)

        t_accum = 0.0
        for i in range(steps_p1):
            tau = (i + 1) / steps_p1
            # 5th-degree Minimum Jerk Polynomial
            poly = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)

            # Base linear interpolation
            base_x = start_x + (ballistic_x - start_x) * poly
            base_y = start_y + (ballistic_y - start_y) * poly

            # Apply Organic Joint Curvature Offset
            if curve_style == "arc":
                curve_offset = math.sin(math.pi * tau) * curve_amp
            elif curve_style == "s_curve":
                curve_offset = math.sin(2 * math.pi * tau) * curve_amp * 0.6
            else:  # elbow_swing
                curve_offset = 4 * tau * (1.0 - tau) * curve_amp

            curved_x = base_x + perp_x * curve_offset
            curved_y = base_y + perp_y * curve_offset

            # Physiological Tremor
            tremor_x = 0.8 * math.sin(2 * math.pi * f1 * t_accum + phi1) + 0.4 * math.cos(2 * math.pi * f2 * t_accum + phi2)
            tremor_y = 0.8 * math.cos(2 * math.pi * f1 * t_accum + phi1) + 0.4 * math.sin(2 * math.pi * f2 * t_accum + phi2)

            self.record_raw(curved_x + tremor_x, curved_y + tremor_y, event_type=0, evt_name="mouse_move")
            time.sleep(dt1)
            t_accum += dt1

        # Sub-movement Deceleration Pause
        pause_duration = random.uniform(0.03, 0.08)
        time.sleep(pause_duration)
        t_accum += pause_duration

        # Phase 2: Secondary Corrective Stroke to Target
        phase2_duration = total_duration * 0.22
        steps_p2 = max(8, int(phase2_duration * 100))
        dt2 = phase2_duration / steps_p2

        start_p2_x = self.events[-1]['x'] if self.events else ballistic_x
        start_p2_y = self.events[-1]['y'] if self.events else ballistic_y

        for i in range(steps_p2):
            tau = (i + 1) / steps_p2
            poly = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)

            cx = start_p2_x + (target_x - start_p2_x) * poly
            cy = start_p2_y + (target_y - start_p2_y) * poly

            tremor_x = 0.4 * math.sin(2 * math.pi * f1 * t_accum + phi1)
            tremor_y = 0.4 * math.cos(2 * math.pi * f1 * t_accum + phi1)

            self.record_raw(cx + tremor_x, cy + tremor_y, event_type=0, evt_name="mouse_move")
            time.sleep(dt2)
            t_accum += dt2

    def click_target(self, x: float, y: float, button: str = "mouse_left"):
        # Human click dwell time: Gaussian distribution N(mu=88ms, sigma=16ms)
        dwell_ms = max(45.0, random.gauss(88.0, 16.0)) / 1000.0

        self.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
        time.sleep(dwell_ms)
        self.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")
