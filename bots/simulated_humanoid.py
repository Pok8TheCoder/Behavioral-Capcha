"""Offline human-like motor bot with simulated timestamps (no wall-clock sleep)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from bots.base_bot import BaseBot


def _minimum_jerk(tau: float) -> float:
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


@dataclass
class MotorProfile:
    name: str
    fitts_a: float
    fitts_b: float
    tremor_scale: float
    curvature_scale: float
    pause_probability: float
    overshoot_probability: float
    hesitation_ms: Tuple[float, float]
    sample_hz: Tuple[float, float]


DEFAULT_PROFILES: List[MotorProfile] = [
    MotorProfile("minimum_jerk_precise", 0.11, 0.16, 0.55, 0.12, 0.14, 0.20, (18.0, 55.0), (85.0, 120.0)),
    MotorProfile("curved_submovement", 0.13, 0.19, 0.75, 0.20, 0.22, 0.35, (25.0, 80.0), (70.0, 105.0)),
    MotorProfile("hesitant_trackpad", 0.15, 0.21, 1.05, 0.14, 0.38, 0.22, (40.0, 120.0), (55.0, 90.0)),
    MotorProfile("fast_confident_mouse", 0.09, 0.14, 0.40, 0.08, 0.08, 0.12, (12.0, 40.0), (95.0, 140.0)),
    MotorProfile("imperfect_targeting", 0.12, 0.18, 0.85, 0.17, 0.30, 0.48, (30.0, 95.0), (65.0, 100.0)),
    MotorProfile("careful_drag_stroke", 0.14, 0.20, 0.65, 0.10, 0.26, 0.15, (35.0, 100.0), (75.0, 115.0)),
]


class SimulatedHumanoidBot(BaseBot):
    """Humanoid Fitts + minimum-jerk bot that advances a synthetic clock."""

    simulated = True

    def __init__(self, profile: MotorProfile, target_width: float = 40.0):
        super().__init__(f"Simulated Humanoid ({profile.name})")
        self.profile = profile
        self.target_width = target_width
        self._t_ms = 0.0
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

    @classmethod
    def random(cls, rng: random.Random) -> SimulatedHumanoidBot:
        profile = rng.choice(DEFAULT_PROFILES)
        width = rng.uniform(24.0, 72.0)
        return cls(profile=profile, target_width=width)

    def reset(self) -> None:
        super().reset()
        self._t_ms = 0.0
        self._last_x = None
        self._last_y = None

    def advance(self, ms: float) -> None:
        self._t_ms += max(0.4, ms + random.gauss(0.0, 1.4))

    def record_raw(
        self,
        x: float,
        y: float,
        dx: float = 0.0,
        dy: float = 0.0,
        event_type: int = 0,
        target_id: int = 0,
        evt_name: str = "mouse_move",
    ) -> None:
        self.events.append({
            "x": round(x, 2),
            "y": round(y, 2),
            "scroll_x": round(self.cumulative_scroll_x, 1),
            "scroll_y": round(self.cumulative_scroll_y, 1),
            "scroll_dx": int(dx),
            "scroll_dy": int(dy),
            "t": round(self._t_ms, 3),
            "event_type": evt_name,
        })
        self._last_x = x
        self._last_y = y

    def idle_jitter(self, duration_ms: float, x: float, y: float) -> None:
        steps = max(2, int(duration_ms / random.uniform(14.0, 28.0)))
        for _ in range(steps):
            self.advance(duration_ms / steps)
            self.record_raw(
                x + random.gauss(0.0, 0.7),
                y + random.gauss(0.0, 0.7),
                evt_name="mouse_move",
            )

    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float) -> None:
        distance = math.hypot(target_x - start_x, target_y - start_y)
        if distance < 1.0:
            self.record_raw(target_x, target_y, evt_name="mouse_move")
            return

        fitts_id = math.log2(1.0 + distance / max(5.0, self.target_width))
        total_duration = (
            self.profile.fitts_a
            + self.profile.fitts_b * fitts_id
            + random.uniform(-0.03, 0.05)
        )
        total_duration = max(0.18, total_duration)

        ballistic_ratio = random.uniform(0.86, 0.94)
        ballistic_x = start_x + (target_x - start_x) * ballistic_ratio + random.uniform(-5.0, 5.0)
        ballistic_y = start_y + (target_y - start_y) * ballistic_ratio + random.uniform(-5.0, 5.0)

        dx_dir = ballistic_x - start_x
        dy_dir = ballistic_y - start_y
        perp_x = -dy_dir / (distance + 1e-6)
        perp_y = dx_dir / (distance + 1e-6)

        curve_style = random.choice(["arc", "s_curve", "elbow_swing"])
        curve_amp = (
            random.uniform(0.06, 0.22)
            * distance
            * self.profile.curvature_scale
            * random.choice([-1.0, 1.0])
        )

        phase1_duration = total_duration * random.uniform(0.74, 0.82)
        hz = random.uniform(*self.profile.sample_hz)
        steps_p1 = max(16, int(phase1_duration * hz))
        dt1_ms = (phase1_duration * 1000.0) / steps_p1

        f1 = random.uniform(7.5, 9.5)
        f2 = random.uniform(10.5, 12.5)
        phi1 = random.uniform(0.0, math.tau)
        phi2 = random.uniform(0.0, math.tau)
        tremor_scale = self.profile.tremor_scale

        t_accum = 0.0
        for i in range(steps_p1):
            tau = (i + 1) / steps_p1
            poly = _minimum_jerk(tau)
            base_x = start_x + (ballistic_x - start_x) * poly
            base_y = start_y + (ballistic_y - start_y) * poly

            if curve_style == "arc":
                curve_offset = math.sin(math.pi * tau) * curve_amp
            elif curve_style == "s_curve":
                curve_offset = math.sin(2.0 * math.pi * tau) * curve_amp * 0.6
            else:
                curve_offset = 4.0 * tau * (1.0 - tau) * curve_amp

            tremor_x = tremor_scale * (
                0.8 * math.sin(math.tau * f1 * t_accum + phi1)
                + 0.4 * math.cos(math.tau * f2 * t_accum + phi2)
            )
            tremor_y = tremor_scale * (
                0.8 * math.cos(math.tau * f1 * t_accum + phi1)
                + 0.4 * math.sin(math.tau * f2 * t_accum + phi2)
            )

            self.advance(dt1_ms)
            self.record_raw(
                base_x + perp_x * curve_offset + tremor_x,
                base_y + perp_y * curve_offset + tremor_y,
                evt_name="mouse_move",
            )
            t_accum += phase1_duration / steps_p1

        if random.random() < self.profile.pause_probability:
            self.advance(random.uniform(25.0, 140.0))

        phase2_duration = total_duration - phase1_duration
        steps_p2 = max(8, int(phase2_duration * hz))
        dt2_ms = (phase2_duration * 1000.0) / steps_p2

        start_p2_x = self._last_x if self._last_x is not None else ballistic_x
        start_p2_y = self._last_y if self._last_y is not None else ballistic_y

        for i in range(steps_p2):
            tau = (i + 1) / steps_p2
            poly = _minimum_jerk(tau)
            cx = start_p2_x + (target_x - start_p2_x) * poly
            cy = start_p2_y + (target_y - start_p2_y) * poly
            tremor_x = tremor_scale * 0.45 * math.sin(math.tau * f1 * t_accum + phi1)
            tremor_y = tremor_scale * 0.45 * math.cos(math.tau * f1 * t_accum + phi1)
            self.advance(dt2_ms)
            self.record_raw(cx + tremor_x, cy + tremor_y, evt_name="mouse_move")
            t_accum += phase2_duration / steps_p2

        if random.random() < self.profile.overshoot_probability:
            overshoot_px = random.uniform(4.0, 18.0)
            ux = (target_x - start_x) / (distance + 1e-6)
            uy = (target_y - start_y) / (distance + 1e-6)
            ox = target_x + ux * overshoot_px
            oy = target_y + uy * overshoot_px
            self.advance(random.uniform(20.0, 55.0))
            self.move_to(self._last_x or target_x, self._last_y or target_y, ox, oy)
            self.advance(random.uniform(30.0, 90.0))
            self.move_to(self._last_x or ox, self._last_y or oy, target_x, target_y)

    def click_target(self, x: float, y: float, button: str = "mouse_left") -> None:
        self.advance(random.uniform(*self.profile.hesitation_ms))
        click_x = x + random.gauss(0.0, 1.6)
        click_y = y + random.gauss(0.0, 1.6)
        self.record_raw(click_x, click_y, event_type=1, target_id=1, evt_name="mouse_click_down")
        dwell_ms = max(42.0, random.gauss(88.0, 18.0))
        self.advance(dwell_ms)
        self.record_raw(
            click_x + random.gauss(0.0, 0.35),
            click_y + random.gauss(0.0, 0.35),
            event_type=2,
            target_id=1,
            evt_name="mouse_click_up",
        )
