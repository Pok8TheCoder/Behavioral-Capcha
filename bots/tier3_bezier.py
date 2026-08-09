import time
import random
from bots.base_bot import BaseBot

class BezierBot(BaseBot):
    """
    Tier 3: Bezier Curve Bot
    Navigates along smooth Cubic Bezier curves with parametric velocity curves.
    - Curved path (S_ratio < 1.0).
    - Varying velocity.
    - Zero physiological micro-tremor (jitter_ratio = 0.0).
    """
    def __init__(self, steps: int = 40, step_delay_sec: float = 0.008):
        super().__init__("Tier 3 - Bezier Bot")
        self.steps = steps
        self.step_delay_sec = step_delay_sec

    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float):
        # Calculate random control points for cubic Bezier curve
        dx = target_x - start_x
        dy = target_y - start_y

        p0 = (start_x, start_y)
        p3 = (target_x, target_y)

        # Perpendicular offsets for control points
        p1 = (start_x + dx * 0.25 + random.uniform(-60, 60), start_y + dy * 0.25 + random.uniform(-60, 60))
        p2 = (start_x + dx * 0.75 + random.uniform(-60, 60), start_y + dy * 0.75 + random.uniform(-60, 60))

        for i in range(self.steps):
            u = i / max(1, self.steps - 1)
            # Cubic Bezier formula
            cx = (1-u)**3 * p0[0] + 3*(1-u)**2 * u * p1[0] + 3*(1-u) * u**2 * p2[0] + u**3 * p3[0]
            cy = (1-u)**3 * p0[1] + 3*(1-u)**2 * u * p1[1] + 3*(1-u) * u**2 * p2[1] + u**3 * p3[1]

            self.record_raw(cx, cy, event_type=0, evt_name="mouse_move")
            time.sleep(self.step_delay_sec)

    def click_target(self, x: float, y: float, button: str = "mouse_left"):
        self.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
        time.sleep(random.uniform(0.010, 0.025)) # 10-25ms dwell
        self.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")
