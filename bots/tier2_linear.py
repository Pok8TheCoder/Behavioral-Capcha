import time
from bots.base_bot import BaseBot

class LinearBot(BaseBot):
    """
    Tier 2: Linear Vector Bot
    Moves in perfect straight lines using constant velocity linear interpolation.
    - Straightness Index S_ratio = 1.0.
    - Uniform time steps, zero velocity variance.
    """
    def __init__(self, steps: int = 30, step_delay_sec: float = 0.010):
        super().__init__("Tier 2 - Linear Bot")
        self.steps = steps
        self.step_delay_sec = step_delay_sec

    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float):
        for i in range(self.steps):
            alpha = i / max(1, self.steps - 1)
            cx = start_x + alpha * (target_x - start_x)
            cy = start_y + alpha * (target_y - start_y)
            self.record_raw(cx, cy, event_type=0, evt_name="mouse_move")
            time.sleep(self.step_delay_sec)

    def click_target(self, x: float, y: float, button: str = "mouse_left"):
        self.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
        time.sleep(0.005) # 5ms click dwell
        self.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")
