import time
from bots.base_bot import BaseBot

class NaiveBot(BaseBot):
    """
    Tier 1: Naive Script Bot
    Simulates direct DOM clicks and instant headless scrapers.
    - Instant position jumps (0 intermediate movement points).
    - Instant press/release click (< 2ms).
    """
    def __init__(self):
        super().__init__("Tier 1 - Naive Bot")

    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float):
        # Naive scrapers don't record mousemove trajectories; instant jump to target
        self.record_raw(target_x, target_y, event_type=0, evt_name="mouse_move")

    def click_target(self, x: float, y: float, button: str = "mouse_left"):
        # Instant click down -> up (< 2ms)
        self.record_raw(x, y, event_type=1, target_id=1, evt_name="mouse_click_down")
        time.sleep(0.001)  # 1ms dwell
        self.record_raw(x, y, event_type=2, target_id=1, evt_name="mouse_click_up")
