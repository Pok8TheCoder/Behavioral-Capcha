import time
import math
import struct
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Compatible with input_recorder Struct (<QHHbbBH, 20 bytes)
STRUCT_FMT = "<QHHbbBH"
RECORD_SIZE = struct.calcsize(STRUCT_FMT)

class BaseBot(ABC):
    def __init__(self, name: str):
        self.name = name
        self.events: List[Dict[str, Any]] = []
        self.binary_buffer = bytearray()
        self.start_perf_ns = None
        self.start_time_epoch = None
        self.end_time_epoch = None
        self.cumulative_scroll_x = 0.0
        self.cumulative_scroll_y = 0.0

    def reset(self):
        self.events.clear()
        self.binary_buffer.clear()
        self.start_perf_ns = None
        self.start_time_epoch = None
        self.end_time_epoch = None
        self.cumulative_scroll_x = 0.0
        self.cumulative_scroll_y = 0.0

    def record_raw(self, x: float, y: float, dx: float = 0.0, dy: float = 0.0, event_type: int = 0, target_id: int = 0, evt_name: str = "mouse_move"):
        now_ns = time.perf_counter_ns()
        now_epoch = time.time()

        if self.start_perf_ns is None:
            self.start_perf_ns = now_ns
            self.start_time_epoch = now_epoch
        self.end_time_epoch = now_epoch

        dt_us = int((now_ns - self.start_perf_ns) // 1000)

        self.cumulative_scroll_x += dx
        self.cumulative_scroll_y += dy

        clamp_x = max(0, min(65535, int(x)))
        clamp_y = max(0, min(65535, int(y)))
        clamp_dx = max(-128, min(127, int(dx)))
        clamp_dy = max(-128, min(127, int(dy)))

        packed = struct.pack(
            STRUCT_FMT,
            dt_us,
            clamp_x,
            clamp_y,
            clamp_dx,
            clamp_dy,
            event_type,
            target_id
        )
        self.binary_buffer.extend(packed)

        self.events.append({
            "x": round(x, 2),
            "y": round(y, 2),
            "scroll_x": round(self.cumulative_scroll_x, 1),
            "scroll_y": round(self.cumulative_scroll_y, 1),
            "scroll_dx": int(dx),
            "scroll_dy": int(dy),
            "t": round((now_epoch - self.start_time_epoch) * 1000.0, 3),
            "event_type": evt_name
        })

    @abstractmethod
    def move_to(self, start_x: float, start_y: float, target_x: float, target_y: float):
        """Generates trajectory from start (x0, y0) to target (x1, y1)."""
        pass

    @abstractmethod
    def click_target(self, x: float, y: float, button: str = "mouse_left"):
        """Generates click down and click up telemetry at specified location."""
        pass
