"""
High-Speed Privacy-First Input Telemetry Recorder (v3.3 Multi-Monitor & Organized Datasets)
---------------------------------------------------------------------------------------------
Dataset Storage:
- Automatically saves human recordings to datasets/human/
- Periodic RAM snapshots saved to temp/
- Scratch/throwaway benchmark datasets saved to temp/scratch_datasets/

Multi-Monitor & Resolution Independence Features:
- Virtual Desktop Bounding: Detects multi-monitor arrangements (including negative X/Y monitors).
- Quantized Normalized Screen Mapping: Maps virtual screen space to 16-bit uint16 (0..65535).
  Sub-pixel accuracy across 1080p, 1440p, 4K, 8K, and multi-monitor setups.
- Dual CSV Export: Outputs both raw screen pixel coordinates (x_px, y_px) AND normalized 
  resolution-independent ratios (x_norm, y_norm in 0.0000..1.0000).

Privacy Guarantee:
- ZERO logging of text, passwords, letters (A-Z), digits (0-9), or symbols/punctuation.
- ONLY records non-sensitive functionality/modifier/navigation keys.
"""

import os
import sys
import time
import struct
import atexit
import signal
import threading
import glob
import numpy as np
from datetime import datetime
from pynput import mouse, keyboard

# Win32 High-Precision Timer & Virtual Screen Metrics setup
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception as e:
        print(f"[Warning] Failed setting winmm timeBeginPeriod(1): {e}")

    u32 = ctypes.windll.user32
    SCREEN_X_MIN = u32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
    SCREEN_Y_MIN = u32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
    SCREEN_WIDTH = max(1, u32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
    SCREEN_HEIGHT = max(1, u32.GetSystemMetrics(79)) # SM_CYVIRTUALSCREEN
else:
    SCREEN_X_MIN = 0
    SCREEN_Y_MIN = 0
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

# Strict Allowlist of Non-Sensitive Special Keys
ALLOWED_SPECIAL_KEYS = {
    # Modifiers
    'ctrl', 'ctrl_l', 'ctrl_r',
    'alt', 'alt_l', 'alt_r', 'alt_gr',
    'shift', 'shift_l', 'shift_r',
    'cmd', 'cmd_l', 'cmd_r', 'win',
    
    # Navigation & Editing
    'tab', 'caps_lock', 'esc', 'enter', 'backspace', 'delete',
    'insert', 'home', 'end', 'page_up', 'page_down',
    'up', 'down', 'left', 'right',
    
    # Function keys
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
    'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    
    # System / Lock keys
    'print_screen', 'scroll_lock', 'pause', 'num_lock', 'menu'
}

# Binary Struct Format (20 Bytes per record):
# <Q  : uint64 (microsecond delta dt_us since recording start)
# H   : uint16 (quantized x coordinate: 0..65535 mapping normalized Virtual Desktop X)
# H   : uint16 (quantized y coordinate: 0..65535 mapping normalized Virtual Desktop Y)
# b   : int8   (dx scroll step: -128..127)
# b   : int8   (dy scroll step: -128..127)
# B   : uint8  (event_type: 0=move, 1=click_down, 2=click_up, 3=scroll, 4=key_down, 5=key_up)
# H   : uint16 (target_id: mapped mouse button ID or special key ID)
STRUCT_FMT = "<QHHbbBH"
RECORD_SIZE = struct.calcsize(STRUCT_FMT)  # 20 bytes

# In-Memory Binary Buffer & Lock
ram_binary_buffer = bytearray()
buffer_lock = threading.Lock()

start_perf_ns = None
start_time_epoch = None
end_time_epoch = None
event_count = 0
is_flushed = False
autosave_seq = 0
AUTOSAVE_INTERVAL_SEC = 300

last_mouse_x = 0
last_mouse_y = 0

# Dynamic ID Mapping for Mouse Buttons and Special Keys
ID_TO_NAME = {
    0: "none",
    1: "mouse_left",
    2: "mouse_right",
    3: "mouse_middle",
    4: "mouse_x1",
    5: "mouse_x2"
}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}
next_id = 6

def get_target_id(name: str) -> int:
    global next_id
    if not name:
        return 0

    name_clean = name.lower()
    if name_clean not in NAME_TO_ID:
        NAME_TO_ID[name_clean] = next_id
        ID_TO_NAME[next_id] = name_clean
        next_id += 1

    return NAME_TO_ID[name_clean]

def get_allowed_key_name(key) -> str:
    """Filters keyboard inputs, returning only non-sensitive allowed special key names."""
    if isinstance(key, keyboard.Key):
        kname = key.name.lower()
        if kname in ALLOWED_SPECIAL_KEYS:
            return f"key_{kname}"
        base_name = kname.split('_')[0]
        if base_name in ALLOWED_SPECIAL_KEYS:
            return f"key_{kname}"
    return None

def record_event(x: float, y: float, dx: int, dy: int, event_type: int, target_name: str = "none"):
    global ram_binary_buffer, start_perf_ns, start_time_epoch, end_time_epoch, event_count, last_mouse_x, last_mouse_y

    now_ns = time.perf_counter_ns()
    now_epoch = time.time()

    if start_perf_ns is None:
        start_perf_ns = now_ns
        start_time_epoch = now_epoch
    end_time_epoch = now_epoch

    last_mouse_x = float(x)
    last_mouse_y = float(y)

    dt_us = int((now_ns - start_perf_ns) // 1000)

    # Normalize coordinate relative to Virtual Screen Bounding Box -> [0.0, 1.0]
    norm_x = (float(x) - SCREEN_X_MIN) / float(SCREEN_WIDTH)
    norm_y = (float(y) - SCREEN_Y_MIN) / float(SCREEN_HEIGHT)

    # Quantize normalized ratio [0.0, 1.0] to 16-bit uint16 integer [0..65535]
    quant_x = max(0, min(65535, int(round(norm_x * 65535.0))))
    quant_y = max(0, min(65535, int(round(norm_y * 65535.0))))

    clamp_dx = max(-128, min(127, int(dx)))
    clamp_dy = max(-128, min(127, int(dy)))
    t_id = get_target_id(target_name)

    packed = struct.pack(
        STRUCT_FMT,
        dt_us,
        quant_x,
        quant_y,
        clamp_dx,
        clamp_dy,
        event_type,
        t_id
    )

    with buffer_lock:
        ram_binary_buffer.extend(packed)
        event_count += 1

# Mouse Event Handlers
def on_mouse_move(x, y):
    record_event(x, y, 0, 0, event_type=0, target_name="none")

def on_mouse_click(x, y, button, pressed):
    if hasattr(button, 'name') and button.name:
        bname = f"mouse_{button.name.lower()}"
    else:
        bname = f"mouse_{str(button).replace('Button.', '').lower()}"
    
    evt_type = 1 if pressed else 2
    record_event(x, y, 0, 0, event_type=evt_type, target_name=bname)

def on_mouse_scroll(x, y, dx, dy):
    record_event(x, y, dx, dy, event_type=3, target_name="none")

# Keyboard Event Handlers (Privacy-Filtered)
def on_key_press(key):
    key_name = get_allowed_key_name(key)
    if key_name:
        record_event(last_mouse_x, last_mouse_y, 0, 0, event_type=4, target_name=key_name)

def on_key_release(key):
    key_name = get_allowed_key_name(key)
    if key_name:
        record_event(last_mouse_x, last_mouse_y, 0, 0, event_type=5, target_name=key_name)

def periodic_autosave_worker():
    global autosave_seq
    while not is_flushed:
        time.sleep(AUTOSAVE_INTERVAL_SEC)
        if is_flushed or event_count == 0:
            break

        autosave_seq += 1
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)

        start_dt = datetime.fromtimestamp(start_time_epoch or time.time()).strftime("%Y%m%d_%H%M%S")
        autosave_file = os.path.join(temp_dir, f"autosave_{start_dt}_part{autosave_seq:03d}.bin")

        with buffer_lock:
            buffer_copy = bytes(ram_binary_buffer)

        try:
            with open(autosave_file, "wb") as f:
                f.write(buffer_copy)
            print(f"\n[Autosave] Periodic RAM snapshot saved ({len(buffer_copy)/1024:.1f} KB) -> [file:///{autosave_file.replace(os.sep, '/')}]")
        except Exception as e:
            print(f"\n[Autosave Warning] Periodic snapshot failed: {e}")

def calculate_polling_stats(buffer_copy: bytes):
    if len(buffer_copy) < RECORD_SIZE * 2:
        return 0.0

    dts = []
    offsets = range(0, len(buffer_copy) - RECORD_SIZE + 1, RECORD_SIZE)
    prev_us = None

    for offset in offsets:
        dt_us = struct.unpack_from("<Q", buffer_copy, offset)[0]
        if prev_us is not None:
            diff_ms = (dt_us - prev_us) / 1000.0
            if 0.1 <= diff_ms <= 20.0:  # Active movement window (< 20ms)
                dts.append(diff_ms)
        prev_us = dt_us

    active_hz = (1000.0 / float(np.median(dts))) if dts else 0.0
    return active_hz

def unpack_bin_to_csv(bin_filepath: str, csv_filepath: str, base_epoch: float = None):
    """Unpacks binary dataset into CSV with both raw pixel coords and normalized 0..1 ratios."""
    event_type_names = {
        0: "mouse_move",
        1: "mouse_click_down",
        2: "mouse_click_up",
        3: "mouse_scroll",
        4: "key_down",
        5: "key_up"
    }

    with open(bin_filepath, "rb") as f_bin:
        data = f_bin.read()

    total_records = len(data) // RECORD_SIZE
    if total_records == 0:
        print(f"[Converter] Empty or corrupted binary file: {bin_filepath}")
        return

    cumulative_sc_x = 0.0
    cumulative_sc_y = 0.0

    with open(csv_filepath, "w", encoding="utf-8") as f_csv:
        # Include Display Metadata Header
        f_csv.write(f"# Display Metrics: VirtualBounds=({SCREEN_X_MIN},{SCREEN_Y_MIN}), Size=({SCREEN_WIDTH}x{SCREEN_HEIGHT})\n")
        
        if base_epoch is not None:
            f_csv.write("timestamp_epoch,rel_time_ms,x_px,y_px,x_norm,y_norm,scroll_x,scroll_y,scroll_dx,scroll_dy,event_type,target_item\n")
        else:
            f_csv.write("rel_time_ms,x_px,y_px,x_norm,y_norm,scroll_x,scroll_y,scroll_dx,scroll_dy,event_type,target_item\n")

        for offset in range(0, len(data) - RECORD_SIZE + 1, RECORD_SIZE):
            dt_us, quant_x, quant_y, dx, dy, evt_type, t_id = struct.unpack_from(STRUCT_FMT, data, offset)
            
            rel_ms = dt_us / 1000.0
            
            x_norm = quant_x / 65535.0
            y_norm = quant_y / 65535.0

            x_px = round(SCREEN_X_MIN + (x_norm * SCREEN_WIDTH), 1)
            y_px = round(SCREEN_Y_MIN + (y_norm * SCREEN_HEIGHT), 1)

            cumulative_sc_x += dx
            cumulative_sc_y += dy

            evt_str = event_type_names.get(evt_type, "unknown")
            target_str = ID_TO_NAME.get(t_id, f"item_{t_id}")

            if base_epoch is not None:
                abs_epoch = base_epoch + (rel_ms / 1000.0)
                f_csv.write(f"{abs_epoch:.6f},{rel_ms:.3f},{x_px},{y_px},{x_norm:.5f},{y_norm:.5f},{cumulative_sc_x:.1f},{cumulative_sc_y:.1f},{dx},{dy},{evt_str},{target_str}\n")
            else:
                f_csv.write(f"{rel_ms:.3f},{x_px},{y_px},{x_norm:.5f},{y_norm:.5f},{cumulative_sc_x:.1f},{cumulative_sc_y:.1f},{dx},{dy},{evt_str},{target_str}\n")

    print(f"[Converter] Successfully converted {total_records:,} records -> [file:///{csv_filepath.replace(os.sep, '/')}]")

def recover_autosaves():
    search_paths = [
        os.path.join(os.path.dirname(__file__), "temp"),
        os.path.join(os.path.dirname(__file__), "temp", "scratch_datasets"),
        os.path.join(os.path.dirname(__file__), "datasets"),
        os.path.join(os.path.dirname(__file__), "datasets", "human"),
        os.path.join(os.path.dirname(__file__), "datasets", "bot")
    ]
    
    bin_files = []
    for sp in search_paths:
        bin_files.extend(glob.glob(os.path.join(sp, "*.bin")))

    if not bin_files:
        print("[Recovery] No leftover .bin files found to recover.")
        return

    print(f"[Recovery Engine] Found {len(bin_files)} binary dataset files. Converting...")
    for bin_file in bin_files:
        csv_file = bin_file[:-4] + ".csv"
        unpack_bin_to_csv(bin_file, csv_file)

def flush_ram_to_csv():
    global is_flushed
    if is_flushed:
        return
    is_flushed = True

    if IS_WINDOWS:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

    with buffer_lock:
        if not ram_binary_buffer or event_count == 0:
            print("\n[Recorder] No input events were recorded.")
            return
        buffer_copy = bytes(ram_binary_buffer)

    end_t = end_time_epoch or time.time()
    start_t = start_time_epoch or end_t
    start_dt = datetime.fromtimestamp(start_t).strftime("%Y%m%d_%H%M%S")
    end_dt = datetime.fromtimestamp(end_t).strftime("%Y%m%d_%H%M%S")

    duration = end_t - start_t
    session_avg_hz = (event_count / duration) if duration > 0 else 0.0
    active_motion_hz = calculate_polling_stats(buffer_copy)

    # Human recordings default to datasets/human/
    output_dir = os.path.join(os.path.dirname(__file__), "datasets", "human")
    os.makedirs(output_dir, exist_ok=True)

    filename_base = f"input_log_{start_dt}_to_{end_dt}"
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    bin_path = os.path.join(output_dir, f"{filename_base}.bin")

    print(f"\n" + "=" * 65)
    print(f"[Recorder Engine v3.3] Flushing RAM binary buffer...")
    print(f" -> Storage Destination:     datasets/human/")
    print(f" -> Display Resolution:       {SCREEN_WIDTH}x{SCREEN_HEIGHT} (Virtual Desktop Offset: {SCREEN_X_MIN},{SCREEN_Y_MIN})")
    print(f" -> Total Events Recorded:     {event_count:,}")
    print(f" -> Compact RAM Buffer Size:   {len(buffer_copy) / 1024:.2f} KB (20 bytes/event)")
    print(f" -> Total Session Duration:    {duration:.2f} seconds")
    print(f" -> Active Motion Hardware Rate: {active_motion_hz:.1f} Hz (True Sensor Frequency)")
    print(f" -> Session Overall Average:   {session_avg_hz:.1f} Hz (Includes Pauses/Idle)")
    print("=" * 65)

    # 1. Save binary backup
    with open(bin_path, "wb") as f_bin:
        f_bin.write(buffer_copy)
    print(f"[Saver] Compact binary buffer saved to: [file:///{bin_path.replace(os.sep, '/')}]")

    # 2. Convert binary RAM buffer to CSV
    unpack_bin_to_csv(bin_path, csv_path, base_epoch=start_t)

# Register exit handlers
atexit.register(flush_ram_to_csv)

def handle_signal(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == "__main__":
    if "--recover" in sys.argv:
        recover_autosaves()
        sys.exit(0)

    print("=" * 65)
    print(" PRIVACY-FIRST INPUT TELEMETRY RECORDER v3.3 (ORGANIZED DATASETS)")
    print("=" * 65)
    print(f" Output Location: datasets/human/")
    print(f" Detected Display Bounds: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (Virtual Desktop Offset: {SCREEN_X_MIN},{SCREEN_Y_MIN})")
    print(" PRIVACY PROMISE: ZERO logging of text, letters, digits, or passwords.")
    print(" - Max Polling Speed: 1ms Win32 Timer (timeBeginPeriod)")
    print(" - Precision Timestamp: 64-bit uint64 Microseconds (0.001ms precision)")
    print(" - Normalized Mapping: x_norm & y_norm (0.0000..1.0000 ratio)")
    print(" - Controls: Press Ctrl+C at any time to export to CSV\n")

    # Start periodic autosave background thread
    autosave_thread = threading.Thread(target=periodic_autosave_worker, daemon=True)
    autosave_thread.start()

    # Start mouse & keyboard listeners
    mouse_listener = mouse.Listener(
        on_move=on_mouse_move,
        on_click=on_mouse_click,
        on_scroll=on_mouse_scroll
    )
    key_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release
    )

    mouse_listener.start()
    key_listener.start()

    try:
        while mouse_listener.is_alive() or key_listener.is_alive():
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        mouse_listener.stop()
        key_listener.stop()
        flush_ram_to_csv()
