"""
High-Speed Privacy-First Input Telemetry Recorder (v3.0)
--------------------------------------------------------
Privacy Guarantee:
- ZERO logging of text, passwords, letters (A-Z), digits (0-9), or symbols/punctuation.
- ONLY records non-sensitive functionality/modifier/navigation keys:
  Ctrl, Alt, Win/Cmd, Shift, Tab, CapsLock, Esc, Enter, Backspace, Delete,
  Insert, Home, End, PageUp, PageDown, Arrow Keys (Up/Down/Left/Right),
  Function Keys (F1-F12), PrintScreen, ScrollLock, Pause, NumLock.

Features:
- Combined Mouse Dynamics + Non-Sensitive Keystroke Biometrics.
- High-Speed 1ms Win32 Timer (winmm.timeBeginPeriod(1)).
- Sub-Millisecond 64-bit uint64 Microsecond Timestamps (time.perf_counter_ns()).
- Ultra-Compact Binary RAM Storage (20 Bytes per event).
- 300s Periodic Autosave & Emergency Recovery Engine (--recover).
"""

import os
import sys
import time
import struct
import atexit
import signal
import threading
import glob
from datetime import datetime
from pynput import mouse, keyboard

# Win32 High-Precision Timer Resolution setup
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception as e:
        print(f"[Warning] Failed setting winmm timeBeginPeriod(1): {e}")

# Strict Allowlist of Non-Sensitive Special Keys (All alphanumeric keys are REJECTED)
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
# H   : uint16 (x coordinate: 0..65535)
# H   : uint16 (y coordinate: 0..65535)
# b   : int8   (dx scroll step: -128..127)
# b   : int8   (dy scroll step: -128..127)
# B   : uint8  (event_type: 0=move, 1=click_down, 2=click_up, 3=scroll, 4=key_down, 5=key_up)
# H   : uint16 (target_id: mapped mouse button ID or special key ID)
STRUCT_FMT = "<QHHbbBH"
RECORD_SIZE = struct.calcsize(STRUCT_FMT)  # Exactly 20 bytes

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

# Track current mouse position for key events
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
    # All character keys (A-Z, 0-9, symbols, passwords) are explicitly ignored!
    return None

def record_event(x: int, y: int, dx: int, dy: int, event_type: int, target_name: str = "none"):
    global ram_binary_buffer, start_perf_ns, start_time_epoch, end_time_epoch, event_count, last_mouse_x, last_mouse_y

    now_ns = time.perf_counter_ns()
    now_epoch = time.time()

    if start_perf_ns is None:
        start_perf_ns = now_ns
        start_time_epoch = now_epoch
    end_time_epoch = now_epoch

    last_mouse_x = int(x)
    last_mouse_y = int(y)

    dt_us = int((now_ns - start_perf_ns) // 1000)
    
    clamp_x = max(0, min(65535, int(x)))
    clamp_y = max(0, min(65535, int(y)))
    clamp_dx = max(-128, min(127, int(dx)))
    clamp_dy = max(-128, min(127, int(dy)))
    t_id = get_target_id(target_name)

    packed = struct.pack(
        STRUCT_FMT,
        dt_us,
        clamp_x,
        clamp_y,
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
    """Background thread that snapshots binary RAM buffer to temp folder every 300s."""
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

def unpack_bin_to_csv(bin_filepath: str, csv_filepath: str, base_epoch: float = None):
    """Unpacks a 20-byte .bin dataset file into formatted .csv."""
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
        if base_epoch is not None:
            f_csv.write("timestamp_epoch,rel_time_ms,x,y,scroll_x,scroll_y,scroll_dx,scroll_dy,event_type,target_item\n")
        else:
            f_csv.write("rel_time_ms,x,y,scroll_x,scroll_y,scroll_dx,scroll_dy,event_type,target_item\n")

        for offset in range(0, len(data) - RECORD_SIZE + 1, RECORD_SIZE):
            dt_us, x, y, dx, dy, evt_type, t_id = struct.unpack_from(STRUCT_FMT, data, offset)
            
            rel_ms = dt_us / 1000.0
            cumulative_sc_x += dx
            cumulative_sc_y += dy

            evt_str = event_type_names.get(evt_type, "unknown")
            target_str = ID_TO_NAME.get(t_id, f"item_{t_id}")

            if base_epoch is not None:
                abs_epoch = base_epoch + (rel_ms / 1000.0)
                f_csv.write(f"{abs_epoch:.6f},{rel_ms:.3f},{x},{y},{cumulative_sc_x:.1f},{cumulative_sc_y:.1f},{dx},{dy},{evt_str},{target_str}\n")
            else:
                f_csv.write(f"{rel_ms:.3f},{x},{y},{cumulative_sc_x:.1f},{cumulative_sc_y:.1f},{dx},{dy},{evt_str},{target_str}\n")

    print(f"[Converter] Successfully converted {total_records:,} records -> [file:///{csv_filepath.replace(os.sep, '/')}]")

def recover_autosaves():
    """Scans temp/ and datasets/ for un-converted .bin files and exports them to .csv."""
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    
    bin_files = glob.glob(os.path.join(temp_dir, "*.bin")) + glob.glob(os.path.join(datasets_dir, "*.bin"))
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
    avg_hz = (event_count / duration) if duration > 0 else 0.0

    output_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(output_dir, exist_ok=True)

    filename_base = f"input_log_{start_dt}_to_{end_dt}"
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    bin_path = os.path.join(output_dir, f"{filename_base}.bin")

    print(f"\n" + "=" * 65)
    print(f"[Recorder Engine v3.0] Flushing RAM binary buffer...")
    print(f" -> Total Events Recorded: {event_count:,}")
    print(f" -> Compact RAM Buffer:   {len(buffer_copy) / 1024:.2f} KB (20 bytes/event)")
    print(f" -> Recording Duration:   {duration:.2f} seconds")
    print(f" -> Average Polling Rate: {avg_hz:.1f} Hz")
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
    print(" PRIVACY-FIRST INPUT TELEMETRY RECORDER v3.0 (RAM BINARY BUFFER)")
    print("=" * 65)
    print(" PRIVACY PROMISE: ZERO logging of text, letters, digits, or passwords.")
    print(" ALLOWED KEYS: Ctrl, Alt, Win, Shift, Tab, Caps, Esc, Enter, Backspace,")
    print("               Delete, Arrows, F1-F12, PageUp/Down, Home, End, System keys.")
    print(" - Max Polling Speed: 1ms Win32 Timer (timeBeginPeriod)")
    print(" - Precision Timestamp: 64-bit uint64 Microseconds (0.001ms precision)")
    print(" - Compact RAM Buffer: 20 Bytes/Event")
    print(" - Periodic Autosave: Dumps to temp/ every 300s")
    print(" - Controls: Press Ctrl+C at any time to export to CSV")
    print(" - Recovery: Run 'python mouse_recorder.py --recover' to convert temp files\n")

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
