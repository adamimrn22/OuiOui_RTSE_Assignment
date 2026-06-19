"""
core.py — shared state, the real-time framework, networking and camera I/O.

Holds the global shared_data/data_lock, the DriveState enum, and the two
"do not change" blocks (RTTask framework + network setup), kept verbatim.
Other modules import shared_data / data_lock / DriveState from here.
"""
import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes

from config import *      # noqa: F401,F403  (CAMERA_HOST, ports, NUM_LANES, CENTER_LANE, ...)

# ---------------------------------------------------------
# Drive States — explicit priority state machine
# (highest priority first; resolved once per decision cycle)
# ---------------------------------------------------------
class DriveState:
    LOW_LIGHT       = 'LOW_LIGHT'        # 1 — restore the light, nothing else
    CHASING_EVASION = 'CHASING_EVASION'  # 2 — green car behind, survive
    POLICE          = 'POLICE'           # 3 — collect a red token in time
    COIN_AVOID      = 'COIN_AVOID'       # 4 — dangerous coin in our lane
    NORMAL          = 'NORMAL'           # 5 — cruise + collect green

# ---------------------------------------------------------
# Shared Resources
# ---------------------------------------------------------
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame':  None,
    'steering_input':     0.0,
    'acceleration_input': 0.0,

    # Written by detection_task, read by decision_task
    'tokens':         [],     # list of {color,x,y,w,h,area,lane}
    'frame_w':        640,
    'frame_h':        480,
    'blanked_lanes':  [False] * NUM_LANES,   # lanes blacked-out by the yellow attack

    # Event / mode state
    'event_active': None,

    # Challenge 1 — Low Light state
    'low_light_active':     False,
    'low_light_start_time': 0.0,

    # Lane / steering state
    'target_lane':  CENTER_LANE,
    'target_x':     0,
    'lookahead_y':  0,
    'lane_scores':  [0.0] * NUM_LANES,
    'commit_until': 0.0,
    'state':        DriveState.NORMAL,

    # Challenge 3 — Police Car state
    'police_active':     False,
    'police_start_time': 0.0,
    'police_urgent':     False,
    'police_bbox':       None,
    'police_bbox_time':  0.0,
    'police_last_seen':  0.0,

    # Challenge 2 — Chasing Car state
    'chasing_active':           False,
    'chasing_start_time':       0.0,
    'chasing_last_seen':        0.0,
    'chasing_appearance_count': 0,   # 1st appearance = 10s window, 2nd = 3s window
    'chasing_raw_detected':     False,
    'chasing_teal_area':        0.0,
    'chasing_teal_solidity':    0.0,
    'chasing_bbox':             None,
    'chasing_detected':         False,
    'chasing_mask_roi_top':     0,
    'chasing_mask_roi_left':    0,
    'chasing_mask_roi_right':   0,
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH   = 1
    MEDIUM = 2
    LOW    = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name         = name
        self.period       = period
        self.priority     = priority
        self.execute_func = execute_func
        self.daemon       = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                handle = ctypes.windll.kernel32.GetCurrentThread()
                if self.priority == TaskPriority.HIGH:
                    ctypes.windll.kernel32.SetThreadPriority(handle, 2)
                elif self.priority == TaskPriority.MEDIUM:
                    ctypes.windll.kernel32.SetThreadPriority(handle, 0)
                elif self.priority == TaskPriority.LOW:
                    ctypes.windll.kernel32.SetThreadPriority(handle, -2)
            else:
                # Linux/Posix: os.nice() affects the whole process.
                # Per-thread scheduling requires root/CAP_SYS_NICE or pthread extensions.
                # For this assignment, we gracefully bypass thread priority on Linux.
                pass
        except Exception as e:
            print(f"Note: Could not set thread priority ({e})")

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time  = time.time() - start_time
            sleep_time = self.period - exec_time
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock  = None
control_conn      = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    print("Connecting to Cameras...")
    front_connected = False
    back_connected  = False

    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass

        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass

        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")

    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Camera Read Tasks
# ---------------------------------------------------------
def read_single_camera(sock, window_name, data_key):
    if sock is None:
        return

    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length   = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet

        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes

        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break

            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length   = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes

        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Lock held only for the write, not during decode
                with data_lock:
                    shared_data[data_key] = frame

    except Exception:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')
