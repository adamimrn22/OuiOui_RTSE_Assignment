import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# ---------------------------------------------------------
# RTSE Improvement Constants
# ---------------------------------------------------------
# Proposal 5 — time-to-impact threshold for green token commit
ESTIMATED_FORWARD_SPEED_PX_S = 300.0  # pixels/second (tune based on game speed)
TIME_TO_IMPACT_COMMIT_S      = 0.3    # below this, hard-commit steering toward green

# Proposal 3 — proximity urgency thresholds (token contour area)
URGENCY_CRITICAL_AREA = 3000  # very close token
URGENCY_HIGH_AREA     = 1000  # medium distance token

# Challenge 1 — Low Light
LOW_BRIGHTNESS_THRESHOLD = 60  # mean pixel value below this = light is off

# Challenge 3 — Police Car
POLICE_DEADLINE          = 10.0  # seconds to collect a red token before 50% speed penalty
POLICE_URGENT_THRESHOLD  = 3.0   # seconds remaining when we enter full-commit mode
POLICE_MIN_AREA          = 2000  # min contour area to qualify as a police car (not a token)
POLICE_ABSENT_GRACE      = 1.5   # seconds police car can be absent before mode deactivates

# ---------------------------------------------------------
# Shared Resources
# ---------------------------------------------------------
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame':  None,
    'steering_input':     0.0,
    'acceleration_input': 0.0,

    # Written by detection_task, read by decision_task
    # Proposal 2 — decoupled detection results
    'detected_token': None,   # 'green', 'red', 'yellow', None
    'token_x':        0,      # centroid x of detected token (pixels)
    'token_y':        0,      # centroid y of detected token (pixels, absolute in frame)
    'token_area':     0,      # contour area of detected token
    'token_urgency':  'LOW',  # 'CRITICAL', 'HIGH', 'LOW'  — Proposal 3
    'frame_w':        640,    # updated by detection_task each cycle
    'frame_h':        480,

    # Event / mode state
    'last_token':   None,
    'event_active': None,     # 'low_brightness', None

    # Challenge 1 — Low Light state
    # Written by brightness_task, read by detection_task and decision_task
    'low_light_active':     False,  # True while the light is off
    'low_light_start_time': 0.0,    # timestamp of when the light went off

    # Dodge / lane state
    'dodge_direction': 0.0,
    'dodge_until':     0.0,
    'lane_offset':     0.0,

    # Challenge 3 — Police Car state
    # Written by detection_task + police_watchdog_task, read by decision_task
    'police_active':     False,  # True while police car challenge is running
    'police_start_time': 0.0,    # timestamp when police was first detected
    'police_urgent':     False,  # True when < POLICE_URGENT_THRESHOLD seconds remain
    'police_bbox':       None,   # (x, y, w, h) of police car — used as no-go zone
    'police_last_seen':  0.0,    # timestamp of most recent positive police detection
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
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception:
            pass

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
                # Proposal 4 — lock held only for the write, not during decode
                with data_lock:
                    shared_data[data_key] = frame

                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)

    except Exception:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

# ---------------------------------------------------------
# Challenge 1 — Brightness Task (HIGH priority, 5 ms period)
#
# Runs independently of the heavier detection_task so that the
# light-off event is caught immediately — it only reads one frame
# and computes a mean pixel value, which is extremely fast.
#
# RTSE rationale:
#   - Dedicated task ensures brightness is checked every 5ms regardless
#     of how long token detection takes (which runs at 10ms/MEDIUM).
#   - HIGH priority means it preempts MEDIUM tasks if they overrun.
#   - Mode change (low_light_active) is the single flag that all other
#     tasks gate on — clean separation of detection and response.
# ---------------------------------------------------------
def brightness_task():
    # Snapshot only — lock released before any computation
    with data_lock:
        frame = shared_data['latest_front_frame']

    if frame is None:
        return

    # Compute brightness outside the lock (fast operation)
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    now        = time.time()

    with data_lock:
        currently_active = shared_data['low_light_active']

    if brightness < LOW_BRIGHTNESS_THRESHOLD:
        if not currently_active:
            # Light just went off — start the event
            print(f"[LOW LIGHT] Detected! Brightness={brightness:.1f}. Sending acceleration = -1.0 to recover.")
            with data_lock:
                shared_data['low_light_active']     = True
                shared_data['low_light_start_time'] = now
                shared_data['event_active']         = 'low_brightness'
    else:
        if currently_active:
            # Light has recovered — end the event
            with data_lock:
                elapsed = now - shared_data['low_light_start_time']
            print(f"[LOW LIGHT] Recovered after {elapsed:.2f}s. Resuming normal mode.")
            with data_lock:
                shared_data['low_light_active'] = False
                shared_data['event_active']     = None


# ---------------------------------------------------------
# Challenge 3 — Police Car Helper
#
# Scans the HSV frame for a large blue-coloured object.
# Blue (H 100–130) is not used by any regular token, making it a
# clean discriminator.  Contour must exceed POLICE_MIN_AREA to
# rule out small blue artefacts.
# Called inside detection_task (already has the HSV frame ready).
# ---------------------------------------------------------
def detect_police_car(hsv, frame_h, frame_w):
    blue_mask = cv2.inRange(
        hsv,
        np.array([100, 80, 80]),
        np.array([130, 255, 255])
    )
    # Only search the forward portion of the frame
    roi_top = int(frame_h * 0.10)
    roi_bot = int(frame_h * 0.80)
    blue_mask[:roi_top, :] = 0
    blue_mask[roi_bot:,  :] = 0

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < POLICE_MIN_AREA:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    return (x, y, bw, bh)


# ---------------------------------------------------------
# Challenge 3 — Police Watchdog Task (LOW priority, 100 ms period)
#
# Sole responsibility: track the 10-second deadline and flip the
# police_urgent flag when time is running short.
#
# RTSE rationale:
#   - Deadline tracking is decoupled from heavy detection work so it
#     fires reliably regardless of how long detection_task takes.
#   - LOW priority is appropriate — it only updates a flag, and
#     100 ms resolution is more than sufficient for a 10 s deadline.
#   - When deadline expires, resets all police state so the system
#     cleanly returns to normal mode.
# ---------------------------------------------------------
def police_watchdog_task():
    with data_lock:
        police_active     = shared_data['police_active']
        police_start_time = shared_data['police_start_time']

    if not police_active:
        return

    elapsed   = time.time() - police_start_time
    remaining = POLICE_DEADLINE - elapsed

    if remaining <= 0:
        # Deadline expired — game applies the 50% speed penalty
        with data_lock:
            shared_data['police_active']  = False
            shared_data['police_urgent']  = False
            shared_data['police_bbox']    = None
            shared_data['event_active']   = None
        print("[POLICE] Deadline expired — 50% speed penalty applied by game.")
        return

    urgent = remaining <= POLICE_URGENT_THRESHOLD
    with data_lock:
        shared_data['police_urgent'] = urgent

    if urgent:
        print(f"[POLICE] URGENT — {remaining:.1f}s left! Committing to RED token only!")
    else:
        print(f"[POLICE] Active — {remaining:.1f}s remaining to collect RED token.")


# ---------------------------------------------------------
# Proposal 2 — Detection Task (MEDIUM priority, 10 ms period)
#
# Sole responsibility: read the front frame, run all OpenCV work,
# classify the token and its urgency, then write results to
# shared_data in one brief lock.  No steering decisions here.
# ---------------------------------------------------------
def detection_task():
    # --- Proposal 4: snapshot only, lock released before any cv2 work ---
    with data_lock:
        frame            = shared_data['latest_front_frame']
        low_light_active = shared_data['low_light_active']

    if frame is None:
        return

    # Challenge 1 — skip all token detection while light is off.
    # Tokens are invisible so running cv2 would waste CPU and produce
    # garbage results.  brightness_task handles the event independently.
    if low_light_active:
        with data_lock:
            shared_data['detected_token'] = None
            shared_data['token_urgency']  = 'LOW'
        return

    # All heavy computation happens outside the lock
    h, w = frame.shape[:2]

    # HSV conversion for token detection
    hsv           = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask_high = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))

    token_colors = {
        'green':  (np.array([40,  80,  80]),  np.array([80,  255, 255])),
        'red':    (np.array([0,  120,  70]),   np.array([10,  255, 255])),
        'yellow': (np.array([20, 100, 100]),   np.array([35,  255, 255])),
    }

    roi_top = int(h * 0.25)
    roi_bot = int(h * 0.75)

    detected   = None
    token_x    = w // 2
    token_y    = roi_top
    token_area = 0

    for color_name, (lower, upper) in token_colors.items():
        mask = cv2.inRange(hsv[roi_top:roi_bot, :], lower, upper)
        if color_name == 'red':
            mask = cv2.bitwise_or(mask, red_mask_high[roi_top:roi_bot, :])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 300:
                x, y, cw, ch = cv2.boundingRect(largest)
                token_area    = cv2.contourArea(largest)
                detected      = color_name
                token_x       = x + cw // 2
                # Absolute y in full frame (roi_top offset added back)
                token_y       = roi_top + y + ch // 2
                break

    # --- Proposal 3: classify urgency from token area ---
    if detected is not None:
        if token_area > URGENCY_CRITICAL_AREA:
            urgency = 'CRITICAL'
        elif token_area > URGENCY_HIGH_AREA:
            urgency = 'HIGH'
        else:
            urgency = 'LOW'
    else:
        urgency = 'LOW'

    # --- Challenge 3: police car detection (reuses the HSV frame already computed) ---
    police_bbox = detect_police_car(hsv, h, w)
    now         = time.time()

    with data_lock:
        police_active = shared_data['police_active']

    if police_bbox is not None:
        if not police_active:
            print("[POLICE] Police car detected! 10 seconds to collect a RED token!")
        with data_lock:
            shared_data['police_active']     = True
            shared_data['police_start_time'] = shared_data['police_start_time'] if police_active else now
            shared_data['police_bbox']       = police_bbox
            shared_data['police_last_seen']  = now
            shared_data['event_active']      = 'police'
    else:
        with data_lock:
            last_seen = shared_data['police_last_seen']

        if police_active and (now - last_seen) > POLICE_ABSENT_GRACE:
            print("[POLICE] Police car gone — returning to normal mode.")
            with data_lock:
                shared_data['police_active'] = False
                shared_data['police_urgent'] = False
                shared_data['police_bbox']   = None
                shared_data['event_active']  = None
        else:
            with data_lock:
                shared_data['police_bbox'] = None  # clear bbox each frame if not visible

    # --- Proposal 4: single brief write-back lock ---
    with data_lock:
        # Write detection results for decision_task to consume
        shared_data['detected_token'] = detected
        shared_data['token_x']        = token_x
        shared_data['token_y']        = token_y
        shared_data['token_area']     = token_area
        shared_data['token_urgency']  = urgency
        shared_data['frame_w']        = w
        shared_data['frame_h']        = h
        shared_data['last_token']     = detected


# ---------------------------------------------------------
# Proposal 2 — Decision Task (HIGH priority, 5 ms period)
#
# Sole responsibility: read detection results and compute
# steering/acceleration.  No cv2 work — runs fast and clean.
# ---------------------------------------------------------
def decision_task():
    # --- Proposal 4: snapshot all inputs in one brief lock ---
    with data_lock:
        low_light_active = shared_data['low_light_active']
        detected         = shared_data['detected_token']
        token_x          = shared_data['token_x']
        token_y          = shared_data['token_y']
        token_area       = shared_data['token_area']
        urgency          = shared_data['token_urgency']
        dodge_direction  = shared_data['dodge_direction']
        dodge_until      = shared_data['dodge_until']
        lane_offset      = shared_data['lane_offset']
        frame_w          = shared_data['frame_w']
        frame_h          = shared_data['frame_h']
        police_active    = shared_data['police_active']
        police_urgent    = shared_data['police_urgent']
        police_bbox      = shared_data['police_bbox']

    # ------------------------------------------------------------------
    # Challenge 1 — Low Light Mode
    #
    # When active, ALL other control logic is suppressed.
    # Send acceleration = -1.0 and steer = 0.0 to trigger light recovery.
    # Steering is zeroed so the car holds its lane while braking.
    # ------------------------------------------------------------------
    if low_light_active:
        with data_lock:
            shared_data['steering_input']     = 0.0
            shared_data['acceleration_input'] = -1.0
        return   # skip all token logic — tokens are invisible anyway

    # All decision logic runs outside the lock
    mid_x = frame_w // 2
    car_y = int(frame_h * 0.85)
    steer = 0.0
    accel = 1.0
    now   = time.time()

    # ------------------------------------------------------------------
    # Challenge 3 — Police Mode
    #
    # Goal inversion: red tokens switch from "dodge" to "collect".
    # Two hard constraints run simultaneously:
    #   1. Never steer into the police car bounding box (game over risk).
    #   2. Seek the nearest red token within the 10s deadline.
    # When urgent (< 3s left), green tokens are ignored entirely.
    # ------------------------------------------------------------------
    if police_active:
        # Work out which side the police car occupies so we can avoid it
        police_side = 0   # -1 = left, 0 = not a direct threat, 1 = right
        if police_bbox is not None:
            px, py, pbw, pbh = police_bbox
            pcx = px + pbw // 2
            # Only a collision threat if it is laterally close and ahead
            if abs(pcx - mid_x) < (frame_w * 0.25) and (car_y - (py + pbh)) < 200:
                police_side = -1 if pcx >= mid_x else 1

        if detected == 'red':
            # Steer TOWARD the red token (opposite of normal behaviour)
            offset = token_x - mid_x
            if abs(offset) > 40:
                steer = max(-1.0, min(1.0, offset / (mid_x * 0.6)))
            else:
                steer = 0.0
            # If police car is on the same side as the token, approach cautiously
            if police_side != 0 and np.sign(steer) == np.sign(police_side):
                steer *= 0.5
            lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.1))

        elif detected == 'green' and not police_urgent:
            # Green is fine when time is not critical — but avoid police car side
            offset = token_x - mid_x
            steer  = max(-1.0, min(1.0, offset / (mid_x * 0.6))) if abs(offset) > 40 else 0.0
            if police_side != 0 and np.sign(steer) == np.sign(police_side):
                steer = 0.0   # skip this green — too risky
            lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.1))

        else:
            # No red visible (or urgent + no red) — scan or drift away from police
            if police_side != 0:
                steer = float(police_side) * -0.4   # drift away from police car
            elif police_urgent:
                steer = 0.3 if (int(now * 2) % 2 == 0) else -0.3  # gentle scan
            else:
                steer = -0.3 if lane_offset > 0.15 else (0.3 if lane_offset < -0.15 else 0.0)
            lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.05))

        with data_lock:
            shared_data['lane_offset']        = lane_offset
            shared_data['steering_input']     = steer
            shared_data['acceleration_input'] = accel
        return   # skip normal token logic entirely

    # ------------------------------------------------------------------
    # GREEN token → steer TOWARD it to collect
    # ------------------------------------------------------------------
    if detected == 'green':
        offset = token_x - mid_x

        # --- Proposal 5: time-to-impact based commit ---
        dist_y          = max(car_y - token_y, 1)
        time_to_impact  = dist_y / ESTIMATED_FORWARD_SPEED_PX_S

        if time_to_impact < TIME_TO_IMPACT_COMMIT_S:
            # Very close — hard commit regardless of lateral offset
            steer = 1.0 if offset > 0 else -1.0

        # --- Proposal 3: urgency-scaled steering ---
        elif urgency == 'CRITICAL':
            steer = 1.0 if offset > 0 else -1.0         # full commit
        elif urgency == 'HIGH':
            steer = max(-1.0, min(1.0, offset / (mid_x * 0.6))) * 0.7
        else:
            # LOW urgency — proportional steering as before
            if abs(offset) > 40:
                steer = max(-1.0, min(1.0, offset / (mid_x * 0.6)))
            else:
                steer = 0.0

        lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.1))
        new_dodge_direction = 0.0
        new_dodge_until     = 0.0
        new_lane_offset     = lane_offset
        new_steer           = steer
        new_accel           = accel

    # ------------------------------------------------------------------
    # RED / YELLOW token → dodge AWAY from it
    # ------------------------------------------------------------------
    elif detected in ('red', 'yellow'):
        if now < dodge_until and dodge_direction != 0.0:
            steer = dodge_direction
        else:
            # Choose dodge direction opposite to token side
            preferred = 1.0 if token_x < mid_x else -1.0

            # Boundary check
            if lane_offset >= 0.9 and preferred > 0:
                preferred = -1.0
            elif lane_offset <= -0.9 and preferred < 0:
                preferred = 1.0

            # --- Proposal 3: urgency-scaled hold duration ---
            if urgency == 'CRITICAL':
                hold = 0.40   # longest hold — very close, commit hard
            elif urgency == 'HIGH':
                hold = 0.22
            else:
                hold = 0.12   # far away, short nudge

            steer           = preferred
            dodge_direction = steer
            dodge_until     = now + hold
            lane_offset     = max(-1.0, min(1.0, lane_offset + steer * 0.5))

        new_dodge_direction = dodge_direction
        new_dodge_until     = dodge_until
        new_lane_offset     = lane_offset
        new_steer           = steer
        new_accel           = accel

    # ------------------------------------------------------------------
    # No token → hold active dodge, then re-centre gently
    # ------------------------------------------------------------------
    else:
        if now < dodge_until and dodge_direction != 0.0:
            steer = dodge_direction
        else:
            if lane_offset > 0.15:
                steer = -0.4
            elif lane_offset < -0.15:
                steer = 0.4
            else:
                steer       = 0.0
                lane_offset = 0.0

            lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.05))

        new_dodge_direction = dodge_direction
        new_dodge_until     = dodge_until
        new_lane_offset     = lane_offset
        new_steer           = steer
        new_accel           = accel

    # --- Proposal 4: single brief write-back lock ---
    with data_lock:
        shared_data['dodge_direction']    = new_dodge_direction
        shared_data['dodge_until']        = new_dodge_until
        shared_data['lane_offset']        = new_lane_offset
        shared_data['steering_input']     = new_steer
        shared_data['acceleration_input'] = new_accel


# ---------------------------------------------------------
# Send Controls Task
# ---------------------------------------------------------
def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    # Proposal 4 — lock held only for the read, not for the send
    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']
        low_light_active   = shared_data['low_light_active']
        police_active      = shared_data['police_active']

    # Force full throttle in normal mode and police mode.
    # In low light, decision_task sets -1.0 and we must not override it.
    if not low_light_active:
        acceleration_input = 1.0

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main — Proposal 1: Rate Monotonic Scheduling
#
# Shorter period = higher priority (RMS rule).
#   SendControls  : 2 ms  → most critical, always runs first
#   ReadFrontCamera: 5 ms → feeds detection pipeline
#   decision_task  : 5 ms → fast (no cv2), must stay responsive
#   ReadBackCamera : 10 ms → less urgent
#   detection_task : 10 ms → heaviest task, lower frequency is fine
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks ---\n")
    print("RMS Schedule:")
    print("  SendControls   : 2ms   | HIGH   — always sends latest control")
    print("  ReadFrontCamera: 5ms   | HIGH   — feeds brightness + detection pipeline")
    print("  BrightnessTask : 5ms   | HIGH   — Challenge 1: hard real-time light detection")
    print("  DecisionTask   : 5ms   | HIGH   — fast steering logic, no cv2")
    print("  ReadBackCamera : 10ms  | MEDIUM — back camera feed")
    print("  DetectionTask  : 10ms  | MEDIUM — heavy cv2 token + police car detection")
    print("  PoliceWatchdog : 100ms | LOW    — Challenge 3: deadline countdown\n")

    # Proposal 1 — RMS-ordered task definitions
    t_send_controls   = RTTask("SendControls",    period=0.002, priority=TaskPriority.HIGH,   execute_func=send_controls_task)
    t_front_camera    = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
    t_brightness      = RTTask("BrightnessTask",  period=0.005, priority=TaskPriority.HIGH,   execute_func=brightness_task)
    t_decision        = RTTask("DecisionTask",    period=0.005, priority=TaskPriority.HIGH,   execute_func=decision_task)
    t_back_camera     = RTTask("ReadBackCamera",  period=0.010, priority=TaskPriority.MEDIUM, execute_func=read_back_camera_task)
    t_detection       = RTTask("DetectionTask",   period=0.010, priority=TaskPriority.MEDIUM, execute_func=detection_task)
    t_police_watchdog = RTTask("PoliceWatchdog",  period=0.100, priority=TaskPriority.LOW,    execute_func=police_watchdog_task)

    t_send_controls.start()
    t_front_camera.start()
    t_brightness.start()
    t_decision.start()
    t_back_camera.start()
    t_detection.start()
    t_police_watchdog.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    t_send_controls.join()
    t_front_camera.join()
    t_brightness.join()
    t_decision.join()
    t_back_camera.join()
    t_detection.join()
    t_police_watchdog.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
