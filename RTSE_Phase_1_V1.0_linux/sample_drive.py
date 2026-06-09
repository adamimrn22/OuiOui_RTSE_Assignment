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

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'display_front_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 1.0
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

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
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time

            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock

    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False

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
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

# Tunable AI constants
LATERAL_SPEED_PX_S     = 400.0  # max car lateral movement speed in pixels/second
FORWARD_SPEED_DEFAULT  = 300.0  # initial forward speed estimate in pixels/second
FORWARD_SPEED_CAP      = 500.0  # optical flow estimate is capped here to keep reachability sane
LOOKAHEAD_SECS         = 1.2    # seconds of travel defining the actionable sensing window
MIN_LOOKAHEAD_PX       = 200    # floor: always look at least this far ahead (pixels)
MAX_LOOKAHEAD_HARD_CAP = 500    # ceiling: never look further than this (pixels)
LANE_COUNT             = 5      # number of equal-width lanes across the road
BEHIND_MARGIN_PX       = 40     # px behind car_y still treated as a live threat
COLLISION_RADIUS_PX    = 100    # horizontal px corridor: tokens within this of car_x = direct hit
EVADE_SAFETY_RATIO     = 1.3    # escape side must have danger this many × farther than front threat
STEER_HOLD_FRAMES      = 10     # sustain a steering decision for this many cycles (~50ms at 200Hz)

# State persisted between processing_task() calls
_proc_state = {
    'prev_gray':        None,
    'prev_time':        None,
    'fwd_speed_px_s':   FORWARD_SPEED_DEFAULT,
    'held_steering':    0.0,   # last committed non-zero steering decision
    'hold_frames_left': 0,     # frames remaining to sustain held_steering
}

def time_to_intercept(dist_y, forward_speed_px_s):
    if forward_speed_px_s <= 0:
        return float('inf')
    return dist_y / forward_speed_px_s

def is_reachable(dist_x, dist_y, forward_speed_px_s, lateral_speed_px_s):
    t = time_to_intercept(dist_y, forward_speed_px_s)
    if t == float('inf'):
        return False
    return abs(dist_x) / lateral_speed_px_s <= t

def read_single_camera(sock, data_key):
    if sock is None:
        return

    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length = int.from_bytes(length_bytes, 'little')
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
            image_length = int.from_bytes(length_bytes, 'little')
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
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame

    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, 'latest_back_frame')

def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    frame = front_frame.copy()
    h, w = frame.shape[:2]
    roi_top = int(h * 0.25)

    car_x = w // 2
    car_y = int(h * 0.85)

    # --- Forward speed via Lucas-Kanade optical flow ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now  = time.time()
    fwd_speed = _proc_state['fwd_speed_px_s']

    if _proc_state['prev_gray'] is not None and _proc_state['prev_time'] is not None:
        dt = now - _proc_state['prev_time']
        if 0 < dt < 0.5:
            road_y1, road_y2 = int(h * 0.65), int(h * 0.85)
            road_x1, road_x2 = int(w * 0.30), int(w * 0.70)
            road_region = _proc_state['prev_gray'][road_y1:road_y2, road_x1:road_x2]

            corners = cv2.goodFeaturesToTrack(
                road_region, maxCorners=30, qualityLevel=0.2, minDistance=7,
            )
            if corners is not None and len(corners) >= 3:
                corners_full = corners + np.array([[[road_x1, road_y1]]], dtype=np.float32)
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    _proc_state['prev_gray'], gray, corners_full, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                )
                if next_pts is not None and status is not None:
                    good_dy = [
                        next_pts[i][0][1] - corners_full[i][0][1]
                        for i, st in enumerate(status) if st[0] == 1
                    ]
                    if good_dy:
                        median_dy = float(np.median(good_dy))
                        if median_dy > 0:
                            raw_speed = median_dy / dt
                            fwd_speed = 0.3 * raw_speed + 0.7 * fwd_speed
                            fwd_speed = min(fwd_speed, FORWARD_SPEED_CAP)

    _proc_state['prev_gray']      = gray
    _proc_state['prev_time']      = now
    _proc_state['fwd_speed_px_s'] = fwd_speed

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green: strict H=52-85 so no yellow-green coin gets mistaken for collectible
    green_mask = cv2.inRange(hsv, np.array([52, 100, 100]), np.array([85, 255, 255]))
    green_mask[:roi_top, :] = 0

    # Red: hue wraps at 0/180
    red_lo   = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([15,  255, 255]))
    red_hi   = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_lo, red_hi)
    red_mask[:roi_top, :] = 0

    # Yellow/orange: H=15-52 catches amber through yellow-green so anything
    # that is NOT strict green is treated as danger
    yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([52, 255, 255]))
    yellow_mask[:roi_top, :] = 0

    def get_contours(mask, min_area=80):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) < 4.0:
                result.append(c)
        return result

    green_coins   = get_contours(green_mask,  min_area=80)
    red_tokens    = get_contours(red_mask,    min_area=100)
    yellow_tokens = get_contours(yellow_mask, min_area=100)

    def centroid(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    def dist_car(px, py):
        return np.sqrt((px - car_x) ** 2 + (py - car_y) ** 2)

    def safe_dist(c):
        pt = centroid(c)
        return dist_car(*pt) if pt else float('inf')

    steering   = 0.0
    action     = 'HOLD'
    all_danger = red_tokens + yellow_tokens

    lookahead = int(np.clip(fwd_speed * LOOKAHEAD_SECS,
                            MIN_LOOKAHEAD_PX, MAX_LOOKAHEAD_HARD_CAP))

    # -----------------------------------------------------------------
    # Detection
    #
    # DIRECT THREAT: token within COLLISION_RADIUS_PX of car_x horizontally.
    #   = car will hit it if going straight.  Triggers avoidance.
    #
    # LEFT / RIGHT side: token is outside the corridor on that side.
    #   = used to decide if the escape route is clear.
    #
    # Using |ox - car_x| instead of zone_of() means the centre trigger
    # is a narrow corridor (200px wide for COLLISION_RADIUS_PX=100),
    # not the full centre third of the screen.  Tokens that are offset
    # but will naturally pass the car on the side are ignored.
    # -----------------------------------------------------------------
    direct_d = float('inf')
    left_d   = float('inf')
    right_d  = float('inf')

    for dc in all_danger:
        pt = centroid(dc)
        if pt is None: continue
        ox, oy = pt
        if car_y - oy < -BEHIND_MARGIN_PX: continue
        d = dist_car(ox, oy)
        if d > lookahead: continue
        if abs(ox - car_x) < COLLISION_RADIUS_PX:
            direct_d = min(direct_d, d)
        elif ox < car_x:
            left_d = min(left_d, d)
        else:
            right_d = min(right_d, d)

    ctr_greens   = []
    left_greens  = []
    right_greens = []

    for gc in green_coins:
        pt = centroid(gc)
        if pt is None: continue
        gx, gy = pt
        if car_y - gy < -BEHIND_MARGIN_PX: continue
        if dist_car(gx, gy) > lookahead: continue
        if abs(gx - car_x) < COLLISION_RADIUS_PX:
            ctr_greens.append(gc)
        elif gx < car_x:
            left_greens.append(gc)
        else:
            right_greens.append(gc)

    has_threat   = direct_d < float('inf')
    # Escape side usable when its danger is far enough vs the front threat
    right_usable = (not has_threat) or (right_d > direct_d * EVADE_SAFETY_RATIO)
    left_usable  = (not has_threat) or (left_d  > direct_d * EVADE_SAFETY_RATIO)

    def steer_to(c):
        pt = centroid(c)
        if not pt: return 0.0
        gx, gy = pt
        d = dist_car(gx, gy)
        return float(np.clip(((gx - car_x) / d) * 2.5, -1.0, 1.0)) if d > 0 else 0.0

    # -----------------------------------------------------------------
    # Decision cascade
    #
    # 1. Direct threat → AVOID (full ±1.0 steering)
    #    a. Both sides usable → prefer side with green, else right
    #    b. Only right usable → RIGHT
    #    c. Only left usable  → LEFT
    #    d. Both blocked      → side with green, else less dangerous side
    #
    # 2. No threat → collect green
    #    a. Green in corridor (straight ahead) → steer toward it
    #    b. Green right (right usable)         → steer right
    #    c. Green left  (left usable)          → steer left
    #    d. Nothing                            → HOLD
    # -----------------------------------------------------------------
    if has_threat:
        if right_usable and left_usable:
            rgt_sc = right_d + (300 if right_greens else 0)
            lft_sc = left_d  + (300 if left_greens  else 0)
            steering = 1.0 if rgt_sc >= lft_sc else -1.0
            action   = f'AVOID {"RIGHT" if steering > 0 else "LEFT"}  d={int(direct_d)}px'
        elif right_usable:
            steering = 1.0
            action   = f'AVOID RIGHT  d={int(direct_d)}px  r={int(right_d)}px'
        elif left_usable:
            steering = -1.0
            action   = f'AVOID LEFT  d={int(direct_d)}px  l={int(left_d)}px'
        else:
            rgt_sc = right_d + (500 if right_greens else 0)
            lft_sc = left_d  + (500 if left_greens  else 0)
            steering = 1.0 if rgt_sc >= lft_sc else -1.0
            action   = f'AVOID FORCE {"R" if steering > 0 else "L"} BLOCKED d={int(direct_d)}px'

    elif ctr_greens:
        target   = min(ctr_greens, key=safe_dist)
        steering = steer_to(target)
        pt       = centroid(target)
        action   = f'GREEN AHEAD  d={int(dist_car(*pt)) if pt else 0}px'

    elif right_greens and right_usable:
        target   = min(right_greens, key=safe_dist)
        steering = max(steer_to(target), 0.45)   # minimum magnitude so car commits to the lane change
        pt       = centroid(target)
        action   = f'GREEN RIGHT  d={int(dist_car(*pt)) if pt else 0}px'

    elif left_greens and left_usable:
        target   = min(left_greens, key=safe_dist)
        steering = min(steer_to(target), -0.45)  # minimum magnitude leftward
        pt       = centroid(target)
        action   = f'GREEN LEFT  d={int(dist_car(*pt)) if pt else 0}px'

    # Steering inertia: once a direction is committed, lock it for STEER_HOLD_FRAMES
    # cycles.  A new signal in the SAME direction refreshes the counter; a signal in
    # the OPPOSITE direction is ignored unless it comes from direct-collision avoidance
    # (|steering| == 1.0 and has_threat) — that's an emergency override.
    if steering != 0.0:
        same_dir = (np.sign(steering) == np.sign(_proc_state['held_steering']))
        emergency = (abs(steering) >= 1.0 and has_threat)
        if same_dir or emergency or _proc_state['hold_frames_left'] == 0:
            _proc_state['held_steering']    = steering
            _proc_state['hold_frames_left'] = STEER_HOLD_FRAMES
        else:
            # mid-hold direction flip from non-emergency: keep current direction
            steering = _proc_state['held_steering']
            _proc_state['hold_frames_left'] -= 1
    elif _proc_state['hold_frames_left'] > 0:
        steering = _proc_state['held_steering']
        _proc_state['hold_frames_left'] -= 1

    steering = float(np.clip(steering, -1.0, 1.0))

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    corr_l = car_x - COLLISION_RADIUS_PX
    corr_r = car_x + COLLISION_RADIUS_PX
    cv2.line(frame, (corr_l, roi_top), (corr_l, car_y), (0, 100, 200), 1)
    cv2.line(frame, (corr_r, roi_top), (corr_r, car_y), (0, 100, 200), 1)
    horizon_y = max(roi_top, car_y - lookahead)
    cv2.line(frame, (0, horizon_y), (w, horizon_y), (60, 60, 60), 1)

    for lbl, dx, dd, gr in [
        ('L', corr_l // 2,        left_d,   left_greens),
        ('C', car_x,              direct_d, ctr_greens),
        ('R', (corr_r + w) // 2, right_d,  right_greens),
    ]:
        has_d = dd < float('inf')
        dot_col = (0, 0, 200) if has_d else (0, 200, 0) if gr else (80, 80, 80)
        cv2.circle(frame, (dx, roi_top + 12), 8, dot_col, -1)
        if has_d:
            cv2.putText(frame, f"{int(dd)}",
                        (dx - 14, roi_top + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 160, 255), 1)

    for gc in ctr_greens + left_greens + right_greens:
        bx, by, bw2, bh = cv2.boundingRect(gc)
        cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh), (0, 255, 0), 2)

    for dc in red_tokens:
        pt = centroid(dc)
        if pt and car_y - pt[1] >= -BEHIND_MARGIN_PX and dist_car(*pt) <= lookahead:
            bx, by, bw2, bh = cv2.boundingRect(dc)
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh), (0, 0, 255), 2)
    for dc in yellow_tokens:
        pt = centroid(dc)
        if pt and car_y - pt[1] >= -BEHIND_MARGIN_PX and dist_car(*pt) <= lookahead:
            bx, by, bw2, bh = cv2.boundingRect(dc)
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh), (0, 200, 255), 2)

    cv2.circle(frame, (car_x, car_y), 8, (0, 0, 255), -1)
    if abs(steering) > 0.05:
        cv2.arrowedLine(frame, (car_x, car_y),
                        (car_x + int(steering * (w // 4)), car_y),
                        (255, 255, 0), 3, tipLength=0.3)

    if   steering < -0.6: steer_label = "HARD LEFT"
    elif steering < -0.2: steer_label = "Left"
    elif steering >  0.6: steer_label = "HARD RIGHT"
    elif steering >  0.2: steer_label = "Right"
    else:                 steer_label = "Center"

    def dfmt(d, g):
        return f"{int(d)}px" if d < float('inf') else ('G' if g else '-')

    cv2.putText(frame, f"Action: {action}",
                (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame,
                f"L:{dfmt(left_d,left_greens)}  "
                f"C:{dfmt(direct_d,ctr_greens)}  "
                f"R:{dfmt(right_d,right_greens)}  "
                f"look={lookahead}px  spd={int(fwd_speed)}px/s",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(frame, f"[{steer_label}] {steering:+.2f}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    with data_lock:
        shared_data['steering_input']      = steering
        shared_data['acceleration_input']  = 1.0
        shared_data['display_front_frame'] = frame

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    with data_lock:
        steering_input = shared_data.get('steering_input', 0.0)
        acceleration_input = shared_data.get('acceleration_input', 1.0)

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        keyboard.press('w')
        print("W key held for speed boost.")
    except Exception:
        pass

    try:
        while is_running:
            time.sleep(0.033)
            with data_lock:
                front = shared_data.get('display_front_frame')
                if front is None:
                    front = shared_data.get('latest_front_frame')
                back = shared_data.get('latest_back_frame')
            if front is not None:
                cv2.imshow("Front Camera", cv2.resize(front, (640, 480)))
            if back is not None:
                cv2.imshow("Back Camera", cv2.resize(back, (640, 480)))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False
        try:
            keyboard.release('w')
        except Exception:
            pass

    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
