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

def read_single_camera(sock, data_key):
    # Reads the latest frame from the camera socket and stores it in shared_data.
    # Display is intentionally NOT done here — cv2.imshow must be called from the
    # main thread on Linux (Qt backend requirement). The main loop handles display.
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
    # -----------------------------------------------------------------
    # Step 1: Object Detection
    # Car is fixed at bottom-center (perspective view).
    # Detect GREEN coins (collect) and non-green colored coins (avoid).
    # -----------------------------------------------------------------
    with data_lock:
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    frame = front_frame.copy()
    h, w = frame.shape[:2]
    roi_top = int(h * 0.25)  # ignore top 25% — score/UI overlay

    car_x = w // 2
    car_y = int(h * 0.85)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green coin mask (collect these)
    green_mask = cv2.inRange(hsv, np.array([40, 80, 100]), np.array([80, 255, 255]))
    green_mask[:roi_top, :] = 0

    # Obstacle mask: any highly-saturated color that is NOT green (avoid these)
    saturated = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    obstacle_mask = cv2.bitwise_and(saturated, cv2.bitwise_not(green_mask))
    obstacle_mask[:roi_top, :] = 0

    def get_contours(mask, min_area=80):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            # Reject highly elongated shapes (road borders/lane markings, not coins)
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) < 4.0:
                result.append(c)
        return result

    green_coins = get_contours(green_mask, min_area=80)
    obstacles   = get_contours(obstacle_mask, min_area=150)

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

    steering = 0.0
    avoiding = False

    # -----------------------------------------------------------------
    # Step 2 & 3: Obstacle avoidance — HIGHEST PRIORITY
    # Danger zone scales with sqrt(area): bigger object = closer/faster
    # approach = larger avoidance zone (speed-aware proxy).
    #
    # User rule:
    #   obstacle in FRONT or on LEFT  (dx <= 80)  -> steer RIGHT (+1.0)
    #   obstacle on RIGHT side only   (dx >  80)  -> steer LEFT  (-1.0)
    # -----------------------------------------------------------------
    for obs in sorted(obstacles, key=safe_dist):
        obs_pt = centroid(obs)
        if obs_pt is None:
            continue
        ox, oy = obs_pt
        obs_dist = dist_car(ox, oy)
        danger_radius = max(150, float(np.sqrt(cv2.contourArea(obs))) * 4)

        if obs_dist < danger_radius:
            dx_obs = ox - car_x
            steering = 1.0 if dx_obs <= 80 else -1.0   # front/left -> right, right-only -> left
            avoiding = True

            bx, by, bw, bh = cv2.boundingRect(obs)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            cv2.line(frame, (car_x, car_y), (ox, oy), (0, 0, 255), 2)
            cv2.putText(frame, f"AVOID {'RIGHT' if steering > 0 else 'LEFT'}  d={int(obs_dist)}px",
                        (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            break   # handle nearest threat only

    # -----------------------------------------------------------------
    # Step 2 & 3: Green coin targeting (only when no avoidance active)
    #
    # Steering angle = sin(angle to coin) * 2.5
    #   sin = dx / line_length  (naturally distance-aware)
    #   * 2.5 -> full lock at ~24 degrees off-centre
    # Far coin slightly off-centre -> gentle steer
    # Close coin to the side      -> hard steer
    # -----------------------------------------------------------------
    if not avoiding and green_coins:
        target = min(green_coins, key=safe_dist)
        pt = centroid(target)

        if pt:
            gx, gy = pt
            dx = gx - car_x
            d_coin = dist_car(gx, gy)

            if d_coin > 0:
                steering = float(np.clip((dx / d_coin) * 2.5, -1.0, 1.0))

            direction = ("RIGHT" if steering > 0.05
                         else "LEFT" if steering < -0.05
                         else "CENTER")

            bx, by, bw, bh = cv2.boundingRect(target)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(frame, (gx, gy), 5, (0, 255, 0), -1)
            cv2.line(frame, (car_x, car_y), (gx, gy), (0, 255, 255), 2)
            cv2.circle(frame, (car_x, car_y), 8, (0, 0, 255), -1)

            if abs(steering) > 0.05:
                cv2.arrowedLine(frame, (car_x, car_y),
                                (car_x + int(steering * (w // 2)), car_y),
                                (0, 0, 255), 3, tipLength=0.3)

            cv2.putText(frame, f"GREEN {direction} [{steering:+.2f}]  dist={int(d_coin)}px",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif not avoiding:
        cv2.putText(frame, "Scanning for GREEN...",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    with data_lock:
        shared_data['steering_input'] = steering
        shared_data['acceleration_input'] = 1.0   # always full speed forward
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

    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    # Hold W key down for any additional in-game speed boost the game accepts via keyboard
    try:
        keyboard.press('w')
        print("W key held for speed boost.")
    except Exception:
        pass

    try:
        # All cv2.imshow calls must happen on the main thread (Qt/Linux requirement).
        # ~30 fps display loop — does not affect the 200 Hz control tasks above.
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

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()

    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
