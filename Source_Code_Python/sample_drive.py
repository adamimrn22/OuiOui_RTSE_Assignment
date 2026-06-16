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
    'steering_input' : 0.0,
    'acceleration_input' : 0.0,
    # Token detection
    'last_token': None,
    'event_active': None,
    'corrupted_frame': False,
    'steering_cooldown': 0.0,
    # Dodge / lane state
    'dodge_direction': 0.0,
    'dodge_until': 0.0,
    'lane_offset': 0.0,
    # Challenge 1 — Low Light (one-shot guard)
    'low_light_active': False,  # True while brightness is below threshold
    'low_light_done':   False,  # True after the event resolves — never re-triggers
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

def read_single_camera(sock, window_name, data_key):
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
                
                #frame_resized = cv2.resize(frame, (640, 480))
                #cv2.imshow(window_name, frame_resized)
                #cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def processing_task():
    with data_lock:
        front_frame       = shared_data['latest_front_frame']
        low_light_active  = shared_data['low_light_active']
        low_light_done    = shared_data['low_light_done']
    
    if front_frame is None:
        return

    # ------------------------------------------------------------------
    # Challenge 1 — Low Light (one-shot, first 10 s only)
    #
    # low_light_done is set to True the moment brightness recovers.
    # After that, the entire brightness block is skipped forever —
    # any dark frame later (shadow, glitch) cannot re-trigger -1.0.
    # ------------------------------------------------------------------
    if not low_light_done:
        brightness = np.mean(cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY))

        if brightness < 60:
            # Light is off — send recovery signal, hold lane, skip tokens
            if not low_light_active:
                print(f"[LOW LIGHT] Detected (brightness={brightness:.1f}). Sending -1.0.")
            with data_lock:
                shared_data['low_light_active']   = True
                shared_data['event_active']       = 'low_brightness'
                shared_data['steering_input']     = 0.0
                shared_data['acceleration_input'] = -1.0
            return   # tokens invisible — skip detection entirely

        else:
            if low_light_active:
                # Brightness just recovered — close the event, arm the one-shot guard
                print("[LOW LIGHT] Recovered. Resuming normal drive.")
                with data_lock:
                    shared_data['low_light_active'] = False
                    shared_data['low_light_done']   = True   # never fires again
                    shared_data['event_active']     = None

    # ------------------------------------------------------------------
    # Normal token detection (runs every cycle when light is on,
    # and permanently after low_light_done = True)
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
    red_mask_high = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))

    token_colors = {
        'green':  (np.array([40, 80, 80]),   np.array([80, 255, 255])),
        'red':    (np.array([0, 120, 70]),    np.array([10, 255, 255])),
        'yellow': (np.array([20, 100, 100]),  np.array([35, 255, 255])),
    }

    h, w = front_frame.shape[:2]
    roi_top = int(h * 0.25)
    roi_bot = int(h * 0.75)

    detected   = None
    token_x    = w // 2
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
                token_area = cv2.contourArea(largest)
                detected   = color_name
                token_x    = x + cw // 2
                break

    mid_x = w // 2
    steer = 0.0
    accel = 1.0
    now   = time.time()

    with data_lock:
        dodge_direction = shared_data['dodge_direction']
        dodge_until     = shared_data['dodge_until']
        lane_offset     = shared_data['lane_offset']

    # GREEN → steer toward it
    if detected == 'green':
        offset = token_x - mid_x
        steer  = max(-1.0, min(1.0, offset / (mid_x * 0.6))) if abs(offset) > 40 else 0.0
        lane_offset = max(-1.0, min(1.0, lane_offset + steer * 0.1))
        with data_lock:
            shared_data['dodge_direction']    = 0.0
            shared_data['dodge_until']        = 0.0
            shared_data['lane_offset']        = lane_offset
            shared_data['steering_input']     = steer
            shared_data['acceleration_input'] = accel

    # RED / YELLOW → dodge away
    elif detected in ('red', 'yellow'):
        if now < dodge_until and dodge_direction != 0.0:
            steer = dodge_direction
        else:
            preferred = 1.0 if token_x < mid_x else -1.0
            if lane_offset >= 0.9 and preferred > 0:
                preferred = -1.0
            elif lane_offset <= -0.9 and preferred < 0:
                preferred = 1.0

            steer = preferred
            hold  = 0.35 if token_area > 2000 else 0.20 if token_area > 800 else 0.12

            dodge_direction = steer
            dodge_until     = now + hold
            lane_offset     = max(-1.0, min(1.0, lane_offset + steer * 0.5))

            with data_lock:
                shared_data['dodge_direction'] = dodge_direction
                shared_data['dodge_until']     = dodge_until
                shared_data['lane_offset']     = lane_offset

        with data_lock:
            shared_data['steering_input']     = steer
            shared_data['acceleration_input'] = accel

    # Nothing → hold dodge then re-centre
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
            with data_lock:
                shared_data['lane_offset'] = lane_offset

        with data_lock:
            shared_data['steering_input']     = steer
            shared_data['acceleration_input'] = accel

    with data_lock:
        shared_data['last_token'] = detected


def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']
        low_light_active   = shared_data['low_light_active']

    if low_light_active:
        # Challenge 1: send recovery signal as-is, never override
        acceleration_input = -1.0
        steering_input     = 0.0
    else:
        # Normal drive: always full throttle
        acceleration_input = 1.0

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
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.HIGH, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

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