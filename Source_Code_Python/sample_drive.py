"""
sample_drive.py — entry point. Wires the team's modules together and runs the
Rate-Monotonic schedule. Run this:  python sample_drive.py
Owner: Adam (integration)

Module map (who owns what):
    config.py          - shared constants                         (Adam)
    core.py            - shared state + LOCKED framework/network   (Adam)
    lane_geometry.py   - perspective lane math                     (shared)
    low_light.py       - Challenge 1: brightness/blackout          (Adam)
    token_detection.py - coins / police / chasing detection        (syednopal)
    steering.py        - scoring + decision + control output       (umayra)
    visualization.py   - debug overlay                             (aisyah)
"""
import threading
import time
import cv2

import core
from core import (RTTask, TaskPriority, shared_data, data_lock,
                  setup_cameras, setup_control_server,
                  read_front_camera_task, read_back_camera_task)
from config import CENTER_LANE
from low_light import brightness_task
from token_detection import detection_task
from steering import decision_task, send_controls_task, police_watchdog_task
from visualization import draw_debug, draw_back_debug


# ---------------------------------------------------------
# Main — Rate Monotonic Scheduling (shorter period = higher priority)
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
    print("  DecisionTask   : 5ms   | HIGH   — 5-lane scoring state machine, no cv2")
    print("  ReadBackCamera : 10ms  | MEDIUM — back camera feed")
    print("  DetectionTask  : 10ms  | MEDIUM — heavy cv2: coins + police + chasing car")
    print("  PoliceWatchdog : 100ms | LOW    — Challenge 3: deadline countdown\n")

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
        while core.is_running:
            with data_lock:
                front_frame = shared_data['latest_front_frame']
                back_frame  = shared_data['latest_back_frame']
                dbg_tokens  = shared_data.get('tokens', [])
                dbg_scores  = shared_data.get('lane_scores', [])
                dbg_state   = shared_data.get('state', '')
                dbg_target  = shared_data.get('target_lane', CENTER_LANE)
                dbg_tx      = shared_data.get('target_x', 0)
                dbg_ly      = shared_data.get('lookahead_y', 0)

            # OpenCV GUI must run in the main thread
            if front_frame is not None:
                cv2.imshow("Front Camera",
                           cv2.resize(draw_debug(front_frame, dbg_tokens, dbg_scores,
                                                 dbg_state, dbg_target, dbg_tx, dbg_ly),
                                      (640, 480)))
            if back_frame is not None:
                cv2.imshow("Back Camera", cv2.resize(draw_back_debug(back_frame), (640, 480)))

            if front_frame is not None or back_frame is not None:
                cv2.waitKey(10)
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        core.is_running = False

    t_send_controls.join()
    t_front_camera.join()
    t_brightness.join()
    t_decision.join()
    t_back_camera.join()
    t_detection.join()
    t_police_watchdog.join()

    if core.front_camera_sock:
        core.front_camera_sock.close()
    if core.back_camera_sock:
        core.back_camera_sock.close()
    if core.control_conn:
        core.control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
