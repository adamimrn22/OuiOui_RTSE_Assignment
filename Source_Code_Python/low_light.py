"""
low_light.py — Challenge 1 (low-light detection) + frame-health / blackout check. 

brightness_task: dedicated fast task that flips the single low_light_active flag
the whole system gates on. detect_blanked_lanes: finds blacked-out lanes caused
by the yellow perception attack.
"""
import cv2
import numpy as np
import time

from config import NUM_LANES, LOW_BRIGHTNESS_THRESHOLD, BLANK_LANE_THRESHOLD
from core import shared_data, data_lock


# ---------------------------------------------------------
# Challenge 1 — Brightness Task (HIGH priority, 5 ms period)
#
# Reads one frame, computes the mean road brightness, and flips low_light_active.
# Kept independent of the heavier detection_task so the light-off event is caught
# immediately and reliably.
# ---------------------------------------------------------
def brightness_task():
    with data_lock:
        frame = shared_data['latest_front_frame']

    if frame is None:
        return

    # Brightness of the road area only (ignore dark sky / UI).
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi  = gray[int(h*0.4):int(h*0.9), int(w*0.2):int(w*0.8)]
    brightness = np.mean(roi)
    now        = time.time()

    with data_lock:
        currently_active = shared_data['low_light_active']

    if not currently_active:
        if brightness < LOW_BRIGHTNESS_THRESHOLD:
            print(f"[LOW LIGHT] Detected! Road Brightness={brightness:.1f}. Sending acceleration=-1.0.")
            with data_lock:
                shared_data['low_light_active']     = True
                shared_data['low_light_start_time'] = now
                shared_data['event_active']         = 'low_brightness'
    else:
        if int(now * 10) % 5 == 0:
            print(f"[LOW LIGHT] Waiting for light... Road Brightness={brightness:.1f}")

        # Hysteresis: recover a bit above the trigger so the state doesn't flicker.
        if brightness > LOW_BRIGHTNESS_THRESHOLD + 2:
            with data_lock:
                elapsed = now - shared_data['low_light_start_time']
            print(f"[LOW LIGHT] Recovered after {elapsed:.2f}s! Road Brightness={brightness:.1f}. Resuming.")
            with data_lock:
                shared_data['low_light_active'] = False
                shared_data['event_active']     = None


def detect_blanked_lanes(frame):
    """
    Yellow-attack robustness: find lane columns that have gone near-black (a
    blanked side of the screen). Returns a bool list, one per lane. These lanes
    are NOT treated as free space by the scorer.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    band = gray[int(h*0.3):h, :]   # the drivable region, ignore the sky band
    blanked = []
    for i in range(NUM_LANES):
        x0 = int(i * w / NUM_LANES)
        x1 = int((i + 1) * w / NUM_LANES)
        col = band[:, x0:x1]
        blanked.append(bool(np.mean(col) < BLANK_LANE_THRESHOLD))
    return blanked
