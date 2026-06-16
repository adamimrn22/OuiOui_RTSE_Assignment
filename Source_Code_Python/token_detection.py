"""
token_detection.py — perception: coins, police car, chasing car (+ optional YOLO).

Runs all the OpenCV work and publishes results into shared_data. No steering here.
Coins are classified by colour (red/yellow dangerous, green collectible), the
police car by blue, the chasing car by teal. detection_task is the orchestrator.
"""
import cv2
import numpy as np
import time

from config import (USE_YOLO, NUM_LANES, TOKEN_MIN_AREA, GRAY_MIN_AREA,
                    TOKEN_MIN_EXTENT, TOKEN_AR_LO, TOKEN_AR_HI,
                    CAR_MASK_TOP_FRAC, CAR_MASK_X0_FRAC, CAR_MASK_X1_FRAC,
                    CHASING_MIN_AREA, CHASING_GRACE_S,
                    POLICE_MIN_AREA, POLICE_ABSENT_GRACE,
                    YOLO_MODEL_PATH, YOLO_CONF)
from core import shared_data, data_lock
from lane_geometry import lane_of_x, road_polygon
from low_light import detect_blanked_lanes


# ---------------------------------------------------------
# Challenge 2 — Chasing car (TEAL, back camera)
# ---------------------------------------------------------
def detect_chasing_car(frame):
    """
    The chasing car is TEAL (hue ~86-104) — distinct from the grass (plain green,
    hue ~60). Require teal hue + SIZE + SOLIDITY in the CENTRAL ROAD band of the
    back frame so grass on curves isn't mistaken for the car.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    teal_mask = cv2.inRange(hsv, np.array([86, 70, 70]), np.array([104, 255, 255]))

    teal_mask[:int(h * 0.25), :] = 0          # ignore sky/horizon
    teal_mask[:, :int(w * 0.25)] = 0          # ignore left margin
    teal_mask[:, int(w * 0.75):] = 0          # ignore right margin

    contours, _ = cv2.findContours(teal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < CHASING_MIN_AREA:
        return False   # too small => a coin, not the car

    x, y, bw, bh = cv2.boundingRect(largest)
    solidity     = area / max(1, bw * bh)
    return solidity > 0.40


# ---------------------------------------------------------
# Challenge 3 — Police car (BLUE, front camera)
# ---------------------------------------------------------
def detect_police_car(hsv, frame_h, frame_w):
    """Large BLUE object on the road = police car. Returns its bbox or None."""
    blue_mask = cv2.inRange(hsv, np.array([108, 90, 90]), np.array([128, 255, 255]))
    # Search well BELOW the skyline — blue/purple city buildings sit at the horizon.
    roi_top = int(frame_h * 0.52)
    roi_bot = int(frame_h * 0.85)
    blue_mask[:roi_top, :] = 0
    blue_mask[roi_bot:,  :] = 0

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < POLICE_MIN_AREA:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    aspect = bw / max(1, bh)
    if bw > 0.45 * frame_w or aspect > 2.5:    # reject wide flat strips (buildings)
        return None
    return (x, y, bw, bh)


# ---------------------------------------------------------
# Coins
# ---------------------------------------------------------
def _coin_shape_ok(c, area, min_area):
    """Compact/round blob test — rejects HUD digits, lane dashes, edge stripes."""
    if area < min_area:
        return None
    x, y, cw, ch = cv2.boundingRect(c)
    ar     = cw / max(1, ch)
    extent = area / max(1, cw * ch)
    if not (TOKEN_AR_LO < ar < TOKEN_AR_HI) or extent < TOKEN_MIN_EXTENT:
        return None
    return x, y, cw, ch


def classify_tokens(hsv, frame_h, frame_w, enable_gray=False):
    """
    Classify coins by colour inside the road trapezoid, with our own car sprite
    masked out. red/yellow = dangerous, green = collectible, unknown (grayed) only
    during an active blackout. Each token is tagged with its perspective lane.
    """
    road_mask = np.zeros((frame_h, frame_w), np.uint8)
    cv2.fillPoly(road_mask, [road_polygon(frame_w, frame_h)], 255)
    cx0 = int(CAR_MASK_X0_FRAC * frame_w); cx1 = int(CAR_MASK_X1_FRAC * frame_w)
    cy0 = int(CAR_MASK_TOP_FRAC * frame_h)
    road_mask[cy0:, cx0:cx1] = 0               # blank our own car

    mask_red1 = cv2.inRange(hsv, np.array([0,   45, 50]),  np.array([15,  255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 45, 50]),  np.array([180, 255, 255]))
    masks = {
        'red':    cv2.bitwise_or(mask_red1, mask_red2),
        'yellow': cv2.inRange(hsv, np.array([16, 45, 50]),  np.array([34,  255, 255])),
        'green':  cv2.inRange(hsv, np.array([40, 60, 70]),  np.array([85,  255, 255])),
    }

    tokens = []
    def collect(mask, color_name, min_area):
        mask = cv2.bitwise_and(mask, road_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            box  = _coin_shape_ok(c, area, min_area)
            if box is None:
                continue
            x, y, cw, ch = box
            tx, ty = x + cw // 2, y + ch // 2
            tokens.append({'color': color_name, 'x': tx, 'y': ty,
                           'w': cw, 'h': ch, 'area': area,
                           'lane': lane_of_x(tx, ty, frame_w, frame_h)})

    for color_name, mask in masks.items():
        collect(mask, color_name, TOKEN_MIN_AREA)
    if enable_gray:
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 45, 255]))
        collect(gray_mask, 'unknown', GRAY_MIN_AREA)
    return tokens


# ---------------------------------------------------------
# Optional YOLOv8 backend — same outputs as the HSV detectors.
# ---------------------------------------------------------
_yolo_detector = None
_YOLO_CLS_COLOR = {0: 'red', 1: 'yellow', 2: 'green'}  # 3=police_car, 4=chasing_car

def _get_yolo():
    global _yolo_detector
    if _yolo_detector is None:
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, 'yolo'))
        from yolo_detect import YoloDetector
        path = YOLO_MODEL_PATH
        if not os.path.isabs(path):
            path = os.path.join(here, path)
        _yolo_detector = YoloDetector(path, conf=YOLO_CONF)
        print(f"[YOLO] Loaded model: {path}")
    return _yolo_detector

def detect_with_yolo(frame, back_frame, frame_h, frame_w):
    """YOLO equivalent of the HSV pipeline -> (tokens, police_bbox, chasing_detected)."""
    det = _get_yolo()
    tokens = []
    police_bbox = None
    best_police_area = 0.0

    for d in det.infer(frame):
        cls = d['cls']
        cx, cy = int(d['cx']), int(d['cy'])
        bw, bh = int(d['w']), int(d['h'])
        if cls in _YOLO_CLS_COLOR:
            tokens.append({'color': _YOLO_CLS_COLOR[cls], 'x': cx, 'y': cy,
                           'w': bw, 'h': bh, 'area': bw * bh,
                           'lane': lane_of_x(cx, cy, frame_w, frame_h)})
        elif cls == 3:
            if bw * bh > best_police_area:
                best_police_area = bw * bh
                police_bbox = (cx - bw // 2, cy - bh // 2, bw, bh)

    chasing_detected = False
    if back_frame is not None:
        for d in det.infer(back_frame):
            if d['cls'] == 4:
                chasing_detected = True
                break
    return tokens, police_bbox, chasing_detected


# ---------------------------------------------------------
# Detection Task (MEDIUM priority, 10 ms period)
# Runs all the perception and publishes tokens / police / chasing state.
# ---------------------------------------------------------
def detection_task():
    with data_lock:
        frame            = shared_data['latest_front_frame']
        back_frame       = shared_data['latest_back_frame']
        low_light_active = shared_data['low_light_active']

    if frame is None:
        return

    # Challenge 1 — while the light is off, tokens are invisible; skip cv2.
    if low_light_active:
        with data_lock:
            shared_data['tokens'] = []
        return

    h, w = frame.shape[:2]

    # Frame-health / lane-blanking is always done in gray (YOLO can't see blackouts).
    blanked_lanes = detect_blanked_lanes(frame)

    if USE_YOLO:
        tokens, police_bbox, chasing_detected = detect_with_yolo(frame, back_frame, h, w)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        tokens      = classify_tokens(hsv, h, w, enable_gray=any(blanked_lanes))
        police_bbox = detect_police_car(hsv, h, w)
        chasing_detected = detect_chasing_car(back_frame) if back_frame is not None else False

    now = time.time()
    with data_lock:
        police_active  = shared_data['police_active']
        chasing_active = shared_data['chasing_active']

    # --- Chasing state update (track 1st vs 2nd appearance) ---
    if chasing_detected:
        if not chasing_active:
            with data_lock:
                shared_data['chasing_appearance_count'] += 1
                count = shared_data['chasing_appearance_count']
            print(f"[CHASING CAR] Appearance #{count} detected behind us! Evasive action!")
        with data_lock:
            shared_data['chasing_active']     = True
            shared_data['chasing_start_time'] = shared_data['chasing_start_time'] if chasing_active else now
            shared_data['chasing_last_seen']  = now
            shared_data['event_active']       = 'chasing'
    else:
        with data_lock:
            chasing_last_seen = shared_data['chasing_last_seen']
        if chasing_active and (now - chasing_last_seen) > CHASING_GRACE_S:
            print("[CHASING CAR] Evaded / gone. Returning to normal mode.")
            with data_lock:
                shared_data['chasing_active'] = False
                if shared_data.get('event_active') == 'chasing':
                    shared_data['event_active'] = None

    # --- Police state update ---
    if police_bbox is not None:
        if not police_active:
            print("[POLICE] Police car detected! 10 seconds to collect a RED token!")
        with data_lock:
            shared_data['police_active']     = True
            shared_data['police_start_time'] = shared_data['police_start_time'] if police_active else now
            shared_data['police_bbox']       = police_bbox
            shared_data['police_bbox_time']  = now
            shared_data['police_last_seen']  = now
            shared_data['event_active']      = 'police'
    else:
        with data_lock:
            last_seen      = shared_data['police_last_seen']
            last_bbox_time = shared_data['police_bbox_time']
        if police_active and (now - last_seen) > POLICE_ABSENT_GRACE:
            print("[POLICE] Police car gone — returning to normal mode.")
            with data_lock:
                shared_data['police_active'] = False
                shared_data['police_urgent'] = False
                shared_data['police_bbox']   = None
                shared_data['event_active']  = None
        elif (now - last_bbox_time) > 1.0:
            with data_lock:
                shared_data['police_bbox'] = None

    # --- Single brief write-back ---
    with data_lock:
        shared_data['tokens']        = tokens
        shared_data['blanked_lanes'] = blanked_lanes
        shared_data['frame_w']       = w
        shared_data['frame_h']       = h
