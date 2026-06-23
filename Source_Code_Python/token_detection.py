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
                    CHASING_MIN_AREA, CHASING_GRACE_S, CHASING_SOLIDITY_MIN,
                    CHASING_ROI_TOP_FRAC, CHASING_ROI_LEFT_FRAC, CHASING_ROI_RIGHT_FRAC,
                    POLICE_MIN_AREA, POLICE_ABSENT_GRACE,
                    YOLO_MODEL_PATH, YOLO_CONF,
                    TACTICAL_NET_GREEN_TARGET,
                    GOLDEN_LANE_HUD_TOP_FRAC, GOLDEN_LANE_HUD_BOT_FRAC)
from core import shared_data, data_lock
from lane_geometry import lane_of_x, road_polygon, road_bounds
from low_light import detect_blanked_lanes


# ---------------------------------------------------------
# Challenge 2 — Chasing car (TEAL, back camera)
# ---------------------------------------------------------
def _chasing_roi_pixels(h, w):
    return (int(h * CHASING_ROI_TOP_FRAC),
            int(w * CHASING_ROI_LEFT_FRAC),
            int(w * CHASING_ROI_RIGHT_FRAC))


def build_chasing_teal_mask(frame):
    """Teal HSV mask with chasing ROI margins zeroed — for debug overlay."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    roi_top, roi_left, roi_right = _chasing_roi_pixels(h, w)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    teal_mask = cv2.inRange(hsv, np.array([86, 70, 70]), np.array([104, 255, 255]))
    teal_mask[:roi_top, :] = 0
    teal_mask[:, :roi_left] = 0
    teal_mask[:, roi_right:] = 0
    return teal_mask


def _empty_chasing_debug(h=0, w=0):
    roi_top, roi_left, roi_right = _chasing_roi_pixels(h, w) if h and w else (0, 0, 0)
    return {
        'detected': False,
        'raw_detected': False,
        'teal_area': 0.0,
        'teal_solidity': 0.0,
        'bbox': None,
        'reject_reason': None,
        'roi_top': roi_top,
        'roi_left': roi_left,
        'roi_right': roi_right,
    }


def detect_chasing_car_debug(frame):
    """
    Full chasing-car debug pass on the back frame.

    Returns dict with detection result plus ROI, bbox, area, solidity, reject reason.
    ``detected`` matches the original acceptance test (driving unchanged).
    """
    if frame is None:
        return _empty_chasing_debug()

    h, w = frame.shape[:2]
    roi_top, roi_left, roi_right = _chasing_roi_pixels(h, w)
    teal_mask = build_chasing_teal_mask(frame)

    contours, _ = cv2.findContours(teal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_chasing_debug(h, w)

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    x, y, bw, bh = cv2.boundingRect(largest)
    solidity     = area / max(1, bw * bh)
    dbg = {
        'raw_detected': True,
        'teal_area': float(area),
        'teal_solidity': float(solidity),
        'bbox': (x, y, bw, bh),
        'detected': False,
        'reject_reason': None,
        'roi_top': roi_top,
        'roi_left': roi_left,
        'roi_right': roi_right,
    }
    if area < CHASING_MIN_AREA:
        dbg['reject_reason'] = 'area'
        return dbg
    if solidity <= CHASING_SOLIDITY_MIN:
        dbg['reject_reason'] = 'solidity'
        return dbg
    dbg['detected'] = True
    return dbg


def detect_chasing_car(frame):
    """Backward-compatible wrapper — returns debug dict (``detected`` = pass/fail)."""
    return detect_chasing_car_debug(frame)


_chase_dbg_prev = {
    'raw': False, 'detected': False, 'active': False,
    'reject_reason': None, 'near_reject': None,
}


def _log_chase_debug_changes(chase_dbg, chasing_detected, chasing_active, appearance_count):
    """Console logs only on chasing detection status changes."""
    global _chase_dbg_prev
    area = chase_dbg['teal_area']
    sol = chase_dbg['teal_solidity']
    reject_reason = chase_dbg.get('reject_reason')

    if chasing_detected and not _chase_dbg_prev['detected']:
        print(f"[CHASE DBG] Cyan car detected — area={area:.0f} sol={sol:.2f} "
              f"bbox={chase_dbg['bbox']}")

    if not chasing_detected and _chase_dbg_prev['detected']:
        print("[CHASE DBG] Cyan car lost from back camera")

    if chasing_active and not _chase_dbg_prev['active']:
        print(f"[CHASING CAR] chasing_active=True (appearance #{appearance_count})")
    elif not chasing_active and _chase_dbg_prev['active']:
        print("[CHASE DBG] chasing_active=False")

    if (reject_reason == 'area' and reject_reason != _chase_dbg_prev['reject_reason']
            and chase_dbg['raw_detected'] and not chasing_detected):
        print(f"[CHASE DBG] Teal rejected (area too small): area={area:.0f} < {CHASING_MIN_AREA}")

    if (reject_reason == 'solidity' and reject_reason != _chase_dbg_prev['reject_reason']
            and chase_dbg['raw_detected'] and not chasing_detected):
        print(f"[CHASE DBG] Teal rejected (solidity): area={area:.0f} sol={sol:.2f} <= "
              f"{CHASING_SOLIDITY_MIN:.2f}")

    near_reject = None
    if chase_dbg['raw_detected'] and not chasing_detected:
        if area >= CHASING_MIN_AREA * 0.70 and area < CHASING_MIN_AREA:
            near_reject = ('area', area, sol)
        elif area >= CHASING_MIN_AREA and sol <= CHASING_SOLIDITY_MIN:
            near_reject = ('solidity', area, sol)

    if near_reject and near_reject != _chase_dbg_prev['near_reject']:
        kind, a, s = near_reject
        if kind == 'area':
            print(f"[CHASE DBG] Near threshold — area={a:.0f} (min={CHASING_MIN_AREA})")
        else:
            print(f"[CHASE DBG] Near threshold — sol={s:.2f} (min={CHASING_SOLIDITY_MIN:.2f})")

    _chase_dbg_prev = {
        'raw': chase_dbg['raw_detected'],
        'detected': chasing_detected,
        'active': chasing_active,
        'reject_reason': reject_reason if chase_dbg['raw_detected'] and not chasing_detected else None,
        'near_reject': near_reject,
    }


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
# Coins — trust / curb classification (front camera only)
# ---------------------------------------------------------
_TOKEN_MAX_COIN_AREA   = 3200
_TOKEN_CURB_STRIPE_AR  = 1.55
_TOKEN_CURB_MAX_EXTENT = 0.42
_TOKEN_WIDE_BBOX_FRAC  = 0.45
_TOKEN_CONF_MIN        = 0.12
_TOKEN_CONF_EDGE_START = 0.68
_TOKEN_CONF_LARGE_AREA = 1600
_FAR_TOKEN_Y_FRAC      = 0.55
_TOKEN_MIN_AREA_FAR    = 12
_FAR_TOKEN_CONF_FLOOR  = 0.72
_CURB_EDGE_NEAR_FRAC   = 0.78
_CURB_STRIPE_ELONG     = 1.38
_CURB_LOW_EXTENT       = 0.40
_CURB_COMPACT_ELONG    = 1.30
_CURB_COMPACT_EXTENT   = 0.42
_CURB_LIKE_CONF        = 0.15


def _dynamic_min_area(cy, frame_h, base_min):
    """Allow smaller blobs far away (higher on screen)."""
    far_y = _FAR_TOKEN_Y_FRAC * frame_h
    if cy >= far_y:
        return base_min
    t = cy / max(1.0, far_y)
    return max(_TOKEN_MIN_AREA_FAR, int(base_min * (0.4 + 0.6 * t)))


def _shape_metrics(cw, ch, area):
    ar = cw / max(1, ch)
    extent = area / max(1, cw * ch)
    elong = max(ar, 1.0 / max(ar, 1e-3))
    return ar, extent, elong


def _is_compact_coin(elong, extent):
    return elong < _CURB_COMPACT_ELONG and extent >= _CURB_COMPACT_EXTENT


def _bad_shape(elong, extent, cw, road_w):
    if elong >= _CURB_STRIPE_ELONG and extent < _CURB_LOW_EXTENT:
        return True
    if cw / max(1, road_w) > _TOKEN_WIDE_BBOX_FRAC and extent < _CURB_LOW_EXTENT:
        return True
    return False


def _near_road_edge(tx, ty, frame_w, frame_h):
    left, right = road_bounds(ty, frame_w, frame_h)
    road_w = max(1, right - left)
    frac = (tx - left) / road_w
    edge_band = 1.0 - _CURB_EDGE_NEAR_FRAC
    return frac <= edge_band or frac >= _CURB_EDGE_NEAR_FRAC


def _is_curb_like(color, tx, ty, cw, ch, elong, extent, frame_w, frame_h):
    """Red curb rumble: near road edge AND stripe-like — not compact edge coins."""
    if color != 'red':
        return False
    if _is_compact_coin(elong, extent):
        return False
    left, right = road_bounds(ty, frame_w, frame_h)
    road_w = max(1, right - left)
    return _near_road_edge(tx, ty, frame_w, frame_h) and _bad_shape(elong, extent, cw, road_w)


def _token_confidence(tx, ty, cw, ch, area, elong, extent, frame_w, frame_h, is_curb):
    if is_curb:
        return _CURB_LIKE_CONF
    conf = 1.0
    if _is_compact_coin(elong, extent):
        conf = max(conf, 0.88)
    if area > _TOKEN_CONF_LARGE_AREA:
        conf *= max(0.5, 1.0 - (area - _TOKEN_CONF_LARGE_AREA) / 4000.0)
    left, right = road_bounds(ty, frame_w, frame_h)
    road_w = max(1, right - left)
    frac = (tx - left) / road_w
    edge_dist = min(frac, 1.0 - frac)
    if _near_road_edge(tx, ty, frame_w, frame_h) and _bad_shape(elong, extent, cw, road_w):
        conf *= max(_TOKEN_CONF_MIN, edge_dist / _TOKEN_CONF_EDGE_START)
    if ty < _FAR_TOKEN_Y_FRAC * frame_h and _is_compact_coin(elong, extent):
        conf = max(conf, _FAR_TOKEN_CONF_FLOOR)
    return max(_TOKEN_CONF_MIN, min(1.0, conf))


def _coin_shape_ok(c, area, min_area, frame_w, frame_h, ty, cw, ch):
    """Compact/round blob test — rejects HUD digits, lane dashes, edge stripes."""
    if area < min_area:
        return None
    if area > _TOKEN_MAX_COIN_AREA:
        return None
    ar, extent, elong = _shape_metrics(cw, ch, area)
    if not (TOKEN_AR_LO < ar < TOKEN_AR_HI) or extent < TOKEN_MIN_EXTENT:
        return None
    left, right = road_bounds(ty, frame_w, frame_h)
    road_w = max(1, right - left)
    if elong >= _TOKEN_CURB_STRIPE_AR and extent < _TOKEN_CURB_MAX_EXTENT:
        return None
    if cw / road_w > _TOKEN_WIDE_BBOX_FRAC and extent < _TOKEN_CURB_MAX_EXTENT:
        return None
    return ar, extent, elong


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
            x, y, cw, ch = cv2.boundingRect(c)
            tx, ty = x + cw // 2, y + ch // 2
            dyn_min = _dynamic_min_area(ty, frame_h, min_area)
            shape = _coin_shape_ok(c, area, dyn_min, frame_w, frame_h, ty, cw, ch)
            if shape is None:
                continue
            ar, extent, elong = shape
            is_curb = _is_curb_like(color_name, tx, ty, cw, ch, elong, extent, frame_w, frame_h)
            conf = _token_confidence(tx, ty, cw, ch, area, elong, extent, frame_w, frame_h, is_curb)
            tokens.append({'color': color_name, 'x': tx, 'y': ty,
                           'w': cw, 'h': ch, 'area': area,
                           'lane': lane_of_x(tx, ty, frame_w, frame_h),
                           'confidence': conf, 'is_curb_like': is_curb})

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
                           'lane': lane_of_x(cx, cy, frame_w, frame_h),
                           'confidence': float(d.get('conf', 1.0)),
                           'is_curb_like': False})
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
# Golden Lane — HUD text detection (front camera, top strip)
# The game flashes bright yellow "LANE N — ALL GREEN!" in the HUD.
# We mask the yellow pixels, find the blob centroid, and map it to a lane index.
# ---------------------------------------------------------
_GOLD_TEXT_HSV_LO = np.array([18, 160, 180])   # bright warm yellow
_GOLD_TEXT_HSV_HI = np.array([38, 255, 255])
_GOLD_MIN_AREA    = 80                          # minimum yellow-pixel area to count

def detect_golden_lane_hud(frame):
    """
    Scan the top HUD band for the bright-yellow 'LANE N — ALL GREEN!' text.
    Returns (lane_index, detected): lane_index is 0-indexed (0-4), or -1 if not found.
    """
    if frame is None:
        return -1
    h, w = frame.shape[:2]
    y0 = int(GOLDEN_LANE_HUD_TOP_FRAC * h)
    y1 = int(GOLDEN_LANE_HUD_BOT_FRAC * h)
    hud = frame[y0:y1, :]
    hsv  = cv2.cvtColor(hud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GOLD_TEXT_HSV_LO, _GOLD_TEXT_HSV_HI)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < _GOLD_MIN_AREA:
        return -1
    x, y, bw, bh = cv2.boundingRect(largest)
    text_cx   = x + bw // 2
    lane_index = int(round((text_cx / w) * NUM_LANES - 0.5))
    return max(0, min(NUM_LANES - 1, lane_index))


# ---------------------------------------------------------
# Detection Task (MEDIUM priority, 10 ms period)
# Runs all the perception and publishes tokens / police / chasing state.
# ---------------------------------------------------------
def detection_task():
    with data_lock:
        frame              = shared_data['latest_front_frame']
        back_frame         = shared_data['latest_back_frame']
        low_light_active   = shared_data['low_light_active']
        detection_urgent   = shared_data.get('detection_urgent', False)

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
        chase_dbg = detect_chasing_car_debug(back_frame)
    else:
        # AutoRS outer-loop analogue: when chasing/police urgent, always run
        # back-camera detection but optionally reuse last token list to save CPU.
        chase_dbg        = detect_chasing_car_debug(back_frame)
        chasing_detected = chase_dbg['detected']
        if detection_urgent and not chasing_detected:
            # Threat cleared — do full front-camera scan this cycle.
            detection_urgent = False
        with data_lock:
            prev_tokens     = shared_data.get('tokens', [])
            prev_police_bbox = shared_data.get('police_bbox', None)
        if detection_urgent:
            # Skip expensive HSV scan; reuse last known tokens.
            tokens      = prev_tokens
            police_bbox = prev_police_bbox
        else:
            hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            tokens      = classify_tokens(hsv, h, w, enable_gray=any(blanked_lanes))
            police_bbox = detect_police_car(hsv, h, w)

    now = time.time()
    with data_lock:
        police_active  = shared_data['police_active']
        chasing_active = shared_data['chasing_active']

    # --- Chasing state update (track 1st vs 2nd appearance) ---
    if chasing_detected:
        if not chasing_active:
            with data_lock:
                shared_data['chasing_appearance_count'] += 1
        with data_lock:
            shared_data['chasing_active']     = True
            shared_data['chasing_start_time'] = shared_data['chasing_start_time'] if chasing_active else now
            shared_data['chasing_last_seen']  = now
            shared_data['chasing_teal_area']  = chase_dbg['teal_area']
            shared_data['event_active']       = 'chasing'
    else:
        with data_lock:
            chasing_last_seen = shared_data['chasing_last_seen']
        if chasing_active and (now - chasing_last_seen) > CHASING_GRACE_S:
            with data_lock:
                shared_data['chasing_active']    = False
                shared_data['chasing_teal_area'] = 0.0
                if shared_data.get('event_active') == 'chasing':
                    shared_data['event_active'] = None

    with data_lock:
        appearance_count = shared_data['chasing_appearance_count']
        chasing_active = shared_data['chasing_active']
    _log_chase_debug_changes(chase_dbg, chasing_detected, chasing_active, appearance_count)

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
        shared_data['chasing_raw_detected']    = chase_dbg['raw_detected']
        shared_data['chasing_detected']        = chase_dbg['detected']
        shared_data['chasing_teal_area']       = chase_dbg['teal_area']
        shared_data['chasing_teal_solidity']   = chase_dbg['teal_solidity']
        shared_data['chasing_bbox']            = chase_dbg['bbox']
        shared_data['chasing_mask_roi_top']    = chase_dbg['roi_top']
        shared_data['chasing_mask_roi_left']   = chase_dbg['roi_left']
        shared_data['chasing_mask_roi_right']  = chase_dbg['roi_right']

    # ----------------------------------------------------------------
    # Golden Lane — detect HUD text and publish raw detection result.
    # The watchdog in steering.py owns the timer, pass check, and window state.
    # ----------------------------------------------------------------
    detected_gl_lane = detect_golden_lane_hud(frame)
    with data_lock:
        shared_data['golden_lane_detected_raw']  = detected_gl_lane >= 0
        shared_data['golden_lane_detected_lane'] = detected_gl_lane

    # ----------------------------------------------------------------
    # Golden Lane — override token colours in the active golden lane
    # ----------------------------------------------------------------
    with data_lock:
        gl_active = shared_data['golden_lane_active']
        gl_lane   = shared_data['golden_lane_number']

    if gl_active and gl_lane >= 0:
        for t in tokens:
            if t.get('lane', -1) == gl_lane:
                t['color'] = 'green'        # force green for scoring
        # Re-publish the modified token list
        with data_lock:
            shared_data['tokens'] = tokens

    # ----------------------------------------------------------------
    # Tactical scoring — count tokens that have just been picked up
    # (a token is "collected" when it was present last frame but is gone now)
    # ----------------------------------------------------------------
    _update_tactical_score(tokens)


# ---------------------------------------------------------
# Tactical score helper — token disappearance = pickup
# ---------------------------------------------------------
# We compare each previous token's (lane, color) pair against the current frame.
# If a previous token is no longer represented, the car collected it.

_PICKUP_MATCH_RADIUS_PX = 60   # max pixel distance to still be "the same token"

def _token_still_present(prev_tok, current_tokens):
    """Return True if prev_tok has a matching token (same colour, nearby) this frame."""
    px, py = prev_tok['x'], prev_tok['y']
    col    = prev_tok['color']
    for t in current_tokens:
        if t['color'] == col:
            dx = t['x'] - px
            dy = t['y'] - py
            if (dx * dx + dy * dy) <= _PICKUP_MATCH_RADIUS_PX ** 2:
                return True
    return False


def _update_tactical_score(current_tokens):
    """
    Compare current_tokens against the previous frame's token list.
    Tokens that have disappeared are counted as collected.
    Only counts green and red tokens (yellow = obstacle, not scored tactically).
    """
    with data_lock:
        prev_tokens   = shared_data.get('tactical_prev_tokens', [])
        green_count   = shared_data['tactical_green_collected']
        red_count     = shared_data['tactical_red_collected']

    new_green = 0
    new_red   = 0
    for pt in prev_tokens:
        if pt['color'] not in ('green', 'red'):
            continue
        if pt.get('is_curb_like', False):
            continue
        if not _token_still_present(pt, current_tokens):
            # Token disappeared — car drove over it
            if pt['color'] == 'green':
                new_green += 1
            else:
                new_red += 1

    if new_green > 0 or new_red > 0:
        net = (green_count + new_green) - (red_count + new_red)
        print(f"[TACTICAL] +{new_green} green  +{new_red} red  "
              f"(net={net})")
        with data_lock:
            shared_data['tactical_green_collected'] += new_green
            shared_data['tactical_red_collected']   += new_red
            shared_data['tactical_net_green']        = (
                shared_data['tactical_green_collected']
                - shared_data['tactical_red_collected']
            )

    # Store current list for the next frame comparison
    with data_lock:
        shared_data['tactical_prev_tokens'] = list(current_tokens)
