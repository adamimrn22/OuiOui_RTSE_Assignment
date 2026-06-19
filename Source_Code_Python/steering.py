"""
steering.py — decision making: lane scoring, the priority state machine,
pure-pursuit steering, the police deadline watchdog, and sending controls.
Owner: umayra

No OpenCV here. Reads the detection results from shared_data and writes
steering/acceleration back. Priority: LOW_LIGHT > CHASING > POLICE > COIN > NORMAL.
"""
import struct
import time
import numpy as np

from config import (NUM_LANES, CENTER_LANE,
                    STEER_DEADZONE_FRAC, STEER_GAIN_NORMAL, STEER_GAIN_AGGRO,
                    DEFAULT_LOOKAHEAD_FRAC, LANE_COMMIT_S,
                    SCORE_CLEAR, SCORE_LANE_CHANGE, SCORE_GREEN_REWARD,
                    GREEN_WEIGHT_FLOOR, SCORE_RED_PENALTY, SCORE_YELLOW_PENALTY,
                    SCORE_UNKNOWN_PENALTY, SCORE_POLICE_BLOB, SCORE_BLANKED_LANE,
                    SCORE_RED_TARGET, SWITCH_MARGIN,
                    POLICE_DEADLINE, POLICE_URGENT_THRESHOLD,
                    CHASING_1ST_WINDOW, CHASING_2ND_WINDOW)
import core
from core import shared_data, data_lock, DriveState
from lane_geometry import lane_of_x, lane_center_at_row, car_center_x, proximity_weight


# ---------------------------------------------------------
# Lane scoring core
# ---------------------------------------------------------
def score_lanes(tokens, police_bbox, blanked_lanes, current_lane,
                state, police_urgent, frame_w, frame_h):
    """Assign every lane a score; the caller picks the highest (with hysteresis)."""
    scores = [SCORE_CLEAR] * NUM_LANES

    for i in range(NUM_LANES):
        scores[i] -= SCORE_LANE_CHANGE * abs(i - current_lane)

    red_is_target = (state == DriveState.POLICE)

    for t in tokens:
        lane = t.get('lane', lane_of_x(t['x'], t['y'], frame_w, frame_h))
        w    = proximity_weight(t, frame_h)
        color = t['color']

        if color == 'green':
            # Keep a weight FLOOR so far green still pulls us toward its lane early.
            if not (state == DriveState.POLICE and police_urgent):
                scores[lane] += SCORE_GREEN_REWARD * max(GREEN_WEIGHT_FLOOR, w)
        elif color == 'red':
            if red_is_target:
                scores[lane] += SCORE_RED_TARGET * w
            else:
                scores[lane] -= SCORE_RED_PENALTY * w
        elif color == 'yellow':
            scores[lane] -= SCORE_YELLOW_PENALTY * w
        else:  # 'unknown'
            scores[lane] -= SCORE_UNKNOWN_PENALTY * w

    if police_bbox is not None:
        px, py, pw, ph = police_bbox
        ry = py + ph
        left_lane  = lane_of_x(px, ry, frame_w, frame_h)
        right_lane = lane_of_x(px + pw, ry, frame_w, frame_h)
        for i in range(min(left_lane, right_lane), max(left_lane, right_lane) + 1):
            scores[i] -= SCORE_POLICE_BLOB

    for i in range(NUM_LANES):
        if i < len(blanked_lanes) and blanked_lanes[i]:
            scores[i] -= SCORE_BLANKED_LANE

    return scores


def choose_target_lane(scores, current_target):
    """Best lane, but only switch if it beats the current target by SWITCH_MARGIN."""
    best = int(np.argmax(scores))
    if best == current_target:
        return best
    if scores[best] > scores[current_target] + SWITCH_MARGIN:
        return best
    return current_target


def steer_toward(target_x, mid_x, frame_w, gain):
    """Proportional pure-pursuit steering toward target_x, with a centre dead-zone."""
    offset = target_x - mid_x
    if abs(offset) < frame_w * STEER_DEADZONE_FRAC:
        return 0.0
    return float(max(-1.0, min(1.0, offset / (frame_w * gain))))


# ---------------------------------------------------------
# Events Watchdog Task (LOW priority, 100 ms)
# Tracks deadlines for BOTH the police car (10 s) and the chasing car
# (10 s / 3 s depending on appearance count). Previously only police was
# tracked; chasing_2nd_window was defined in config but never enforced.
# ---------------------------------------------------------
def police_watchdog_task():
    now = time.time()

    with data_lock:
        police_active      = shared_data['police_active']
        police_start_time  = shared_data['police_start_time']
        chasing_active     = shared_data['chasing_active']
        chasing_start_time = shared_data['chasing_start_time']
        chasing_count      = shared_data['chasing_appearance_count']

    # --- Police deadline ---
    if police_active:
        elapsed   = now - police_start_time
        remaining = POLICE_DEADLINE - elapsed

        if remaining <= 0:
            with data_lock:
                shared_data['police_active'] = False
                shared_data['police_urgent'] = False
                shared_data['police_bbox']   = None
                shared_data['event_active']  = None
            print("[POLICE] Deadline expired — 50% speed penalty applied by game.")
        else:
            urgent = remaining <= POLICE_URGENT_THRESHOLD
            with data_lock:
                shared_data['police_urgent'] = urgent
            if urgent:
                print(f"[POLICE] URGENT — {remaining:.1f}s left! Committing to RED token only!")
            else:
                print(f"[POLICE] Active — {remaining:.1f}s remaining to collect RED token.")

    # --- Chasing car deadline (was never enforced — CHASING_2ND_WINDOW existed in
    #     config but nothing read it) ---
    if chasing_active:
        window    = CHASING_2ND_WINDOW if chasing_count >= 2 else CHASING_1ST_WINDOW
        elapsed   = now - chasing_start_time
        remaining = window - elapsed

        if remaining <= 0:
            print(f"[CHASING] Appearance #{chasing_count} deadline expired — "
                  f"50% speed penalty applied by game.")
            with data_lock:
                shared_data['chasing_active'] = False
                if shared_data.get('event_active') == 'chasing':
                    shared_data['event_active'] = None
        else:
            print(f"[CHASING] Appearance #{chasing_count} — {remaining:.1f}s remaining.")


# ---------------------------------------------------------
# Decision Task (HIGH priority, 5 ms) — the priority state machine.
# ---------------------------------------------------------
def decision_task():
    with data_lock:
        tokens         = shared_data.get('tokens', [])
        blanked_lanes  = shared_data.get('blanked_lanes', [False] * NUM_LANES)
        low_light      = shared_data.get('low_light_active', False)
        police_active  = shared_data.get('police_active', False)
        police_urgent  = shared_data.get('police_urgent', False)
        police_bbox    = shared_data.get('police_bbox', None)
        chasing_active = shared_data.get('chasing_active', False)
        chasing_count  = shared_data.get('chasing_appearance_count', 0)
        frame_w        = shared_data.get('frame_w', 640)
        frame_h        = shared_data.get('frame_h', 480)
        target_lane    = shared_data.get('target_lane', CENTER_LANE)
        commit_until   = shared_data.get('commit_until', 0.0)

    now   = time.time()
    mid_x = car_center_x(frame_w)        # our car sits at the road centre

    # STATE 1 — LOW_LIGHT: send ONLY acceleration=-1.0 for the whole duration.
    if low_light:
        with data_lock:
            shared_data['steering_input']     = 0.0
            shared_data['acceleration_input'] = -1.0
        return

    # Use the last committed target lane as the reference — not always CENTER_LANE.
    # The lane-change cost (SCORE_LANE_CHANGE) was previously computed against
    # CENTER_LANE even after the car had moved, making far lanes look artificially cheap.
    current_lane = target_lane

    # Resolve the active state (highest priority wins).
    if chasing_active:
        state = DriveState.CHASING_EVASION
    elif police_active:
        state = DriveState.POLICE
    else:
        danger_in_lane = any(
            t['lane'] == current_lane and t['color'] in ('red', 'yellow', 'unknown')
            for t in tokens
        )
        state = DriveState.COIN_AVOID if danger_in_lane else DriveState.NORMAL

    scores = score_lanes(tokens, police_bbox, blanked_lanes, current_lane,
                         state, police_urgent, frame_w, frame_h)

    full_commit = False
    if state == DriveState.CHASING_EVASION:
        # Penalise lanes near the chasing car's approach path (it follows directly
        # behind, so current lane and neighbours are most dangerous).
        for i in range(NUM_LANES):
            proximity = max(0.0, 1.0 - abs(i - current_lane) / max(1, NUM_LANES - 1))
            scores[i] -= 600.0 * proximity
        # Always commit immediately — the chasing car closes fast regardless of
        # which appearance this is. Waiting wastes the head-start.
        full_commit = True

    # Choose a target lane (hysteresis + in-progress commit).
    if now < commit_until and not full_commit:
        new_target = target_lane
    else:
        new_target = choose_target_lane(scores, target_lane)
        if new_target != target_lane:
            commit_until = now + LANE_COMMIT_S

    # Pure-pursuit steering toward the chosen lane at the lookahead row.
    aggressive = full_commit or state in (DriveState.CHASING_EVASION,
                                          DriveState.COIN_AVOID) or \
                 (state == DriveState.POLICE and police_urgent)
    gain = STEER_GAIN_AGGRO if aggressive else STEER_GAIN_NORMAL

    # Pure-pursuit lookahead: fixed fraction ahead (not nearest token).
    lookahead_y = int(DEFAULT_LOOKAHEAD_FRAC * frame_h)

    target_x = lane_center_at_row(new_target, lookahead_y, frame_w, frame_h)
    steer    = steer_toward(target_x, mid_x, frame_w, gain)
    accel    = 1.0

    with data_lock:
        shared_data['target_lane']        = new_target
        shared_data['target_x']           = int(target_x)
        shared_data['lookahead_y']        = int(lookahead_y)
        shared_data['commit_until']       = commit_until
        shared_data['lane_scores']        = scores
        shared_data['state']              = state
        shared_data['steering_input']     = steer
        shared_data['acceleration_input'] = accel


# ---------------------------------------------------------
# Send Controls Task
# ---------------------------------------------------------
def send_controls_task():
    if core.control_conn is None:
        return

    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']
        low_light_active   = shared_data['low_light_active']

    if not low_light_active:
        acceleration_input = 1.0

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        core.control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}. Game has ended - stopping agent.")
        core.control_conn = None
        core.is_running   = False
