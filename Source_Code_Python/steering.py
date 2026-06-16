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
                    POLICE_DEADLINE, POLICE_URGENT_THRESHOLD)
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
# Police Watchdog Task (LOW priority, 100 ms) — tracks the 10s deadline.
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
        with data_lock:
            shared_data['police_active'] = False
            shared_data['police_urgent'] = False
            shared_data['police_bbox']   = None
            shared_data['event_active']  = None
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

    current_lane = CENTER_LANE

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
        scores[current_lane] -= 500.0          # do not sit in the line of fire
        if chasing_count >= 2:                  # 2nd appearance: only ~3s to react
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

    if tokens:
        lookahead_y = max(t['y'] for t in tokens)
    else:
        lookahead_y = int(DEFAULT_LOOKAHEAD_FRAC * frame_h)

    target_x = lane_center_at_row(new_target, lookahead_y, frame_w, frame_h)
    steer    = steer_toward(target_x, mid_x, frame_w, gain)
    accel    = 1.0                              # full throttle except low light

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
# Send Controls Task — packs and sends the two floats to the game.
# (control_conn lives in core and is reassigned to None on send failure.)
# ---------------------------------------------------------
def send_controls_task():
    if core.control_conn is None:
        return

    with data_lock:
        steering_input     = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']
        low_light_active   = shared_data['low_light_active']

    # In low light, decision_task set -1.0 and we must NOT override it.
    if not low_light_active:
        acceleration_input = 1.0

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        core.control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        core.control_conn = None
