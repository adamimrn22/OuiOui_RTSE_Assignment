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
                    SCORE_RED_TARGET, SWITCH_MARGIN, LANE_CHANGE_ESTIMATE_S,
                    GREEN_CLUSTER_BONUS, GREEN_CLUSTER_COUNT_BONUS,
                    RED_UNSAFE_PENALTY, YELLOW_SOFT_PENALTY, RED_UNSAFE_Y_FRAC,
                    POLICE_DEADLINE, POLICE_URGENT_THRESHOLD,
                    GOLDEN_LANE_SCORE_BOOST, GOLDEN_LANE_DURATION,
                    GOLDEN_LANE_ABSENT_GRACE, TACTICAL_NET_GREEN_TARGET)
import core
from core import shared_data, data_lock, DriveState
from lane_geometry import lane_of_x, lane_center_at_row, car_center_x, proximity_weight


# ---------------------------------------------------------
# Lane scoring core
# ---------------------------------------------------------
def _empty_lane_summary():
    n = NUM_LANES
    return {
        'green_count':     [0] * n,
        'red_count':       [0] * n,
        'yellow_count':    [0] * n,
        'nearest_red_y':   [0] * n,
        'nearest_green_y': [0] * n,
    }


def build_lane_summary(tokens, frame_w, frame_h):
    """Per-lane token counts and nearest y for cluster / unsafe logic."""
    summary = _empty_lane_summary()
    for t in tokens:
        lane = t.get('lane', lane_of_x(t['x'], t['y'], frame_w, frame_h))
        if not (0 <= lane < NUM_LANES):
            continue
        color = t['color']
        ty = t['y']
        if color == 'green':
            summary['green_count'][lane] += 1
            summary['nearest_green_y'][lane] = max(summary['nearest_green_y'][lane], ty)
        elif color == 'red' and not t.get('is_curb_like', False):
            summary['red_count'][lane] += 1
            summary['nearest_red_y'][lane] = max(summary['nearest_red_y'][lane], ty)
        elif color == 'yellow':
            summary['yellow_count'][lane] += 1
    return summary


def _apply_lane_summary_bonuses(scores, summary, frame_h):
    """Green cluster pull, near-red block, soft yellow — after token scoring."""
    near_red_y = RED_UNSAFE_Y_FRAC * frame_h
    for i in range(NUM_LANES):
        gc = summary['green_count'][i]
        rc = summary['red_count'][i]
        yc = summary['yellow_count'][i]

        if rc > 0 and summary['nearest_red_y'][i] > near_red_y:
            scores[i] -= RED_UNSAFE_PENALTY

        if gc >= 2 and rc == 0:
            bonus = GREEN_CLUSTER_BONUS + GREEN_CLUSTER_COUNT_BONUS * max(0, gc - 2)
            if yc > 0:
                bonus *= max(0.70, 1.0 - 0.12 * yc)
            scores[i] += bonus

        if yc > 0 and rc == 0:
            if gc >= 2:
                scores[i] -= YELLOW_SOFT_PENALTY * 0.35 * yc
            else:
                scores[i] -= YELLOW_SOFT_PENALTY * min(1.0, yc)


def score_lanes(tokens, police_bbox, blanked_lanes, current_lane,
                state, police_urgent, frame_w, frame_h):
    """Assign every lane a score; the caller picks the highest (with hysteresis)."""
    scores = [SCORE_CLEAR] * NUM_LANES
    summary = build_lane_summary(tokens, frame_w, frame_h)

    for i in range(NUM_LANES):
        scores[i] -= SCORE_LANE_CHANGE * abs(i - current_lane)

    red_is_target = (state == DriveState.POLICE)

    for t in tokens:
        lane = t.get('lane', lane_of_x(t['x'], t['y'], frame_w, frame_h))
        if not (0 <= lane < NUM_LANES):
            continue
        w    = proximity_weight(t, frame_h)
        color = t['color']

        if color == 'green':
            if not (state == DriveState.POLICE and police_urgent):
                scores[lane] += SCORE_GREEN_REWARD * max(GREEN_WEIGHT_FLOOR, w)
        elif color == 'red':
            if t.get('is_curb_like', False):
                continue
            if red_is_target:
                scores[lane] += SCORE_RED_TARGET * w
            else:
                scores[lane] -= SCORE_RED_PENALTY * w
        elif color == 'yellow':
            scores[lane] -= SCORE_YELLOW_PENALTY * w
        else:
            scores[lane] -= SCORE_UNKNOWN_PENALTY * w

    _apply_lane_summary_bonuses(scores, summary, frame_h)

    if police_bbox is not None:
        px, py, pw, ph = police_bbox
        ry = py + ph
        left_lane  = lane_of_x(px, ry, frame_w, frame_h)
        right_lane = lane_of_x(px + pw, ry, frame_w, frame_h)
        for i in range(min(left_lane, right_lane), max(left_lane, right_lane) + 1):
            if 0 <= i < NUM_LANES:
                scores[i] -= SCORE_POLICE_BLOB

    for i in range(NUM_LANES):
        if i < len(blanked_lanes) and blanked_lanes[i]:
            scores[i] -= SCORE_BLANKED_LANE

    # Golden Lane boost: strongly favour the target lane during the 5 s window
    with data_lock:
        gl_active = shared_data.get('golden_lane_active', False)
        gl_lane   = shared_data.get('golden_lane_number', -1)
    if gl_active and 0 <= gl_lane < NUM_LANES:
        scores[gl_lane] += GOLDEN_LANE_SCORE_BOOST

    return scores, summary


def choose_target_lane(scores, current_target):
    """Best lane, but only switch if it beats the current target by SWITCH_MARGIN."""
    best = int(np.argmax(scores))
    if best == current_target:
        return best
    if scores[best] > scores[current_target] + SWITCH_MARGIN:
        return best
    return current_target


def _lane_has_near_red(summary, lane, frame_h):
    """True when lane has a non-curb red token below the unsafe y threshold."""
    if not (0 <= lane < NUM_LANES):
        return False
    near_red_y = RED_UNSAFE_Y_FRAC * frame_h
    return (summary['red_count'][lane] > 0
            and summary['nearest_red_y'][lane] > near_red_y)


def _best_lane_without_near_red(scores, summary, frame_h):
    """Highest-scoring lane with no near red; fall back to global best if none."""
    safe = [i for i in range(NUM_LANES) if not _lane_has_near_red(summary, i, frame_h)]
    if not safe:
        return int(np.argmax(scores))
    return max(safe, key=lambda i: scores[i])


def _choose_chase_escape_lane(scores, summary, frame_h, from_lane):
    """Escape lane away from current position; prefer lanes without near red."""
    from_lane = _clamp_lane(from_lane)
    away = [i for i in range(NUM_LANES) if i != from_lane]
    if not away:
        away = list(range(NUM_LANES))
    safe_away = [i for i in away if not _lane_has_near_red(summary, i, frame_h)]
    pool = safe_away if safe_away else away
    if not pool:
        return int(np.argmax(scores))
    return max(pool, key=lambda i: (abs(i - from_lane), scores[i]))


def steer_toward(target_x, mid_x, frame_w, gain):
    """Proportional pure-pursuit steering toward target_x, with a centre dead-zone."""
    offset = target_x - mid_x
    if abs(offset) < frame_w * STEER_DEADZONE_FRAC:
        return 0.0
    return float(max(-1.0, min(1.0, offset / (frame_w * gain))))


def _clamp_lane(lane):
    return max(0, min(NUM_LANES - 1, int(lane)))


def _update_lane_estimate(now, target_lane, new_target,
                          estimated_lane, lane_from, lane_to, lane_change_start):
    """Debug-only interpolation of which physical lane the car thinks it occupies."""
    if not (0 <= estimated_lane < NUM_LANES):
        estimated_lane = target_lane if 0 <= target_lane < NUM_LANES else CENTER_LANE
    if not (0 <= lane_from < NUM_LANES):
        lane_from = estimated_lane
    if not (0 <= lane_to < NUM_LANES):
        lane_to = new_target if 0 <= new_target < NUM_LANES else CENTER_LANE

    if new_target != target_lane:
        lane_from = estimated_lane
        lane_to = new_target
        lane_change_start = now

    if lane_from == lane_to:
        progress = 1.0
        estimated_lane_float = float(lane_to)
        estimated_lane = lane_to
    else:
        progress = (now - lane_change_start) / max(LANE_CHANGE_ESTIMATE_S, 1e-3)
        progress = max(0.0, min(1.0, progress))
        if progress >= 1.0:
            estimated_lane = lane_to
            estimated_lane_float = float(lane_to)
            lane_from = lane_to
        else:
            estimated_lane_float = lane_from + (lane_to - lane_from) * progress
            estimated_lane = _clamp_lane(round(estimated_lane_float))

    lane_change_active = progress < 1.0 and lane_from != lane_to
    return (estimated_lane, estimated_lane_float, lane_from, lane_to,
            lane_change_start, progress, lane_change_active)


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
# Golden Lane Watchdog Task (LOW priority, 100 ms)
# Reads the raw HUD detection published by detection_task and manages
# the 5 s event window, pass/miss recording, and the Tactical win check.
# Mirrors police_watchdog_task in structure.
# ---------------------------------------------------------
_gl_last_print_s = -1   # throttle countdown prints to once per second

def golden_lane_watchdog_task():
    global _gl_last_print_s
    now = time.time()

    with data_lock:
        detected_raw  = shared_data.get('golden_lane_detected_raw',  False)
        detected_lane = shared_data.get('golden_lane_detected_lane', -1)
        gl_active     = shared_data['golden_lane_active']
        gl_lane       = shared_data['golden_lane_number']
        gl_start      = shared_data['golden_lane_start']
        gl_last_seen  = shared_data['golden_lane_last_seen']
        pass_count    = shared_data['golden_lane_pass_count']
        estimated_lane = shared_data.get('estimated_lane', CENTER_LANE)

    # --- New detection: start or refresh the window ---
    if detected_raw and detected_lane >= 0:
        with data_lock:
            shared_data['golden_lane_last_seen'] = now

        if not gl_active:
            with data_lock:
                shared_data['golden_lane_active'] = True
                shared_data['golden_lane_number'] = detected_lane
                shared_data['golden_lane_start']  = now
                shared_data['golden_lane_passed'] = False
                shared_data['event_active']        = 'golden_lane'
            print(f"[GOLDEN LANE] Lane {detected_lane + 1} detected! "
                  f"{GOLDEN_LANE_DURATION:.0f} s window started.")
            _gl_last_print_s = int(now)
        elif detected_lane != gl_lane:
            with data_lock:
                shared_data['golden_lane_number'] = detected_lane
            print(f"[GOLDEN LANE] Lane corrected to {detected_lane + 1}.")

    # --- Nothing detected this frame: apply grace period ---
    if not detected_raw and gl_active:
        if (now - gl_last_seen) > GOLDEN_LANE_ABSENT_GRACE:
            _expire_golden_window(now, estimated_lane, gl_lane, pass_count)
            return

    # --- Window active: check expiry and print countdown ---
    if not gl_active:
        return

    elapsed   = now - gl_start
    remaining = GOLDEN_LANE_DURATION - elapsed

    if remaining <= 0:
        _expire_golden_window(now, estimated_lane, gl_lane, pass_count)
        return

    # Countdown print once per second
    if int(now) != _gl_last_print_s:
        print(f"[GOLDEN LANE] Lane {gl_lane + 1} — {remaining:.1f} s remaining.")
        _gl_last_print_s = int(now)


def _expire_golden_window(now, estimated_lane, gl_lane, prev_pass_count):
    """Record pass/miss and clear the golden lane window. Then check Tactical win."""
    in_lane = (int(estimated_lane) == int(gl_lane)) if gl_lane >= 0 else False
    new_pass_count = prev_pass_count + (1 if in_lane else 0)

    with data_lock:
        shared_data['golden_lane_passed']     = in_lane
        shared_data['golden_lane_pass_count'] = new_pass_count
        shared_data['golden_lane_active']     = False
        shared_data['golden_lane_number']     = -1
        shared_data['event_active']           = None

    if in_lane:
        print(f"[GOLDEN LANE] PASSED! Car was in lane {gl_lane + 1}. "
              f"Total passes: {new_pass_count}.")
    else:
        print(f"[GOLDEN LANE] MISSED. Car was in lane {int(estimated_lane) + 1}, "
              f"needed lane {gl_lane + 1}.")

    # Tactical win check
    with data_lock:
        net    = shared_data['tactical_net_green']
        already_won = shared_data['tactical_win']
    if not already_won and net >= TACTICAL_NET_GREEN_TARGET and new_pass_count >= 1:
        with data_lock:
            shared_data['tactical_win'] = True
        print(f"[TACTICAL] WIN condition met! "
              f"Net green = {net} (>= {TACTICAL_NET_GREEN_TARGET}), "
              f"passes = {new_pass_count}.")


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
        chasing_active    = shared_data.get('chasing_active', False)
        chasing_count     = shared_data.get('chasing_appearance_count', 0)
        chasing_teal_area = shared_data.get('chasing_teal_area', 0.0)
        chasing_last_seen = shared_data.get('chasing_last_seen', 0.0)
        frame_w        = shared_data.get('frame_w', 640)
        frame_h        = shared_data.get('frame_h', 480)
        target_lane    = shared_data.get('target_lane', CENTER_LANE)
        commit_until   = shared_data.get('commit_until', 0.0)
        estimated_lane = shared_data.get('estimated_lane', target_lane)
        lane_from      = shared_data.get('lane_from', estimated_lane)
        lane_to        = shared_data.get('lane_to', target_lane)
        lane_change_start = shared_data.get('lane_change_start', 0.0)
        golden_lane_active = shared_data.get('golden_lane_active', False)
        golden_lane_number = shared_data.get('golden_lane_number', -1)
        golden_lane_start  = shared_data.get('golden_lane_start', 0.0)

    now   = time.time()
    mid_x = car_center_x(frame_w)        # our car sits at the road centre

    # STATE 1 — LOW_LIGHT: send ONLY acceleration=-1.0 for the whole duration.
    if low_light:
        with data_lock:
            shared_data['steering_input']     = 0.0
            shared_data['acceleration_input'] = -1.0
        return

    if 0 <= estimated_lane < NUM_LANES:
        current_lane = estimated_lane
    elif 0 <= target_lane < NUM_LANES:
        current_lane = target_lane
    else:
        current_lane = CENTER_LANE
    planning_lane = current_lane

    # Resolve the active state (highest priority wins).
    if chasing_active:
        state = DriveState.CHASING_EVASION
    elif police_active:
        state = DriveState.POLICE
    elif golden_lane_active and 0 <= golden_lane_number < NUM_LANES:
        # Golden Lane: priority below Police but above normal coin avoidance
        state = DriveState.GOLDEN_LANE
    else:
        danger_in_lane = any(
            t['lane'] == current_lane
            and t['color'] in ('red', 'unknown')
            and not t.get('is_curb_like', False)
            for t in tokens
        )
        state = DriveState.COIN_AVOID if danger_in_lane else DriveState.NORMAL

    scores, lane_summary = score_lanes(tokens, police_bbox, blanked_lanes, current_lane,
                                      state, police_urgent, frame_w, frame_h)

    full_commit = False
    chase_escape_reason = ''
    # AutoRS-inspired: as golden lane deadline approaches, amplify score boost
    # (analogous to γ increasing when tracking error rises).
    if state == DriveState.GOLDEN_LANE and 0 <= golden_lane_number < NUM_LANES:
        elapsed_gl = now - golden_lane_start
        remaining_gl = max(0.0, GOLDEN_LANE_DURATION - elapsed_gl)
        # Scale from 1× at 5 s remaining to 4× at 0 s remaining.
        pressure = 1.0 + 3.0 * max(0.0, (GOLDEN_LANE_DURATION - remaining_gl) / GOLDEN_LANE_DURATION)
        scores[golden_lane_number] += GOLDEN_LANE_SCORE_BOOST * (pressure - 1.0)

    # Post-evasion lane bias: keep mild pressure away from escape-from lane for 5s.
    chase_cleared_s = now - chasing_last_seen
    if not chasing_active and 0 < chase_cleared_s < 5.0:
        prev_escape = shared_data.get('chasing_escape_from_lane', current_lane)
        for i in range(NUM_LANES):
            prox = max(0.0, 1.0 - abs(i - prev_escape) / max(1, NUM_LANES - 1))
            scores[i] -= 150.0 * prox * (1.0 - chase_cleared_s / 5.0)

    # Police car collision avoidance: penalise police car's lane heavily.
    # Hitting the police car = GAME OVER, so avoid it unless we MUST collect red.
    if police_active and police_bbox is not None and not police_urgent:
        px, py, pw, ph = police_bbox
        ry = py + ph
        police_left  = lane_of_x(px,      ry, frame_w, frame_h)
        police_right = lane_of_x(px + pw, ry, frame_w, frame_h)
        for i in range(max(0, police_left - 1), min(NUM_LANES, police_right + 2)):
            scores[i] -= 2000.0  # heavy penalty — do not enter police car's lane

    if state == DriveState.CHASING_EVASION:
        # Proximity-based penalty — scale by blob area (bigger blob = harder push).
        area_scale = max(1.0, min(2.0, chasing_teal_area / 1500.0))
        for i in range(NUM_LANES):
            proximity = max(0.0, 1.0 - abs(i - current_lane) / max(1, NUM_LANES - 1))
            scores[i] -= 600.0 * proximity * area_scale
        # Cap all scores at 0 — green bonuses must never override escape.
        for i in range(NUM_LANES):
            scores[i] = min(scores[i], 0.0)
        full_commit = True   # commit on ALL appearances, not just 2nd
        chase_escape_reason = 'second_chase' if chasing_count >= 2 else 'first_chase'

    # Golden Lane hard commit: in the final 1.5 s of the window the car MUST be
    # in the golden lane when the timer expires — override all other lane choices.
    golden_hard_commit = False
    if state == DriveState.GOLDEN_LANE and 0 <= golden_lane_number < NUM_LANES:
        elapsed_gl   = now - golden_lane_start
        remaining_gl = max(0.0, GOLDEN_LANE_DURATION - elapsed_gl)
        if remaining_gl < 1.5:
            golden_hard_commit = True
            full_commit = True

    # Choose a target lane (hysteresis + in-progress commit).
    if state == DriveState.CHASING_EVASION:
        # Chasing: break commit, steer to an escape lane away from planning lane.
        escape_from = planning_lane
        new_target = _choose_chase_escape_lane(scores, lane_summary, frame_h, escape_from)
        commit_until = now + 3.0   # hold escape lane for 3s, not the normal LANE_COMMIT_S
    elif golden_hard_commit:
        # AutoRS analogue: tracking error high (deadline imminent) → lock control output.
        new_target   = golden_lane_number
        commit_until = now + 2.0
    else:
        # Break commit immediately if the committed lane gains a near red (not during POLICE).
        target_has_near_red = (
            state != DriveState.POLICE
            and _lane_has_near_red(lane_summary, target_lane, frame_h)
        )
        if target_has_near_red:
            new_target = _best_lane_without_near_red(scores, lane_summary, frame_h)
            commit_until = now + LANE_COMMIT_S
        elif now < commit_until and not full_commit:
            new_target = target_lane
        else:
            new_target = choose_target_lane(scores, target_lane)
            if new_target != target_lane:
                commit_until = now + LANE_COMMIT_S

    (estimated_lane, estimated_lane_float, lane_from, lane_to,
     lane_change_start, lane_change_progress, lane_change_active) = _update_lane_estimate(
        now, target_lane, new_target,
        estimated_lane, lane_from, lane_to, lane_change_start)

    # Pure-pursuit steering toward the chosen lane at the lookahead row.
    aggressive = full_commit or state in (DriveState.CHASING_EVASION,
                                          DriveState.COIN_AVOID,
                                          DriveState.GOLDEN_LANE) or \
                 (state == DriveState.POLICE and police_urgent)
    gain = STEER_GAIN_AGGRO if aggressive else STEER_GAIN_NORMAL

    # Fixed lookahead — aiming at the nearest token causes late jerky steering.
    lookahead_y = int(DEFAULT_LOOKAHEAD_FRAC * frame_h)

    target_x = lane_center_at_row(new_target, lookahead_y, frame_w, frame_h)
    steer    = steer_toward(target_x, mid_x, frame_w, gain)
    accel    = 1.0                              # full throttle except low light
    if state == DriveState.CHASING_EVASION:
        accel = 1.15                            # extra throttle to pull away from chasing car

    if chasing_active:
        chase_reaction = {
            'chasing_reaction_active': True,
            'chasing_reaction_state': DriveState.CHASING_EVASION,
            'chasing_escape_from_lane': current_lane,
            'chasing_escape_target_lane': new_target,
            'chasing_escape_full_commit': full_commit,
            'chasing_escape_steer': steer,
            'chasing_escape_reason': chase_escape_reason,
        }
    else:
        chase_reaction = {'chasing_reaction_active': False}

    with data_lock:
        shared_data['target_lane']            = new_target
        shared_data['target_x']               = int(target_x)
        shared_data['lookahead_y']            = int(lookahead_y)
        shared_data['commit_until']           = commit_until
        shared_data['lane_scores']            = scores
        shared_data['lane_summary']           = lane_summary
        shared_data['state']                  = state
        shared_data['planning_lane']          = planning_lane
        shared_data['estimated_lane']         = estimated_lane
        shared_data['estimated_lane_float']   = estimated_lane_float
        shared_data['lane_from']              = lane_from
        shared_data['lane_to']                = lane_to
        shared_data['lane_change_start']      = lane_change_start
        shared_data['lane_change_progress']   = lane_change_progress
        shared_data['lane_change_active']     = lane_change_active
        shared_data['steering_input']         = steer
        shared_data['acceleration_input']     = accel
        for key, val in chase_reaction.items():
            shared_data[key] = val
        # AutoRS outer-loop analogue: flag that chasing is active so detection
        # task can prioritise back-camera processing over front-camera tokens.
        shared_data['detection_urgent'] = chasing_active or police_urgent


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
