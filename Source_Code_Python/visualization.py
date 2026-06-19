"""
visualization.py — debug overlay drawn on the front camera.

Draws lane rays, token markers, lane-position debug, and per-lane scores.
"""
import cv2
import numpy as np

from config import (NUM_LANES, ROAD_HORIZON_FRAC, CENTER_LANE,
                    CHASING_MIN_AREA, CHASING_SOLIDITY_MIN)
from lane_geometry import road_polygon, road_bounds, car_center_x, lane_center_at_row
from token_detection import build_chasing_teal_mask

_COLOR_BY_TOKEN = {'green': (0, 255, 0), 'red': (0, 0, 255),
                   'yellow': (0, 255, 255), 'unknown': (200, 200, 200)}

# Lane debug colours (BGR)
_COL_TARGET    = (0, 255, 255)   # yellow — where the bot wants to go
_COL_ESTIMATED = (255, 255, 0)   # cyan — estimated physical lane
_COL_PLANNING  = (255, 255, 255) # white — lane used for scoring


def _human_lane(idx):
    if 0 <= idx < NUM_LANES:
        return f"L{idx + 1}"
    return "?"


def _read_lane_debug():
    """Read lane-position debug fields written by decision_task."""
    defaults = {
        'planning_lane': CENTER_LANE,
        'estimated_lane': CENTER_LANE,
        'estimated_lane_float': float(CENTER_LANE),
        'lane_from': CENTER_LANE,
        'lane_to': CENTER_LANE,
        'lane_change_progress': 1.0,
        'lane_change_active': False,
    }
    try:
        from core import shared_data, data_lock
        with data_lock:
            for key in defaults:
                if key in shared_data:
                    defaults[key] = shared_data[key]
    except Exception:
        pass
    return defaults


def _boundary_y_samples(frame_h, n=12):
    """Sample y rows from just below horizon to bottom — follows perspective on bends."""
    horizon = int(ROAD_HORIZON_FRAC * frame_h)
    y_start = horizon + int(0.05 * (frame_h - horizon))
    y_end = frame_h - 1
    return np.linspace(y_start, y_end, n).astype(int)


def draw_lane_boundaries(out, frame_w, frame_h):
    """6 boundary polylines for 5 lanes — debug only, not used for driving."""
    ys = _boundary_y_samples(frame_h)
    for b in range(NUM_LANES + 1):
        pts = []
        for y in ys:
            left, right = road_bounds(y, frame_w, frame_h)
            x = left + (b / NUM_LANES) * (right - left)
            pts.append([int(x), int(y)])
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        edge = b == 0 or b == NUM_LANES
        cv2.polylines(out, [poly], False,
                      (180, 180, 180) if edge else (90, 90, 90),
                      2 if edge else 1)


def draw_lane_center_polyline(out, lane, color, thickness, frame_w, frame_h):
    """Full lane-centre polyline for EST / PLAN / TARGET bend debugging."""
    ys = _boundary_y_samples(frame_h)
    pts = []
    for y in ys:
        x = int(lane_center_at_row(lane, y, frame_w, frame_h))
        pts.append([x, int(y)])
    poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [poly], False, color, thickness)


def _read_chase_debug():
    """Read chasing-car debug fields written by detection_task."""
    defaults = {
        'chasing_raw_detected': False,
        'chasing_detected': False,
        'chasing_teal_area': 0.0,
        'chasing_teal_solidity': 0.0,
        'chasing_bbox': None,
        'chasing_active': False,
        'chasing_appearance_count': 0,
        'chasing_mask_roi_top': 0,
        'chasing_mask_roi_left': 0,
        'chasing_mask_roi_right': 0,
        'chasing_reaction_active': False,
        'chasing_reaction_state': '',
        'chasing_escape_from_lane': CENTER_LANE,
        'chasing_escape_target_lane': CENTER_LANE,
        'chasing_escape_full_commit': False,
        'chasing_escape_steer': 0.0,
        'chasing_escape_reason': '',
    }
    try:
        from core import shared_data, data_lock
        with data_lock:
            for key in defaults:
                if key in shared_data:
                    defaults[key] = shared_data[key]
    except Exception:
        pass
    return defaults


def _read_golden_debug():
    """Read golden-lane and tactical fields from shared_data."""
    defaults = {
        'golden_lane_active':       False,
        'golden_lane_number':       -1,
        'golden_lane_start':        0.0,
        'golden_lane_passed':       False,
        'golden_lane_pass_count':   0,
        'tactical_green_collected': 0,
        'tactical_red_collected':   0,
        'tactical_net_green':       0,
        'tactical_win':             False,
    }
    try:
        from core import shared_data, data_lock
        with data_lock:
            for key in defaults:
                if key in shared_data:
                    defaults[key] = shared_data[key]
    except Exception:
        pass
    return defaults


def draw_back_debug(frame):
    """Back-camera overlay for chasing-car detection debug."""
    if frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    chase = _read_chase_debug()

    roi_top = chase['chasing_mask_roi_top'] or int(h * 0.25)
    roi_left = chase['chasing_mask_roi_left'] or int(w * 0.25)
    roi_right = chase['chasing_mask_roi_right'] or int(w * 0.75)

    teal_mask = build_chasing_teal_mask(frame)
    if teal_mask is not None and np.any(teal_mask):
        tint = np.zeros_like(out)
        tint[:, :, 0] = 180
        tint[:, :, 1] = 255
        tint[:, :, 2] = 255
        active = teal_mask > 0
        out[active] = cv2.addWeighted(out, 0.55, tint, 0.45, 0)[active]

    cv2.rectangle(out, (roi_left, roi_top), (roi_right, h - 1), (120, 120, 120), 1)
    cv2.putText(out, "CHASE ROI", (roi_left + 4, max(roi_top + 14, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    bbox = chase['chasing_bbox']
    passed = chase['chasing_detected']
    raw = chase['chasing_raw_detected']
    if bbox is not None:
        x, y, bw, bh = bbox
        if passed:
            box_col = (255, 255, 0)   # cyan — passed detection
            label = "CHASE CAR"
        else:
            box_col = (0, 165, 255)   # orange — rejected contour
            reason = 'area' if chase['chasing_teal_area'] < CHASING_MIN_AREA else 'solidity'
            label = f"REJECT ({reason})"
        cv2.rectangle(out, (x, y), (x + bw, y + bh), box_col, 2)
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA)

    y_text = 22
    lines = [
        f"CHASE raw={raw}  active={chase['chasing_active']}",
        f"area={int(chase['chasing_teal_area'])}  solidity={chase['chasing_teal_solidity']:.2f}",
        f"count={chase['chasing_appearance_count']}  min_area={CHASING_MIN_AREA}  min_sol={CHASING_SOLIDITY_MIN:.2f}",
    ]
    for line in lines:
        cv2.putText(out, line, (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y_text += 18

    if passed:
        cv2.putText(out, "CHASING DETECTED", (8, y_text + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        y_text += 28
    elif raw:
        cv2.putText(out, "TEAL BLOB (rejected)", (8, y_text + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        y_text += 24

    if chase['chasing_active']:
        cv2.putText(out, "EVASION ACTIVE", (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(out, "BACK CAM DEBUG", (w - 155, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def draw_debug(frame, tokens, scores, state, target_lane, target_x, lookahead_y):
    out  = frame.copy()
    h, w = out.shape[:2]
    horizon = int(ROAD_HORIZON_FRAC * h)
    dbg = _read_lane_debug()
    chase = _read_chase_debug()

    planning_lane = dbg['planning_lane']
    estimated_lane = dbg['estimated_lane']
    estimated_float = dbg['estimated_lane_float']
    lane_from = dbg['lane_from']
    lane_to = dbg['lane_to']
    change_pct = int(dbg['lane_change_progress'] * 100)
    changing = dbg['lane_change_active']

    # Road trapezoid
    cv2.polylines(out, [road_polygon(w, h)], True, (120, 120, 120), 1)

    cx = car_center_x(w)
    car_y = h - 1
    y_fan = int(horizon + 0.10 * (h - horizon))

    # ---- Full lane boundary polylines (horizon → bottom) ----
    draw_lane_boundaries(out, w, h)

    # ---- Full lane-centre polylines for EST / PLAN / TARGET ----
    if 0 <= planning_lane < NUM_LANES:
        draw_lane_center_polyline(out, planning_lane, _COL_PLANNING, 1, w, h)
    est_lane_draw = estimated_float if changing else float(estimated_lane)
    draw_lane_center_polyline(out, est_lane_draw, _COL_ESTIMATED, 2, w, h)
    if 0 <= target_lane < NUM_LANES:
        draw_lane_center_polyline(out, target_lane, _COL_TARGET, 2, w, h)

    # Lane labels L1–L5 at bottom
    for i in range(NUM_LANES):
        cx_lane = int(lane_center_at_row(i, h - 18, w, h))
        col = (0, 255, 255) if i == target_lane else (160, 160, 160)
        cv2.putText(out, f"L{i + 1}", (cx_lane - 12, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    cv2.putText(out, "LANE MODEL DEBUG", (w - 175, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ---- Lane rays coloured by token content ----
    lane_has = [None] * NUM_LANES
    for t in tokens:
        ln = t.get('lane', 0)
        if 0 <= ln < NUM_LANES:
            cur = lane_has[ln]
            order = {'yellow': 3, 'red': 2, 'green': 1, 'unknown': 0}
            if cur is None or order.get(t['color'], 0) > order.get(cur, 0):
                lane_has[ln] = t['color']
    for i in range(NUM_LANES):
        end_x = int(lane_center_at_row(i, y_fan, w, h))
        c = _COLOR_BY_TOKEN.get(lane_has[i], (150, 150, 150)) if lane_has[i] else (150, 150, 150)
        thick = 3 if i == target_lane else 1
        cv2.line(out, (cx, car_y), (end_x, y_fan), c, thick)

    # ---- Lane position debug markers ----
    def _draw_lane_marker(lane_idx, colour, label, y_row, thick=2):
        if not (0 <= lane_idx < NUM_LANES):
            return
        ex = int(lane_center_at_row(lane_idx, y_row, w, h))
        cv2.line(out, (cx, car_y), (ex, y_row), colour, thick)
        cv2.putText(out, label, (ex - 18, y_row - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    # Planning lane (scoring reference) — white, mid fan
    _draw_lane_marker(planning_lane, _COL_PLANNING, f"PLAN {_human_lane(planning_lane)}",
                      y_fan - 8, thick=1)

    # Estimated current lane — cyan; use float position when mid-change
    est_lane_draw = estimated_float if changing else float(estimated_lane)
    est_x = int(lane_center_at_row(est_lane_draw, y_fan + 12, w, h))
    cv2.line(out, (cx, car_y), (est_x, y_fan + 12), _COL_ESTIMATED, 2)
    cv2.putText(out, f"EST {_human_lane(estimated_lane)}", (est_x - 22, y_fan + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _COL_ESTIMATED, 1, cv2.LINE_AA)

    # Target lane — yellow, lower fan
    _draw_lane_marker(target_lane, _COL_TARGET, f"TGT {_human_lane(target_lane)}",
                      y_fan + 28, thick=2)

    # Pure-pursuit target marker
    if target_x:
        cv2.circle(out, (int(target_x), int(lookahead_y)), 7, (255, 255, 0), 2)

    # Detected coins
    for t in tokens:
        conf = t.get('confidence', 1.0)
        ln = t.get('lane', CENTER_LANE)
        if t.get('is_curb_like'):
            col = (255, 0, 255)
            label = f"curb? {conf:.1f} {_human_lane(ln)}"
        else:
            col = _COLOR_BY_TOKEN.get(t['color'], (255, 255, 255))
            label = f"{conf:.1f} {_human_lane(ln)}"
        cv2.circle(out, (t['x'], t['y']), max(5, int((t['area'] ** 0.5) / 2)), col, 2)
        cv2.putText(out, label, (t['x'] + 6, t['y'] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    # ---- Top-left lane debug text ----
    line1 = (f"EST={_human_lane(estimated_lane)}  "
             f"PLAN={_human_lane(planning_lane)}  "
             f"TARGET={_human_lane(target_lane)}")
    cv2.putText(out, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    if changing:
        line2 = f"CHANGE: {_human_lane(lane_from)}->{_human_lane(lane_to)}  {change_pct}%"
        cv2.putText(out, line2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _COL_ESTIMATED, 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"{state}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    chase_line = (
        f"CHASE raw={chase['chasing_raw_detected']} "
        f"active={chase['chasing_active']} "
        f"area={int(chase['chasing_teal_area'])} "
        f"sol={chase['chasing_teal_solidity']:.2f}"
    )
    react_active = chase['chasing_reaction_active']
    escape_from = chase['chasing_escape_from_lane']
    escape_to = chase['chasing_escape_target_lane']
    escape_steer = chase['chasing_escape_steer']
    escape_reason = chase['chasing_escape_reason']
    count = chase['chasing_appearance_count']
    detected = chase['chasing_detected'] or chase['chasing_raw_detected']
    chasing_on = chase['chasing_active']

    if chasing_on and react_active and escape_to != escape_from:
        chase_col = (0, 255, 0)       # green — detected and reacting
    elif detected and chasing_on and not react_active:
        chase_col = (0, 255, 255)     # yellow — detected, not reacting
    elif chasing_on and (escape_to == escape_from or not react_active):
        chase_col = (0, 0, 255)       # red — active but no escape
    else:
        chase_col = (255, 200, 100)

    y_chase = 58
    cv2.putText(out, chase_line, (8, y_chase), cv2.FONT_HERSHEY_SIMPLEX, 0.42, chase_col, 1, cv2.LINE_AA)
    y_chase += 18
    cv2.putText(out, f"CHASE REACT={react_active}  state={chase['chasing_reaction_state'] or state}",
                (8, y_chase), cv2.FONT_HERSHEY_SIMPLEX, 0.42, chase_col, 1, cv2.LINE_AA)
    y_chase += 18
    cv2.putText(out, f"ESCAPE {_human_lane(escape_from)} -> {_human_lane(escape_to)}  ({escape_reason})",
                (8, y_chase), cv2.FONT_HERSHEY_SIMPLEX, 0.42, chase_col, 1, cv2.LINE_AA)
    y_chase += 18
    full_c = chase['chasing_escape_full_commit']
    cv2.putText(out, f"STEER={escape_steer:+.2f}  FULL={full_c}  COUNT={count}",
                (8, y_chase), cv2.FONT_HERSHEY_SIMPLEX, 0.42, chase_col, 1, cv2.LINE_AA)

    # Near car sprite
    car_label_y = int(h * 0.82)
    cv2.putText(out, f"EST LANE: {_human_lane(estimated_lane)}",
                (max(8, cx - 70), car_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_ESTIMATED, 2, cv2.LINE_AA)

    if scores:
        cv2.putText(out, "scores: " + " ".join(f"{s:.0f}" for s in scores),
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ----------------------------------------------------------------
    # Golden Lane overlay
    # ----------------------------------------------------------------
    gold = _read_golden_debug()
    import time as _time
    now_viz = _time.time()

    if gold['golden_lane_active']:
        gl_lane    = gold['golden_lane_number']
        gl_start   = gold['golden_lane_start']
        remaining  = max(0.0, 5.0 - (now_viz - gl_start))

        # Tint the entire golden-lane column with a semi-transparent green flash
        if 0 <= gl_lane < NUM_LANES:
            from config import GOLDEN_LANE_HUD_BOT_FRAC
            from lane_geometry import road_bounds
            tint_ys = np.linspace(int(ROAD_HORIZON_FRAC * h), h - 1, 30).astype(int)
            gl_pts = []
            for yy in tint_ys:
                left_b, right_b = road_bounds(yy, w, h)
                lane_w = (right_b - left_b) / NUM_LANES
                xl = int(left_b + gl_lane * lane_w)
                xr = int(left_b + (gl_lane + 1) * lane_w)
                gl_pts += [(xl, yy), (xr, yy)]
            # Draw translucent green strip by blending a filled polygon
            overlay = out.copy()
            poly_pts = []
            for yy in tint_ys:
                left_b, right_b = road_bounds(yy, w, h)
                lane_w = (right_b - left_b) / NUM_LANES
                xl = int(left_b + gl_lane * lane_w)
                poly_pts.append([xl, int(yy)])
            for yy in reversed(tint_ys):
                left_b, right_b = road_bounds(yy, w, h)
                lane_w = (right_b - left_b) / NUM_LANES
                xr = int(left_b + (gl_lane + 1) * lane_w)
                poly_pts.append([xr, int(yy)])
            poly_arr = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly_arr], (0, 220, 80))
            out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)

        # Golden Lane countdown banner (centered, bright gold text)
        banner = f"GOLDEN LANE {_human_lane(gl_lane)}  {remaining:.1f}s"
        (tw, th_), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        bx = (w - tw) // 2
        by = int(h * 0.22)
        cv2.rectangle(out, (bx - 8, by - th_ - 6), (bx + tw + 8, by + 6), (0, 0, 0), -1)
        cv2.putText(out, banner, (bx, by), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, (0, 220, 255), 2, cv2.LINE_AA)

    # ----------------------------------------------------------------
    # Tactical win condition status (bottom-right corner)
    # ----------------------------------------------------------------
    net        = gold['tactical_net_green']
    pass_count = gold['golden_lane_pass_count']
    win        = gold['tactical_win']
    tac_col    = (0, 255, 120) if win else (200, 200, 100)
    tac_label  = "TACTICAL WIN!" if win else f"TACTICAL: net={net:+d}/60  passes={pass_count}"
    (ttw, tth_), _ = cv2.getTextSize(tac_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(out, tac_label, (w - ttw - 8, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, tac_col, 1, cv2.LINE_AA)

    return out
