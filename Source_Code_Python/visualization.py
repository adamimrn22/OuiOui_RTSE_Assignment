"""
visualization.py — debug overlay drawn on the front camera.

Draws the road trapezoid, the 5 lane "rays" (one per lane, coloured by what's in
that lane), the detected coins, the pure-pursuit target, the current state and the
per-lane scores — so you can SEE what the AI perceives and why it steers.
"""
import cv2

from config import NUM_LANES, ROAD_HORIZON_FRAC
from lane_geometry import road_polygon, road_bounds, car_center_x, lane_center_at_row

_COLOR_BY_TOKEN = {'green': (0, 255, 0), 'red': (0, 0, 255),
                   'yellow': (0, 255, 255), 'unknown': (200, 200, 200)}


def draw_debug(frame, tokens, scores, state, target_lane, target_x, lookahead_y):
    out  = frame.copy()
    h, w = out.shape[:2]
    horizon = int(ROAD_HORIZON_FRAC * h)

    # Road trapezoid
    cv2.polylines(out, [road_polygon(w, h)], True, (120, 120, 120), 1)

    # ---- 5 lane "rays" (one per lane) from the car up to near the horizon ----
    #   green = green coin (go!)   red/yellow = danger   gray = clear
    cx = car_center_x(w)
    y_fan = int(horizon + 0.10 * (h - horizon))     # below horizon so lanes separate
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
        cv2.line(out, (cx, h - 1), (end_x, y_fan), c, thick)

    # Pure-pursuit target marker
    if target_x:
        cv2.circle(out, (int(target_x), int(lookahead_y)), 7, (255, 255, 0), 2)
    # Detected coins
    for t in tokens:
        col = _COLOR_BY_TOKEN.get(t['color'], (255, 255, 255))
        cv2.circle(out, (t['x'], t['y']), max(5, int((t['area'] ** 0.5) / 2)), col, 2)
    # Text: state + per-lane scores
    cv2.putText(out, f"{state}  target_lane={target_lane}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if scores:
        cv2.putText(out, "scores: " + " ".join(f"{s:.0f}" for s in scores),
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
