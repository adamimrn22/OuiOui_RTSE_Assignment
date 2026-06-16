"""
lane_geometry.py — perspective road/lane math (pure functions, no shared state).
Shared helper used by detection, steering and visualization.

The road is a trapezoid: narrow at the horizon, wide at the bottom. These helpers
map points to lanes and lanes to pixel positions, accounting for perspective.
"""
import numpy as np
from config import (NUM_LANES, CENTER_LANE, ROAD_HORIZON_FRAC, ROAD_CENTER_FRAC,
                    ROAD_TOP_HALF_W, ROAD_BOT_LEFT, ROAD_BOT_RIGHT)


def road_bounds(y, frame_w, frame_h):
    """Left/right road edge x at image row y, accounting for perspective."""
    horizon = ROAD_HORIZON_FRAC * frame_h
    t = (y - horizon) / max(1.0, frame_h - horizon)
    t = max(0.0, min(1.0, t))
    cx = ROAD_CENTER_FRAC * frame_w
    left_top  = cx - ROAD_TOP_HALF_W * frame_w
    right_top = cx + ROAD_TOP_HALF_W * frame_w
    left_bot  = ROAD_BOT_LEFT  * frame_w
    right_bot = ROAD_BOT_RIGHT * frame_w
    left  = left_top  + t * (left_bot  - left_top)
    right = right_top + t * (right_bot - right_top)
    return left, right


def lane_of_x(x, y, frame_w, frame_h):
    """Map a point to a lane 0..NUM_LANES-1 using the road bounds at its row."""
    left, right = road_bounds(y, frame_w, frame_h)
    if right <= left:
        return CENTER_LANE
    frac = (x - left) / (right - left)
    lane = int(frac * NUM_LANES)
    return max(0, min(NUM_LANES - 1, lane))


def lane_center_x(lane, frame_w):
    """Pixel x of a lane centre at the BOTTOM of the frame (debug/reference)."""
    left  = ROAD_BOT_LEFT  * frame_w
    right = ROAD_BOT_RIGHT * frame_w
    return int(left + (lane + 0.5) * (right - left) / NUM_LANES)


def lane_center_at_row(lane, y, frame_w, frame_h):
    """Pixel x of a lane centre at image row y (the pure-pursuit steering target)."""
    left, right = road_bounds(y, frame_w, frame_h)
    return left + (lane + 0.5) * (right - left) / NUM_LANES


def car_center_x(frame_w):
    """x of our car at the bottom = road centre."""
    return int(ROAD_CENTER_FRAC * frame_w)


def road_polygon(frame_w, frame_h):
    """Trapezoid covering the drivable road — used to mask out grass/sky/HUD."""
    horizon = int(ROAD_HORIZON_FRAC * frame_h)
    lt, rt = road_bounds(horizon, frame_w, frame_h)
    lb, rb = road_bounds(frame_h, frame_w, frame_h)
    pad = int(0.02 * frame_w)   # small pad so edge-lane coins aren't clipped
    return np.array([[int(lt) - pad, horizon], [int(rt) + pad, horizon],
                     [int(rb) + pad, frame_h], [int(lb) - pad, frame_h]], np.int32)


def proximity_weight(token, frame_h):
    """Nearer tokens matter more (lower in frame / larger area => amplified)."""
    y_factor    = 0.4 + (token['y'] / max(1, frame_h))        # 0.4 .. 1.4
    area_factor = 1.0 + min(1.0, token['area'] / 4000.0)      # 1.0 .. 2.0
    return y_factor * area_factor
