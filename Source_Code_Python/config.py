"""
config.py — shared constants for the RTSE controller.

All tunable numbers live here so every module reads the same values.
"""

# ---------------------------------------------------------
# Network configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# ---------------------------------------------------------
# Detection backend
# ---------------------------------------------------------
# False -> HSV/contour detection (no extra deps, default).
# True  -> YOLOv8 detection (needs a trained model; see yolo/README.md).
USE_YOLO        = False
YOLO_MODEL_PATH = 'yolo/runs/detect/rtse_yolo/weights/best.pt'
YOLO_CONF       = 0.35

# ---------------------------------------------------------
# Lane model (5 lanes across the PERSPECTIVE road trapezoid)
# All fractions are of frame width/height -> resolution independent.
# ---------------------------------------------------------
NUM_LANES        = 5
CENTER_LANE      = 2          # our car sits in the centre lane at the bottom
ROAD_HORIZON_FRAC = 0.43      # y of the horizon / vanishing line
ROAD_CENTER_FRAC  = 0.47      # x of the road centre (vanishing point + our car)
ROAD_TOP_HALF_W   = 0.05      # half road-width at the horizon (narrow)
ROAD_BOT_LEFT     = 0.17      # asphalt left edge at the bottom (inside the rumble)
ROAD_BOT_RIGHT    = 0.77      # asphalt right edge at the bottom (inside the rumble)

# Our own (red) car sprite — blanked out of detection so it isn't read as a coin.
CAR_MASK_TOP_FRAC = 0.78
CAR_MASK_X0_FRAC  = 0.34
CAR_MASK_X1_FRAC  = 0.66

# ---------------------------------------------------------
# Steering shaping (pure-pursuit; ego-centric game)
# ---------------------------------------------------------
STEER_DEADZONE_FRAC = 0.02
STEER_GAIN_NORMAL   = 0.20
STEER_GAIN_AGGRO    = 0.10
DEFAULT_LOOKAHEAD_FRAC = 0.60
LANE_COMMIT_S       = 0.45
LANE_CHANGE_ESTIMATE_S = 0.55   # debug: time to interpolate estimated_lane during a switch

# ---------------------------------------------------------
# Lane scoring weights
# ---------------------------------------------------------
SCORE_CLEAR          =   60.0
SCORE_LANE_CHANGE    =   38.0
SCORE_GREEN_REWARD   =  420.0
GREEN_WEIGHT_FLOOR   =   0.8
SCORE_RED_PENALTY    =  720.0
SCORE_YELLOW_PENALTY =  220.0
SCORE_UNKNOWN_PENALTY=  160.0
SCORE_POLICE_BLOB    = 1000.0
SCORE_BLANKED_LANE   =  130.0
SCORE_RED_TARGET     =  260.0
SWITCH_MARGIN        =   70.0

# Lane-first bonuses (applied after per-token scoring)
GREEN_CLUSTER_BONUS       = 650.0
GREEN_CLUSTER_COUNT_BONUS = 200.0   # per green beyond the 2nd in a lane
RED_UNSAFE_PENALTY        = 1400.0
YELLOW_SOFT_PENALTY       = 120.0
RED_UNSAFE_Y_FRAC         = 0.55

# ---------------------------------------------------------
# Token (coin) detection
# ---------------------------------------------------------
TOKEN_MIN_AREA       = 30
GRAY_MIN_AREA        = 120
TOKEN_MIN_EXTENT     = 0.30
TOKEN_AR_LO, TOKEN_AR_HI = 0.4, 2.6

# Challenge 1 — Low Light
LOW_BRIGHTNESS_THRESHOLD = 25
BLANK_LANE_THRESHOLD     = 12

# Challenge 2 — Chasing Car (TEAL/CYAN car in the back camera)
CHASING_MIN_AREA        = 1500
CHASING_GRACE_S         = 1.0
CHASING_2ND_WINDOW      = 3.0
CHASING_ROI_TOP_FRAC    = 0.25   # ignore top 25% of back frame
CHASING_ROI_LEFT_FRAC   = 0.25   # ignore left 25%
CHASING_ROI_RIGHT_FRAC  = 0.75   # active ROI ends here (ignore right 25%)
CHASING_SOLIDITY_MIN    = 0.40

# Challenge 3 — Police Car (collect a RED token before the deadline)
POLICE_DEADLINE          = 10.0
POLICE_URGENT_THRESHOLD  = 3.0
POLICE_MIN_AREA          = 2000
POLICE_ABSENT_GRACE      = 1.5

# Golden Lane event
# The game displays bright yellow text "LANE N — ALL GREEN!" in the top HUD strip.
# We detect the yellow-pixel centroid in that strip, then parse the digit.
GOLDEN_LANE_DURATION     = 5.0     # window length in seconds
GOLDEN_LANE_HUD_TOP_FRAC = 0.00    # top of HUD scan band (fraction of frame height)
GOLDEN_LANE_HUD_BOT_FRAC = 0.15    # bottom of HUD scan band
GOLDEN_LANE_SCORE_BOOST  = 3000.0  # added to the target lane score during the window
GOLDEN_LANE_ABSENT_GRACE = 0.5     # seconds to hold detection after text disappears

# Tactical win condition
TACTICAL_NET_GREEN_TARGET = 60     # net (green collected − red collected) to win
