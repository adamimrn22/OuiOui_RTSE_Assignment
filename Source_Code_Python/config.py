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
LANE_COMMIT_S       = 0.35

# ---------------------------------------------------------
# Lane scoring weights
# ---------------------------------------------------------
SCORE_CLEAR          =   60.0
SCORE_LANE_CHANGE    =   25.0   # moderate: switches for green but resists random drifts
SCORE_GREEN_REWARD   =  280.0   # raised: green tokens are the primary objective
GREEN_WEIGHT_FLOOR   =   0.7
SCORE_RED_PENALTY    =  220.0
SCORE_YELLOW_PENALTY =  480.0   # raised: avoid yellow harder
SCORE_UNKNOWN_PENALTY=  140.0
SCORE_POLICE_BLOB    = 1000.0
SCORE_BLANKED_LANE   =  130.0
SCORE_RED_TARGET     =  260.0
SWITCH_MARGIN        =   30.0   # slight hysteresis prevents rapid lane oscillation

# ---------------------------------------------------------
# Token (coin) detection
# ---------------------------------------------------------
TOKEN_MIN_AREA       = 20   # lowered: detect smaller/farther tokens earlier
GRAY_MIN_AREA        = 120
TOKEN_MIN_EXTENT     = 0.30
TOKEN_AR_LO, TOKEN_AR_HI = 0.4, 2.6

# Challenge 1 — Low Light
LOW_BRIGHTNESS_THRESHOLD = 25
BLANK_LANE_THRESHOLD     = 12

# Challenge 2 — Chasing Car (TEAL/CYAN car in the back camera)
CHASING_MIN_AREA   = 700    # lowered: detect chasing car earlier
CHASING_GRACE_S    = 1.0
CHASING_1ST_WINDOW = 10.0   # seconds to survive the 1st chasing-car appearance
CHASING_2ND_WINDOW = 3.0    # seconds to survive the 2nd appearance (tighter deadline)

# Challenge 3 - Police Car (collect a RED token before the deadline)
POLICE_DEADLINE          = 10.0
POLICE_URGENT_THRESHOLD  = 3.0
POLICE_MIN_AREA          = 2000
POLICE_ABSENT_GRACE      = 1.5

# Low Light - consecutive dark frames required before activating
DARK_FRAME_REQUIRED = 3

# Lane geometry
PROXIMITY_AREA_NORM = 4000.0
