"""Central configuration for the Indian Traffic Intelligent Car (DQN) simulator.

All tunable numbers live here so that the environment, the agent and the
training loop never contain magic constants.  World units are pixels.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_best.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_last.pth")
SAVE_EVERY_EPISODES = 25

# ---------------------------------------------------------------------------
# Training hyper-parameters (as requested)
# ---------------------------------------------------------------------------
EPISODES = 200
MAX_STEPS = 500
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995          # applied once per episode
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10        # episodes between target-network syncs

# Extra DQN settings
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000         # transitions collected before learning starts
HIDDEN_SIZE = 128
GRAD_CLIP_NORM = 10.0
REWARD_SCALE = 0.05            # rewards are multiplied by this inside the loss only
SEED = 42

# ---------------------------------------------------------------------------
# State / action spaces
# ---------------------------------------------------------------------------
STATE_SIZE = 8                 # [car_x, car_y, front_distance, left_gap, right_gap, current_lane, speed, closing_speed]
ACTION_SIZE = 7
NORMALIZE_STATE = True         # False returns raw pixel values instead of normalised values

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_DIAGONAL_LEFT = 3
ACTION_DIAGONAL_RIGHT = 4
ACTION_SLOW = 5                # new: ease off the throttle / gentle brake
ACTION_STOP = 6                # new: hard brake until the car stands still

ACTION_NAMES = {
    0: "LEFT",
    1: "RIGHT",
    2: "FORWARD",
    3: "DIAG-LEFT",
    4: "DIAG-RIGHT",
    5: "SLOW",
    6: "STOP",
}

# Relative probability of each action when epsilon-greedy explores at random.
# Favouring FORWARD and keeping STOP rare stops the untrained car from parking itself.
EXPLORE_WEIGHTS = [1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 0.5]

# ---------------------------------------------------------------------------
# Ego car dynamics (speeds in px per step)
# ---------------------------------------------------------------------------
EGO_CRUISE_SPEED = 7.0         # speed the car accelerates towards on every non-braking action
EGO_START_SPEED = 7.0
EGO_SLOW_SPEED = 2.0           # crawl speed the SLOW action eases down to
EGO_ACCEL = 0.4                # speed gained per step
EGO_SLOW_DECEL = 0.6           # speed lost per step on SLOW
EGO_BRAKE_DECEL = 1.5          # speed lost per step on STOP
EGO_STEER_FULL_SPEED = 2.0     # at or above this speed the car steers at full rate
EGO_MIN_STEER = 0.35           # fraction of the steering rate still available when nearly stopped

# action -> (lateral direction, lateral speed px/step, speed mode)
# speed modes: "cruise" accelerate to cruise speed, "slow" ease down to crawl speed,
# "stop" brake to a standstill, "hold" keep the current speed (used by the safety layer).
ACTION_EFFECTS = {
    0: (-1, 7.0, "cruise"),
    1: (1, 7.0, "cruise"),
    2: (0, 0.0, "cruise"),
    3: (-1, 5.0, "cruise"),
    4: (1, 5.0, "cruise"),
    5: (0, 0.0, "slow"),
    6: (0, 0.0, "stop"),
}

# ---------------------------------------------------------------------------
# Safety layer: predicts the next step and overrides actions that would hit something.
# Fallback order: dodge while holding speed, dodge while slowing, dodge while stopping,
# then straight slow, then straight stop.
# ---------------------------------------------------------------------------
SAFETY_SHIELD = True
SHIELD_MARGIN = 6.0            # extra bumper gap (px) that must remain after braking distance
SHIELD_SIDE_MARGIN = 2.0       # sideways safety buffer (px) around the car
SHIELD_PENALTY = -2.0          # reward when the safety layer had to override the chosen action

# ---------------------------------------------------------------------------
# Window and camera
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
FPS = 60
FAST_RENDER_EVERY = 5          # in fast mode only every Nth frame is drawn
RENDER_TRAINING = True
EGO_SCREEN_Y = 540             # the ego car stays at this screen row, the world scrolls

# ---------------------------------------------------------------------------
# Road geometry (left-hand traffic like India):
#   lanes 0,1,2 -> forward lanes (same direction as the ego car, left half)
#   lanes 3,4,5 -> opposite-direction lanes (right half)
# ---------------------------------------------------------------------------
NUM_LANES = 6
FORWARD_LANES = (0, 1, 2)
OPPOSITE_LANES = (3, 4, 5)
LANE_WIDTH = 78
ROAD_WIDTH = NUM_LANES * LANE_WIDTH
ROAD_LEFT = 120
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH
PANEL_X = ROAD_RIGHT + 36

DASH_LENGTH = 34
DASH_GAP = 26

DESTINATION_DISTANCE = 2000.0

# The yellow divider is solid: when False the ego car cannot cross into the
# oncoming lanes (they remain dangerous scenery). Set True to allow overtaking
# through the oncoming lanes (much harder to learn).
ALLOW_OPPOSITE_LANES = False

# ---------------------------------------------------------------------------
# Ego car
# ---------------------------------------------------------------------------
EGO_WIDTH = 38
EGO_LENGTH = 72
EGO_COLOR = (30, 110, 255)
COLLISION_SHRINK = 1.0

# Region the ego car may drive in (derived from ALLOW_OPPOSITE_LANES)
EGO_LEFT_LIMIT = ROAD_LEFT
EGO_RIGHT_LIMIT = ROAD_RIGHT if ALLOW_OPPOSITE_LANES else ROAD_LEFT + len(FORWARD_LANES) * LANE_WIDTH - 4
EGO_MAX_LANE = (NUM_LANES - 1) if ALLOW_OPPOSITE_LANES else (len(FORWARD_LANES) - 1)

# ---------------------------------------------------------------------------
# Vehicle catalogue
# ---------------------------------------------------------------------------
VEHICLE_TYPES = {
    "bike": {
        "width": 20, "length": 44, "min_speed": 2.5, "max_speed": 4.0,
        "colors": [(200, 40, 40), (40, 40, 160), (230, 120, 20), (60, 60, 60)],
    },
    "auto": {
        "width": 34, "length": 54, "min_speed": 2.0, "max_speed": 3.4,
        "colors": [(245, 205, 30), (60, 150, 60), (240, 190, 20)],
    },
    "car": {
        "width": 40, "length": 72, "min_speed": 2.5, "max_speed": 4.2,
        "colors": [(210, 210, 215), (200, 40, 40), (240, 240, 240), (60, 60, 70),
                   (150, 20, 20), (120, 160, 120), (230, 150, 40)],
    },
    "bus": {
        "width": 50, "length": 140, "min_speed": 2.0, "max_speed": 3.4,
        "colors": [(190, 40, 40), (40, 90, 170), (220, 120, 30)],
    },
    "truck": {
        "width": 50, "length": 120, "min_speed": 1.8, "max_speed": 3.0,
        "colors": [(230, 130, 30), (70, 120, 70), (150, 90, 50), (60, 90, 160)],
    },
}

MOVING_TYPE_WEIGHTS = {"bike": 0.28, "auto": 0.22, "car": 0.30, "bus": 0.10, "truck": 0.10}
PARKED_TYPE_WEIGHTS = {"car": 0.45, "auto": 0.20, "bike": 0.15, "truck": 0.12, "bus": 0.08}

# ---------------------------------------------------------------------------
# Traffic behaviour
# ---------------------------------------------------------------------------
SPAWN_AHEAD = 800.0            # moving traffic appears this far ahead of the ego car
DESPAWN_BEHIND = 300.0
TARGET_FORWARD_VEHICLES = 8
TARGET_OPPOSITE_VEHICLES = 10
SPAWN_PROBABILITY = 0.25
SPAWN_GAP = 45.0               # minimum bumper-to-bumper gap between vehicles at spawn time
INITIAL_CLEAR_ZONE = 160.0     # nothing spawns this close to the ego car at reset

SAFE_GAP_BASE = 18.0           # safe gap = base + speed * time
SAFE_GAP_TIME = 10.0
MIN_GAP = 10.0                 # bumper-to-bumper distance below which an NPC stops dead
NPC_ACCEL = 0.12
NPC_BRAKE = 0.5
FOLLOW_LATERAL_MARGIN = 2.0

LANE_CHANGE_SPEED = 2.0
LANE_CHANGE_COOLDOWN = 90
LANE_CHANGE_CLEARANCE = 55.0
LANE_CHANGE_TRIGGER_RATIO = 0.8
STUCK_LIMIT = 180

PARKED_START = 250.0
PARKED_END_EXTRA = 400.0
PARKED_GAP_MIN = 120.0
PARKED_GAP_MAX = 380.0
PARKED_EDGE_MARGIN = 3.0

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
FRONT_SENSOR_RANGE = 300.0
SIDE_SENSOR_RANGE = 160.0
SIDE_LONGITUDINAL_MARGIN = 45.0
FRONT_BAND_MARGIN = 4.0

# ---------------------------------------------------------------------------
# Rewards (as requested)
# ---------------------------------------------------------------------------
REWARD_SURVIVE = 1.0
REWARD_CLEAR_FRONT = 3.0
REWARD_LANE_EXPLORATION = 0.5
PENALTY_VERY_CLOSE = -5.0
PENALTY_COLLISION = -200.0
REWARD_DESTINATION = 200.0

CLEAR_FRONT_DISTANCE = 150.0   # front gap at or above this earns REWARD_CLEAR_FRONT
VERY_CLOSE_FRONT = 45.0        # front gap below this earns PENALTY_VERY_CLOSE
VERY_CLOSE_SIDE = 8.0          # side gap (vehicle or road edge) below this earns PENALTY_VERY_CLOSE

# Extra reward shaping needed once the car is allowed to slow down and stop
MIN_PROGRESS_SHARE = 0.0       # share of the survive reward still paid when standing still (0 = none)
IDLE_SPEED = 0.5               # below this speed the car counts as standing still
IDLE_CLEAR_GAP = 120.0         # standing still with at least this much free road ahead is idling
IDLE_PENALTY = -2.0
CLOSING_SPEED_THRESHOLD = 0.5  # front gap only counts as "very close" while approaching faster than this
TIMEOUT_PENALTY = -100.0       # running out of steps without reaching the destination

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_GRASS = (46, 110, 56)
COLOR_ROAD = (58, 60, 66)
COLOR_CURB = (200, 200, 200)
COLOR_LANE_LINE = (235, 235, 235)
COLOR_CENTER_LINE = (245, 200, 20)
COLOR_GLASS = (150, 195, 225)
COLOR_PANEL = (24, 26, 32)
COLOR_TEXT = (230, 232, 238)
COLOR_TEXT_DIM = (150, 155, 165)
COLOR_GOOD = (80, 220, 120)
COLOR_WARN = (250, 200, 60)
COLOR_BAD = (240, 80, 80)