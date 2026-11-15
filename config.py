"""Central configuration for the Intelli-Car (DQN) Indian traffic simulator.

All tunable numbers live here so that the environment, the agent and the
training loop never contain magic constants.  World units are pixels.

The simulator has four driving scenarios (see SCENARIOS below). The agent is
trained on each scenario in turn, each with its own number of episodes, and is
then tested on all four. Every episode is limited to MAX_STEPS steps.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_last.pth")
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_final.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dqn_best.pth")       # best model found during training
BEST_INFO_PATH = os.path.join(RESULTS_DIR, "best_model.json")        # when and how well it did
SAVE_EVERY_EPISODES = 25

# ---------------------------------------------------------------------------
# Training hyper-parameters (as requested)
# ---------------------------------------------------------------------------
EPISODES = 200                 # episodes per scenario (each scenario below can override it)
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
# Staged training and the final test
# ---------------------------------------------------------------------------
SCENARIO_ORDER = ["highway", "heavy_traffic", "single_lane", "curvy_road"]

# Every episode ends after this many steps at the latest.
MAX_STEPS = 1000

# Episodes the agent spends in each scenario (edit one entry to change only that scenario).
EPISODES_PER_SCENARIO = {
    "highway": EPISODES,
    "heavy_traffic": EPISODES,
    "single_lane": EPISODES,
    "curvy_road": EPISODES,
}

# Optional extra phase after the four scenarios: every episode picks a random scenario,
# which helps the agent keep what it learned in earlier scenarios. 0 turns it off.
MIXED_EPISODES = 0

# Epsilon is raised to this value at the start of every scenario after the first,
# so the agent explores the new road layout instead of only repeating old habits.
STAGE_EPSILON_RESTART = 0.6

FINAL_TEST_EPISODES = 20       # greedy test episodes per scenario at the end
TEST_SEED = 1234               # test episode i uses seed TEST_SEED + i, so tests are repeatable

# Best-model tracking: during training the agent is quietly tested on all four scenarios
# (with its own seeds, different from the presentation seeds) and the highest scoring
# version is kept as BEST_MODEL_PATH. The score is the overall success rate, then the reward.
VALIDATE_EVERY_EPISODES = 50   # also checked at the end of every scenario stage
VALIDATION_EPISODES = 3        # per scenario, so 12 quiet episodes per check
VALIDATION_SEED = 777000

# ---------------------------------------------------------------------------
# State / action spaces
# ---------------------------------------------------------------------------
# State: [car_x, trip_progress, front_distance, left_gap, right_gap, current_lane,
#         speed, closing_speed, road_slope_here, road_slope_ahead]
# (the last two tell the car which way a winding road bends; they are 0 on straight roads)
STATE_SIZE = 10
ACTION_SIZE = 7
NORMALIZE_STATE = True         # False returns raw pixel values instead of normalised values

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_DIAGONAL_LEFT = 3
ACTION_DIAGONAL_RIGHT = 4
ACTION_SLOW = 5                # ease off the throttle / gentle brake
ACTION_STOP = 6                # hard brake until the car stands still

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
# then straight slow, then straight stop. Replacement moves never cross the double line.
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
# Road drawing (left-hand traffic like India). The road is always centred on
# ROAD_CENTER_X; its width depends on the scenario's lanes_per_side and lane_width.
#   lanes 0 .. n-1   : forward lanes (same direction as the ego car, left half)
#   double yellow line in the middle (solid: crossing it is not allowed)
#   lanes n .. 2n-1  : oncoming lanes (right half)
# ---------------------------------------------------------------------------
ROAD_CENTER_X = 354
PANEL_X = 624

DASH_LENGTH = 34
DASH_GAP = 26

# ---------------------------------------------------------------------------
# Scenarios
#   lanes_per_side     lanes in each direction
#   lane_width         px per lane
#   jitter_scale       how far moving vehicles wander inside their lane (0 = dead centre)
#   forward_vehicles   target number of moving vehicles in the ego's direction
#   opposite_vehicles  target number of oncoming moving vehicles
#   speed_scale        multiplies every moving vehicle's speed range (slow = heavy traffic)
#   spawn_gap          minimum bumper-to-bumper gap when a vehicle is created
#   spawn_probability  chance per step that a new vehicle is added when below target
#   parked             whether parked vehicles line the kerbs
#   destination        finish line distance (px from the start)
#   max_steps          episode length limit
#   curve              None for a straight road, otherwise the winding-road settings:
#                      amplitude (max sideways swing, px), wavelength (px of road per full
#                      wave) and ramp (px of straight road before the bends start); the
#                      phase and the exact size change a little in every episode
# ---------------------------------------------------------------------------
SCENARIOS = {
    "highway": {
        "title": "Highway - 3 lanes each way",
        "lanes_per_side": 3,
        "lane_width": 78,
        "jitter_scale": 1.0,
        "forward_vehicles": 8,
        "opposite_vehicles": 10,
        "speed_scale": 1.0,
        "spawn_gap": 45.0,
        "spawn_probability": 0.25,
        "parked": True,
        "destination": 2000.0,
        "max_steps": MAX_STEPS,
        "curve": None,
    },
    "heavy_traffic": {
        "title": "Heavy slow traffic - 3 lanes each way",
        "lanes_per_side": 3,
        "lane_width": 78,
        "jitter_scale": 1.0,
        "forward_vehicles": 14,
        "opposite_vehicles": 12,
        "speed_scale": 0.45,
        "spawn_gap": 26.0,
        "spawn_probability": 0.4,
        "parked": True,
        "destination": 800.0,
        "max_steps": MAX_STEPS,
        "curve": None,
    },
    "single_lane": {
        "title": "Narrow road - 1 lane each way",
        "lanes_per_side": 1,
        "lane_width": 84,
        "jitter_scale": 0.0,
        "forward_vehicles": 5,
        "opposite_vehicles": 5,
        "speed_scale": 0.9,
        "spawn_gap": 45.0,
        "spawn_probability": 0.25,
        "parked": False,
        "destination": 1100.0,
        "max_steps": MAX_STEPS,
        "curve": None,
    },
    "curvy_road": {
        "title": "Curvy narrow road - 1 lane each way",
        "lanes_per_side": 1,
        "lane_width": 92,
        "jitter_scale": 0.0,
        "forward_vehicles": 4,
        "opposite_vehicles": 4,
        "speed_scale": 0.9,
        "spawn_gap": 60.0,
        "spawn_probability": 0.2,
        "parked": False,
        "destination": 1100.0,
        "max_steps": MAX_STEPS,
        "curve": {"amplitude": 62.0, "wavelength": 950.0, "ramp": 350.0},
    },
}

# Winding roads: how the state describes the bends
CURVE_MAX_SLOPE = 0.6          # road slope (sideways px per px of road) that maps to 1.0 in the state
CURVE_LOOKAHEAD = 120.0        # how far ahead the "road_slope_ahead" value looks (px)
CURVE_AMPLITUDE_JITTER = (0.85, 1.0)    # per-episode scale of the amplitude
CURVE_WAVELENGTH_JITTER = (0.85, 1.15)  # per-episode scale of the wavelength

# ---------------------------------------------------------------------------
# Time limits: every episode is capped at MAX_STEPS. An episode also ends early when
# the car makes no real forward progress for a while.
# ---------------------------------------------------------------------------
STUCK_WINDOW_STEPS = 400       # steps without enough progress before the episode is ended as "stuck"
STUCK_MIN_PROGRESS = 20.0      # px of forward progress that counts as "not stuck"

# ---------------------------------------------------------------------------
# Ego car
# ---------------------------------------------------------------------------
EGO_WIDTH = 38
EGO_LENGTH = 72
EGO_COLOR = (30, 110, 255)
COLLISION_SHRINK = 1.0

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
# Traffic behaviour (shared by all scenarios)
# ---------------------------------------------------------------------------
SPAWN_AHEAD = 800.0            # moving traffic appears this far ahead of the ego car
DESPAWN_BEHIND = 300.0
INITIAL_CLEAR_ZONE = 160.0     # nothing spawns this close to the ego car at reset
POPULATE_ATTEMPTS = 900        # tries when filling the road at reset

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
STUCK_LIMIT = 180              # steps a moving NPC may stand still before it is removed

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
TIMEOUT_PENALTY = -100.0       # running out of steps (or being stuck) without reaching the finish line

# Double yellow line: no crossing and therefore no overtaking, in any condition.
PENALTY_DOUBLE_LINE = -100.0       # when the car crosses the double line
PENALTY_DOUBLE_LINE_STEP = -5.0    # every further step spent over the line (when the episode continues)
END_EPISODE_ON_VIOLATION = True    # crossing the line ends the episode as a failure

# Winding road: staying near the middle of the lane through the bends
REWARD_LANE_CENTER = 0.5           # per step, scaled by how well the car is centred and by its speed

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