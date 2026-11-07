"""Vehicle models: shared geometry, NPC vehicles, the ego car and their drawing code.

Coordinate system
-----------------
* x is the horizontal screen position (pixels).
* s is the position along the road in world units (pixels). Forward lanes move
  towards increasing s, opposite lanes towards decreasing s.
* The camera follows the ego car, so screen_y = EGO_SCREEN_Y - (s - ego_s).
"""

import math
import random

import pygame

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lane_center_x(lane):
    return config.ROAD_LEFT + (lane + 0.5) * config.LANE_WIDTH


def lane_from_x(x):
    lane = int((x - config.ROAD_LEFT) // config.LANE_WIDTH)
    return max(0, min(config.NUM_LANES - 1, lane))


def lane_direction(lane):
    """+1 for forward lanes, -1 for oncoming lanes."""
    return 1 if lane in config.FORWARD_LANES else -1


def lanes_in_same_group(lane):
    return config.FORWARD_LANES if lane in config.FORWARD_LANES else config.OPPOSITE_LANES


def world_to_screen_y(s, ego_s):
    return config.EGO_SCREEN_Y - (s - ego_s)


def choose_vehicle_type(weights):
    names = list(weights.keys())
    probabilities = [weights[name] for name in names]
    return random.choices(names, weights=probabilities, k=1)[0]


# ---------------------------------------------------------------------------
# Geometry base class
# ---------------------------------------------------------------------------
class Body:
    def __init__(self, x, s, width, length):
        self.x = float(x)
        self.s = float(s)
        self.width = float(width)
        self.length = float(length)

    def bounds(self, shrink=0.0):
        """Return (x_min, x_max, s_min, s_max)."""
        return (
            self.x - self.width / 2.0 + shrink,
            self.x + self.width / 2.0 - shrink,
            self.s - self.length / 2.0 + shrink,
            self.s + self.length / 2.0 - shrink,
        )

    def overlaps(self, other, shrink=0.0):
        ax_min, ax_max, as_min, as_max = self.bounds(shrink)
        bx_min, bx_max, bs_min, bs_max = other.bounds(shrink)
        return ax_min < bx_max and ax_max > bx_min and as_min < bs_max and as_max > bs_min


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _centered(cx, cy, w, h):
    rect = pygame.Rect(0, 0, max(1, int(w)), max(1, int(h)))
    rect.center = (int(cx), int(cy))
    return rect


def _shade(color, delta):
    return tuple(max(0, min(255, channel + delta)) for channel in color)


def draw_vehicle_body(surface, rect, vtype, color, heading, parked=False, is_ego=False):
    """Draw a top-down vehicle. heading=+1 means the front points up the screen."""
    cx, cy = rect.centerx, rect.centery
    w, h = rect.width, rect.height
    fy = -heading  # screen-y direction of the vehicle front
    glass = config.COLOR_GLASS
    dark = _shade(color, -55)
    light = _shade(color, 45)

    if vtype == "bike":
        pygame.draw.rect(surface, (25, 25, 25), _centered(cx, cy, max(6, w * 0.3), h), border_radius=3)
        pygame.draw.rect(surface, color, _centered(cx, cy + fy * h * 0.05, max(4, w * 0.2), h * 0.6), border_radius=2)
        pygame.draw.rect(surface, (40, 40, 40), _centered(cx, cy + fy * h * 0.22, w, 4), border_radius=2)
        pygame.draw.ellipse(surface, dark, _centered(cx, cy - fy * h * 0.02, w * 0.8, h * 0.3))
        pygame.draw.circle(surface, (235, 215, 190), (int(cx), int(cy + fy * h * 0.08)), max(3, int(w * 0.22)))
    elif vtype == "auto":
        pygame.draw.rect(surface, color, rect, border_radius=9)
        pygame.draw.rect(surface, (35, 35, 35), _centered(cx, cy - fy * h * 0.12, w - 8, h * 0.5), border_radius=6)
        pygame.draw.rect(surface, glass, _centered(cx, cy + fy * (h * 0.5 - 9), w - 12, 7), border_radius=2)
    elif vtype == "car":
        pygame.draw.rect(surface, color, rect, border_radius=9)
        pygame.draw.rect(surface, dark, _centered(cx, cy, w - 10, h * 0.34), border_radius=5)
        pygame.draw.rect(surface, glass, _centered(cx, cy + fy * h * 0.22, w - 12, h * 0.14), border_radius=3)
        pygame.draw.rect(surface, glass, _centered(cx, cy - fy * h * 0.27, w - 14, h * 0.10), border_radius=3)
        if is_ego:
            pygame.draw.rect(surface, (235, 245, 255), _centered(cx, cy, 5, h * 0.78), border_radius=2)
    elif vtype == "bus":
        pygame.draw.rect(surface, color, rect, border_radius=6)
        pygame.draw.rect(surface, light, _centered(cx, cy, w - 14, h - 30), border_radius=4)
        pygame.draw.rect(surface, glass, _centered(cx, cy + fy * (h / 2 - 9), w - 8, 12), border_radius=2)
        window_count = max(1, (h - 40) // 22)
        for i in range(int(window_count)):
            y = rect.top + 20 + i * 22
            for sx in (rect.left + 2, rect.right - 6):
                pygame.draw.rect(surface, glass, pygame.Rect(sx, y, 4, 14))
    else:  # truck
        cab_h = int(h * 0.26)
        cargo_h = h - cab_h
        if fy < 0:
            cab = pygame.Rect(rect.left, rect.top, w, cab_h)
            cargo = pygame.Rect(rect.left, rect.top + cab_h, w, cargo_h)
        else:
            cab = pygame.Rect(rect.left, rect.bottom - cab_h, w, cab_h)
            cargo = pygame.Rect(rect.left, rect.top, w, cargo_h)
        pygame.draw.rect(surface, (205, 205, 210), cargo, border_radius=4)
        for i in range(1, 4):
            line_y = cargo.top + int(cargo.height * i / 4)
            pygame.draw.line(surface, (160, 160, 168), (cargo.left + 3, line_y), (cargo.right - 3, line_y), 1)
        pygame.draw.rect(surface, color, cab, border_radius=6)
        pygame.draw.rect(surface, glass, _centered(cab.centerx, cab.centery + fy * cab_h * 0.18, w - 12, cab_h * 0.35), border_radius=2)

    if vtype != "bike":
        front_y = cy + fy * (h / 2 - 3)
        rear_y = cy - fy * (h / 2 - 3)
        for sx in (rect.left + 5, rect.right - 5):
            pygame.draw.circle(surface, (255, 250, 200), (int(sx), int(front_y)), 3)
            pygame.draw.circle(surface, (200, 30, 30), (int(sx), int(rear_y)), 2)
        pygame.draw.rect(surface, (20, 20, 20), rect, width=1, border_radius=8)

    if parked:
        for sx in (rect.left + 3, rect.right - 3):
            for sy in (rect.top + 3, rect.bottom - 3):
                pygame.draw.circle(surface, (255, 150, 0), (int(sx), int(sy)), 3)

    if is_ego:
        pygame.draw.rect(surface, (255, 255, 255), rect.inflate(6, 6), width=2, border_radius=12)


# ---------------------------------------------------------------------------
# NPC vehicles
# ---------------------------------------------------------------------------
class Vehicle(Body):
    def __init__(self, vtype, lane, s, speed, parked=False):
        spec = config.VEHICLE_TYPES[vtype]
        width = spec["width"]
        length = spec["length"]

        if parked:
            if lane == 0:
                x = config.ROAD_LEFT + width / 2.0 + config.PARKED_EDGE_MARGIN
            else:
                x = config.ROAD_RIGHT - width / 2.0 - config.PARKED_EDGE_MARGIN
        else:
            x = lane_center_x(lane) + self._random_jitter(width)

        super().__init__(x, s, width, length)

        self.vtype = vtype
        self.lane = lane
        self.target_lane = lane
        self.target_x = self.x
        self.direction = lane_direction(lane)
        self.parked = parked
        self.speed = 0.0 if parked else float(speed)
        self.target_speed = 0.0 if parked else float(speed)
        self.color = random.choice(spec["colors"])
        self.cooldown = 0
        self.stuck = 0

    @staticmethod
    def _random_jitter(width):
        room = max(0.0, (config.LANE_WIDTH - width) / 2.0 - 4.0)
        return random.uniform(-1.0, 1.0) * room * 0.5

    @staticmethod
    def jittered_lane_x(lane, width):
        return lane_center_x(lane) + Vehicle._random_jitter(width)

    def start_lane_change(self, new_lane, new_x):
        self.target_lane = new_lane
        self.target_x = new_x
        self.cooldown = config.LANE_CHANGE_COOLDOWN

    def advance(self):
        """Move one simulation step along the road (and sideways while changing lanes)."""
        if self.parked:
            return

        self.s += self.direction * self.speed

        if self.target_lane != self.lane:
            dx = self.target_x - self.x
            step = min(config.LANE_CHANGE_SPEED, abs(dx))
            self.x += math.copysign(step, dx)
            if abs(self.target_x - self.x) < 0.01:
                self.x = self.target_x
                self.lane = self.target_lane

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.speed < 0.05:
            self.stuck += 1
        else:
            self.stuck = 0

    def draw(self, surface, ego_s):
        cy = world_to_screen_y(self.s, ego_s)
        if cy + self.length / 2.0 < -5 or cy - self.length / 2.0 > config.SCREEN_HEIGHT + 5:
            return
        rect = _centered(self.x, cy, self.width, self.length)
        draw_vehicle_body(surface, rect, self.vtype, self.color, self.direction, parked=self.parked)


# ---------------------------------------------------------------------------
# Ego car
# ---------------------------------------------------------------------------
class EgoCar(Body):
    def __init__(self, x, s=0.0, speed=None):
        super().__init__(x, s, config.EGO_WIDTH, config.EGO_LENGTH)
        self.direction = 1
        self.speed = float(config.EGO_START_SPEED if speed is None else speed)
        self.last_action = config.ACTION_FORWARD
        self.braking = False
        self.overridden = False

    @property
    def lane(self):
        return lane_from_x(self.x)

    def plan_motion(self, lateral_dir, lateral_speed, mode):
        """Return (x, s, speed) after one step of the given motion without changing the car."""
        if mode == "cruise":
            target = config.EGO_CRUISE_SPEED
        elif mode == "slow":
            target = min(config.EGO_SLOW_SPEED, self.speed)
        elif mode == "hold":
            target = self.speed
        else:
            target = 0.0

        speed = self.speed
        if target > speed:
            speed = min(target, speed + config.EGO_ACCEL)
        elif target < speed:
            decel = config.EGO_BRAKE_DECEL if mode == "stop" else config.EGO_SLOW_DECEL
            speed = max(target, speed - decel)

        steer = max(config.EGO_MIN_STEER, min(1.0, speed / config.EGO_STEER_FULL_SPEED))
        half = self.width / 2.0
        x = self.x + lateral_dir * lateral_speed * steer
        x = max(config.EGO_LEFT_LIMIT + half, min(config.EGO_RIGHT_LIMIT - half, x))
        s = self.s + speed
        return x, s, speed

    def apply_motion(self, lateral_dir, lateral_speed, mode):
        previous_speed = self.speed
        self.x, self.s, self.speed = self.plan_motion(lateral_dir, lateral_speed, mode)
        self.braking = self.speed < previous_speed - 1e-6

    def apply_action(self, action):
        lateral_dir, lateral_speed, mode = config.ACTION_EFFECTS[action]
        self.apply_motion(lateral_dir, lateral_speed, mode)
        self.last_action = action

    def draw(self, surface):
        rect = _centered(self.x, config.EGO_SCREEN_Y, self.width, self.length)
        draw_vehicle_body(surface, rect, "car", config.EGO_COLOR, 1, is_ego=True)
        if self.braking or self.speed < config.IDLE_SPEED:
            rear_y = rect.bottom - 7
            for sx in (rect.left + 3, rect.right - 11):
                pygame.draw.rect(surface, (255, 40, 40), pygame.Rect(sx, rear_y, 8, 5), border_radius=2)
                pygame.draw.rect(surface, (255, 170, 170), pygame.Rect(sx + 2, rear_y + 1, 4, 2))