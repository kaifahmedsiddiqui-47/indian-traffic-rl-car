"""Indian-style traffic environment with four scenarios, rendered with pygame.

Scenarios (see config.SCENARIOS):
    highway        3 lanes each way, normal traffic
    heavy_traffic  3 lanes each way, dense slow-moving traffic
    single_lane    1 lane each way, no overtaking possible
    curvy_road     1 lane each way on a road that winds from side to side

Layout (left-hand traffic):
    lanes 0 .. n-1  : forward lanes (same direction as the ego car)
    yellow double divider (solid: the car may not cross it, so no overtaking)
    lanes n .. 2n-1 : oncoming lanes

Traffic: bikes, autos, cars, buses and trucks. Moving vehicles keep a safe
distance, follow the vehicle in front and change lane around obstacles.
Parked vehicles sit at the kerb in the outermost lanes.

Winding road: the road's centre line swings sideways as the car drives forward, so the
car has to keep steering to stay in its lane. Lanes, kerbs, sensors and the double line
are all measured as sideways offsets from the centre line, so traffic and the safety
layer behave the same way on bends as on straight roads.
"""

import math
import random

import numpy as np
import pygame

import config
from models.car_model import (
    Body,
    EgoCar,
    Vehicle,
    choose_vehicle_type,
    world_to_screen_y,
)
from models.road import Road


class TrafficEnv:
    def __init__(self, screen=None, seed=None, scenario="highway"):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.screen = screen
        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE

        self.font = None
        self.font_small = None
        self.font_big = None
        if self.screen is not None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
            self.font_big = pygame.font.Font(None, 32)

        self.scenario_name = None
        self.spec = None
        self.road = None
        self.max_steps = 0

        self.ego = None
        self.vehicles = []
        self.steps = 0
        self.visited_lanes = set()
        self.sensors = (config.FRONT_SENSOR_RANGE, config.SIDE_SENSOR_RANGE, config.SIDE_SENSOR_RANGE)
        self.front_closing = 0.0
        self.last_override = False
        self.last_info = {}
        self.hints = ["ESC  quit", "R    toggle render", "F    fast mode"]

        self.violations = 0
        self.was_crossing = False
        self.finish_s = 0.0
        self.progress_mark_s = 0.0
        self.progress_mark_step = 0

        self._load_scenario(scenario)
        self.reset()

    # ------------------------------------------------------------------
    # Scenario handling
    # ------------------------------------------------------------------
    def _load_scenario(self, name):
        if name not in config.SCENARIOS:
            raise ValueError("Unknown scenario '{}'. Choose from: {}".format(name, ", ".join(config.SCENARIOS)))
        spec = config.SCENARIOS[name]
        self.scenario_name = name
        self.spec = spec
        self.road = Road(spec["lanes_per_side"], spec["lane_width"], spec["jitter_scale"])
        self.max_steps = int(spec["max_steps"])

    def _randomise_curve(self):
        """Give a winding road a fresh phase and size for this episode (straight roads stay straight)."""
        curve = self.spec.get("curve")
        if not curve:
            self.road.set_curve(0.0, 1000.0, 0.0, 300.0)
            return
        amplitude = curve["amplitude"] * random.uniform(*config.CURVE_AMPLITUDE_JITTER)
        wavelength = curve["wavelength"] * random.uniform(*config.CURVE_WAVELENGTH_JITTER)
        phase = random.uniform(0.0, 2.0 * math.pi)
        self.road.set_curve(amplitude, wavelength, phase, curve["ramp"])

    # ------------------------------------------------------------------
    # Reset and population
    # ------------------------------------------------------------------
    def reset(self, scenario=None):
        if scenario is not None:
            self._load_scenario(scenario)

        self._randomise_curve()
        lane = random.choice(self.road.forward_lanes)
        self.ego = EgoCar(self.road, self.road.lane_center_x(lane, 0.0), 0.0)
        self.vehicles = []
        self.steps = 0
        self.visited_lanes = {lane}
        self.front_closing = 0.0
        self.last_override = False

        self.violations = 0
        self.was_crossing = False
        self.finish_s = float(self.spec["destination"])
        self.progress_mark_s = 0.0
        self.progress_mark_step = 0

        self._populate_parked()
        self._populate_moving()

        self.sensors = self._sense()
        self.last_info = {
            "scenario": self.scenario_name,
            "collision": False,
            "destination": False,
            "timeout": False,
            "terminal": False,
            "outcome": "running",
            "step": 0,
            "lane": lane,
            "progress": 0.0,
            "override": False,
            "violation": False,
            "violations": 0,
        }
        return self._get_state()

    def _world_end(self):
        """How far along the road parked vehicles are placed."""
        return self.finish_s + config.PARKED_END_EXTRA

    def _populate_parked(self):
        if not self.spec["parked"]:
            return
        end = self._world_end()
        for lane in (self.road.forward_lanes[0], self.road.opposite_lanes[-1]):
            s = config.PARKED_START + random.uniform(0.0, 150.0)
            while s < end:
                vtype = choose_vehicle_type(config.PARKED_TYPE_WEIGHTS)
                vehicle = Vehicle(self.road, vtype, lane, s, 0.0, parked=True)
                self.vehicles.append(vehicle)
                s += vehicle.length + random.uniform(config.PARKED_GAP_MIN, config.PARKED_GAP_MAX)

    def _count_moving(self):
        forward_count = 0
        opposite_count = 0
        for vehicle in self.vehicles:
            if vehicle.parked:
                continue
            if vehicle.direction > 0:
                forward_count += 1
            else:
                opposite_count += 1
        return forward_count, opposite_count

    def _populate_moving(self):
        """Fill the road ahead of the ego car with moving traffic up to the scenario's targets."""
        target_forward = self.spec["forward_vehicles"]
        target_opposite = self.spec["opposite_vehicles"]
        forward_count, opposite_count = self._count_moving()
        attempts = 0
        while (forward_count < target_forward or opposite_count < target_opposite) \
                and attempts < config.POPULATE_ATTEMPTS:
            attempts += 1
            options = []
            if forward_count < target_forward:
                options.append(self.road.forward_lanes)
            if opposite_count < target_opposite:
                options.append(self.road.opposite_lanes)
            lanes = random.choice(options)
            s = random.uniform(self.ego.s + config.INITIAL_CLEAR_ZONE, self.ego.s + config.SPAWN_AHEAD)
            if self._try_spawn(lanes, s):
                if lanes == self.road.forward_lanes:
                    forward_count += 1
                else:
                    opposite_count += 1

    def _try_spawn(self, lanes, s):
        lane = random.choice(lanes)
        vtype = choose_vehicle_type(config.MOVING_TYPE_WEIGHTS)
        spec = config.VEHICLE_TYPES[vtype]
        speed = random.uniform(spec["min_speed"], spec["max_speed"]) * self.spec["speed_scale"]
        vehicle = Vehicle(self.road, vtype, lane, s, speed)
        if self._region_clear(vehicle.off, vehicle.s, vehicle.width, vehicle.length,
                              ignore=None, long_clearance=self.spec["spawn_gap"],
                              lat_margin=2.0, include_ego=True):
            self.vehicles.append(vehicle)
            return True
        return False

    def _region_clear(self, off, s, width, length, ignore, long_clearance, lat_margin, include_ego):
        """True when no other vehicle occupies the rectangle (plus the given clearance).
        `off` is the sideways offset from the road's centre line."""
        for other in self.vehicles:
            if other is ignore:
                continue
            if abs(other.s - s) - (other.length + length) / 2.0 >= long_clearance:
                continue
            if abs(other.off - off) < (other.width + width) / 2.0 + lat_margin:
                return False
        if include_ego and self.ego is not None:
            if abs(self.ego.s - s) - (self.ego.length + length) / 2.0 < long_clearance:
                if abs(self.ego.off - off) < (self.ego.width + width) / 2.0 + lat_margin:
                    return False
        return True

    # ------------------------------------------------------------------
    # Traffic simulation
    # ------------------------------------------------------------------
    def _spawn_traffic(self):
        if random.random() > self.spec["spawn_probability"]:
            return

        forward_count, opposite_count = self._count_moving()

        options = []
        if forward_count < self.spec["forward_vehicles"]:
            options.append(self.road.forward_lanes)
        if opposite_count < self.spec["opposite_vehicles"]:
            options.append(self.road.opposite_lanes)
        if not options:
            return

        lanes = random.choice(options)
        for _ in range(4):
            if self._try_spawn(lanes, self.ego.s + config.SPAWN_AHEAD):
                break

    @staticmethod
    def _gap_ahead(vehicle, other):
        """Bumper-to-bumper gap to `other` if it is in front of `vehicle`, else None."""
        if abs(other.off - vehicle.off) >= (other.width + vehicle.width) / 2.0 + config.FOLLOW_LATERAL_MARGIN:
            return None
        if vehicle.direction > 0:
            if other.s <= vehicle.s:
                return None
            gap = (other.s - other.length / 2.0) - (vehicle.s + vehicle.length / 2.0)
        else:
            if other.s >= vehicle.s:
                return None
            gap = (vehicle.s - vehicle.length / 2.0) - (other.s + other.length / 2.0)
        return max(0.0, gap)

    def _lead_info(self, vehicle):
        """Return (gap, lead_speed) of the closest obstacle in front of `vehicle`."""
        best_gap = float("inf")
        best_speed = vehicle.target_speed

        for other in self.vehicles:
            if other is vehicle:
                continue
            gap = self._gap_ahead(vehicle, other)
            if gap is not None and gap < best_gap:
                best_gap = gap
                best_speed = other.speed if other.direction == vehicle.direction else 0.0

        gap = self._gap_ahead(vehicle, self.ego)
        if gap is not None and gap < best_gap:
            best_gap = gap
            best_speed = self.ego.speed if vehicle.direction > 0 else 0.0

        return best_gap, best_speed

    def _lane_change_path_clear(self, vehicle, new_off):
        """True when the area the vehicle will sweep while changing lane is free.

        The swept area covers the new sideways territory and extends along the road
        far enough for the distance the vehicle will travel during the manoeuvre.
        Vehicles that are themselves changing lane also block their destination.
        """
        half = vehicle.width / 2.0
        if new_off < vehicle.off:
            x_lo = new_off - half - 2.0
            x_hi = vehicle.off - half
        else:
            x_lo = vehicle.off + half
            x_hi = new_off + half + 2.0

        duration = abs(new_off - vehicle.off) / config.LANE_CHANGE_SPEED
        travel = max(vehicle.speed, vehicle.target_speed) * duration
        back = vehicle.length / 2.0 + config.LANE_CHANGE_CLEARANCE
        front = vehicle.length / 2.0 + travel + config.LANE_CHANGE_CLEARANCE * 0.5
        if vehicle.direction > 0:
            s_lo, s_hi = vehicle.s - back, vehicle.s + front
        else:
            s_lo, s_hi = vehicle.s - front, vehicle.s + back

        others = list(self.vehicles)
        others.append(self.ego)
        for other in others:
            if other is vehicle:
                continue
            _, _, o_s_min, o_s_max = other.bounds()
            o_x_min = other.off - other.width / 2.0
            o_x_max = other.off + other.width / 2.0

            # the other vehicle keeps moving during the manoeuvre (e.g. a faster car closing in from behind)
            other_speed = getattr(other, "target_speed", config.EGO_CRUISE_SPEED)
            other_speed = max(other_speed, other.speed)
            drift = other.direction * other_speed * duration
            o_s_lo = min(o_s_min, o_s_min + drift)
            o_s_hi = max(o_s_max, o_s_max + drift)
            if o_s_hi <= s_lo or o_s_lo >= s_hi:
                continue
            if o_x_max > x_lo and o_x_min < x_hi:
                return False
            target_off = getattr(other, "target_off", None)
            if target_off is not None and other.target_lane != other.lane:
                if target_off + other.width / 2.0 > x_lo and target_off - other.width / 2.0 < x_hi:
                    return False
        return True

    def _try_lane_change(self, vehicle):
        candidates = [
            lane for lane in (vehicle.lane - 1, vehicle.lane + 1)
            if lane in self.road.lanes_in_same_group(vehicle.lane)
        ]
        random.shuffle(candidates)
        for lane in candidates:
            new_off = self.road.jittered_lane_off(lane, vehicle.width)
            if self._lane_change_path_clear(vehicle, new_off):
                vehicle.start_lane_change(lane, new_off)
                return True
        return False

    def _update_npcs(self):
        survivors = []
        for vehicle in self.vehicles:
            if vehicle.parked:
                survivors.append(vehicle)
                continue

            gap, lead_speed = self._lead_info(vehicle)
            safe_gap = config.SAFE_GAP_BASE + vehicle.speed * config.SAFE_GAP_TIME

            desired = vehicle.target_speed
            if gap < safe_gap:
                if gap < config.MIN_GAP:
                    desired = 0.0
                else:
                    ratio = (gap - config.MIN_GAP) / (safe_gap - config.MIN_GAP)
                    desired = lead_speed + ratio * (vehicle.target_speed - lead_speed)
                    desired = max(0.0, min(vehicle.target_speed, desired))

                blocked = lead_speed < vehicle.target_speed * config.LANE_CHANGE_TRIGGER_RATIO
                if blocked and vehicle.cooldown == 0 and vehicle.target_lane == vehicle.lane:
                    self._try_lane_change(vehicle)

            if desired > vehicle.speed:
                vehicle.speed = min(desired, vehicle.speed + config.NPC_ACCEL)
            else:
                vehicle.speed = max(desired, vehicle.speed - config.NPC_BRAKE)

            vehicle.advance()

            if vehicle.s < self.ego.s - config.DESPAWN_BEHIND:
                continue
            if vehicle.s > self.ego.s + config.SPAWN_AHEAD + 300.0:
                continue
            if vehicle.stuck > config.STUCK_LIMIT:
                continue
            survivors.append(vehicle)
        self.vehicles = survivors

    # ------------------------------------------------------------------
    # Double line
    # ------------------------------------------------------------------
    def _crossing_line(self, x, s):
        """True when an ego-sized car at (x, s) touches or passes the double line."""
        half = self.ego.width / 2.0
        return self.road.offset(x, s) + half > 0.0

    def _motion_crosses_line(self, motion):
        lateral_dir, lateral_speed, mode = motion
        nx, ns, nv = self.ego.plan_motion(lateral_dir, lateral_speed, mode)
        return self._crossing_line(nx, ns)

    def _lane_centre_error(self):
        """Distance (px) from the middle of the ego's current lane, as a share of the room it has."""
        road = self.road
        off = self.ego.off
        lane = road.lane_from_offset(off)
        room = max(1.0, road.lane_width / 2.0 - self.ego.width / 2.0)
        return abs(off - road.lane_center_off(lane)) / room

    def _mission_progress(self):
        """Progress towards the finish line, 0 to 1."""
        return max(0.0, min(1.0, self.ego.s / self.finish_s))

    # ------------------------------------------------------------------
    # Sensors and state
    # ------------------------------------------------------------------
    def _walls(self):
        """Left and right limits the car senses, as offsets from the centre line: kerb and double line."""
        half = self.road.half_width
        if self.ego.off < 0.0:
            return -half, 0.0
        return 0.0, half

    def _sense(self):
        """Return (front_distance, left_gap, right_gap) in pixels."""
        ego = self.ego
        _, _, es_min, es_max = ego.bounds()
        ego_off = ego.off
        ex_min = ego_off - ego.width / 2.0
        ex_max = ego_off + ego.width / 2.0
        left_wall, right_wall = self._walls()

        front = config.FRONT_SENSOR_RANGE
        left = min(config.SIDE_SENSOR_RANGE, max(0.0, ex_min - left_wall))
        right = min(config.SIDE_SENSOR_RANGE, max(0.0, right_wall - ex_max))

        band_min = ex_min - config.FRONT_BAND_MARGIN
        band_max = ex_max + config.FRONT_BAND_MARGIN
        lead_velocity = None

        for other in self.vehicles:
            if abs(other.s - ego.s) > config.FRONT_SENSOR_RANGE + 200.0:
                continue
            _, _, os_min, os_max = other.bounds()
            ox_min = other.off - other.width / 2.0
            ox_max = other.off + other.width / 2.0

            if other.s > ego.s and ox_max > band_min and ox_min < band_max:
                gap = max(0.0, os_min - es_max)
                if gap < front:
                    front = gap
                    lead_velocity = other.direction * other.speed

            lateral_overlap = ox_max > ex_min and ox_min < ex_max
            longitudinal_near = (
                os_max > es_min - config.SIDE_LONGITUDINAL_MARGIN
                and os_min < es_max + config.SIDE_LONGITUDINAL_MARGIN
            )
            if longitudinal_near and not lateral_overlap:
                if other.off < ego_off:
                    left = min(left, max(0.0, ex_min - ox_max))
                else:
                    right = min(right, max(0.0, ox_min - ex_max))

        if lead_velocity is None or front >= config.FRONT_SENSOR_RANGE:
            self.front_closing = 0.0
        else:
            self.front_closing = float(ego.speed - lead_velocity)

        return float(front), float(left), float(right)

    def _get_state(self):
        front, left, right = self.sensors
        ego = self.ego
        road = self.road
        lane = ego.lane
        car_x = ego.off + road.half_width        # 0 at the left kerb
        progress = self._mission_progress()
        slope_now = road.slope(ego.s)
        slope_ahead = road.slope(ego.s + config.CURVE_LOOKAHEAD)

        if config.NORMALIZE_STATE:
            limit = config.CURVE_MAX_SLOPE
            state = [
                car_x / road.width,
                progress,
                front / config.FRONT_SENSOR_RANGE,
                left / config.SIDE_SENSOR_RANGE,
                right / config.SIDE_SENSOR_RANGE,
                lane / float(road.num_lanes - 1),
                min(1.0, ego.speed / config.EGO_CRUISE_SPEED),
                max(-1.0, min(1.0, self.front_closing / config.EGO_CRUISE_SPEED)),
                max(-1.0, min(1.0, slope_now / limit)),
                max(-1.0, min(1.0, slope_ahead / limit)),
            ]
        else:
            state = [car_x, progress, front, left, right, float(lane), ego.speed,
                     self.front_closing, slope_now, slope_ahead]
        return np.array(state, dtype=np.float32)

    def _check_collision(self):
        for other in self.vehicles:
            if abs(other.s - self.ego.s) > 250.0:
                continue
            if self.ego.overlaps(other, shrink=config.COLLISION_SHRINK):
                return True
        return False

    # ------------------------------------------------------------------
    # Safety layer: predict the next step, brake or stop instead of crashing
    # ------------------------------------------------------------------
    def _predict_others(self):
        """Predicted (body, velocity along the road, sideways offset) of every nearby vehicle one step ahead."""
        predicted = []
        for other in self.vehicles:
            if abs(other.s - self.ego.s) > 300.0:
                continue
            off = other.off
            s = other.s
            velocity = 0.0
            if not other.parked:
                velocity = other.direction * other.speed
                s += velocity
                if other.target_lane != other.lane:
                    dx = other.target_off - other.off
                    off += math.copysign(min(config.LANE_CHANGE_SPEED, abs(dx)), dx)
            body = Body(self.road.x_from_offset(off, s), s, other.width, other.length)
            predicted.append((body, velocity, off))
        return predicted

    def _motion_is_safe(self, motion, predicted):
        """True when the motion causes no overlap and leaves enough room to brake to a stop."""
        lateral_dir, lateral_speed, mode = motion
        nx, ns, nv = self.ego.plan_motion(lateral_dir, lateral_speed, mode)
        ghost = Body(nx, ns, self.ego.width, self.ego.length)
        half = self.ego.width / 2.0
        new_off = self.road.offset(nx, ns)
        band_min = new_off - half - config.FRONT_BAND_MARGIN
        band_max = new_off + half + config.FRONT_BAND_MARGIN
        ego_front = ns + self.ego.length / 2.0

        for body, velocity, off in predicted:
            if ghost.overlaps(body, shrink=-config.SHIELD_SIDE_MARGIN):
                return False
            if body.s <= ns:
                continue
            _, _, bs_min, bs_max = body.bounds()
            if off + body.width / 2.0 <= band_min or off - body.width / 2.0 >= band_max:
                continue
            gap = bs_min - ego_front
            closing = nv - velocity
            if closing > 0.0:
                braking_distance = closing * closing / (2.0 * config.EGO_BRAKE_DECEL)
                if gap < braking_distance + config.SHIELD_MARGIN:
                    return False
        return True

    def _shield(self, motion):
        """Return (motion to execute, whether the chosen motion had to be replaced).

        The agent's own choice is only checked for collisions, so crossing the double
        line is the agent's own mistake and is penalised. Replacement moves chosen by
        the shield never cross the line.
        """
        predicted = self._predict_others()
        lateral_dir, lateral_speed, mode = motion

        candidates = [motion]
        if lateral_dir != 0:
            candidates.append((lateral_dir, lateral_speed, "hold"))
            candidates.append((lateral_dir, lateral_speed, "slow"))
            candidates.append((lateral_dir, lateral_speed, "stop"))
        candidates.append((0, 0.0, "slow"))
        candidates.append((0, 0.0, "stop"))

        for index, candidate in enumerate(candidates):
            if index > 0 and self._motion_crosses_line(candidate):
                continue
            if self._motion_is_safe(candidate, predicted):
                return candidate, candidate != motion

        fallback = (0, 0.0, "stop")
        return fallback, fallback != motion

    # ------------------------------------------------------------------
    # Gym-style step
    # ------------------------------------------------------------------
    def step(self, action):
        self.steps += 1
        action = int(action)

        motion = config.ACTION_EFFECTS[action]
        override = False
        if config.SAFETY_SHIELD:
            motion, override = self._shield(motion)

        self.ego.apply_motion(*motion)
        self.ego.last_action = action
        self.ego.overridden = override
        self.last_override = override

        self._update_npcs()
        self._spawn_traffic()

        collided = self._check_collision()

        crossing = False
        new_violation = False
        if not collided:
            crossing = self._crossing_line(self.ego.x, self.ego.s)
            new_violation = crossing and not self.was_crossing
            if new_violation:
                self.violations += 1
            self.was_crossing = crossing

        self.sensors = self._sense()
        front, left, right = self.sensors
        lane = self.ego.lane
        speed = self.ego.speed
        speed_ratio = max(0.0, min(1.0, speed / config.EGO_CRUISE_SPEED))

        violated = crossing and config.END_EPISODE_ON_VIOLATION

        reward = 0.0
        reached = False

        if collided:
            reward = config.PENALTY_COLLISION
        elif violated:
            reward = config.PENALTY_DOUBLE_LINE
        else:
            # +1 survive (paid in full only while actually making progress)
            reward += config.REWARD_SURVIVE * (
                config.MIN_PROGRESS_SHARE + (1.0 - config.MIN_PROGRESS_SHARE) * speed_ratio
            )

            # +3 clear front distance (scaled by speed squared so crawling is never worth it)
            if front >= config.CLEAR_FRONT_DISTANCE:
                reward += config.REWARD_CLEAR_FRONT * speed_ratio * speed_ratio

            # +0.5 lane exploration
            if lane not in self.visited_lanes:
                self.visited_lanes.add(lane)
                reward += config.REWARD_LANE_EXPLORATION

            # -5 very close obstacle (only while closing in or squeezing past something)
            approaching = front < config.VERY_CLOSE_FRONT and self.front_closing > config.CLOSING_SPEED_THRESHOLD
            squeezing = (
                (left < config.VERY_CLOSE_SIDE or right < config.VERY_CLOSE_SIDE)
                and speed > config.IDLE_SPEED
            )
            if approaching or squeezing:
                reward += config.PENALTY_VERY_CLOSE

            # standing still on an open road is not allowed
            if speed < config.IDLE_SPEED and front >= config.IDLE_CLEAR_GAP:
                reward += config.IDLE_PENALTY

            if override:
                reward += config.SHIELD_PENALTY

            # double line crossed while the episode continues
            if crossing:
                reward += config.PENALTY_DOUBLE_LINE if new_violation else config.PENALTY_DOUBLE_LINE_STEP

            # winding road: reward for staying in the middle of the lane
            if self.road.curved:
                centring = max(0.0, 1.0 - self._lane_centre_error())
                reward += config.REWARD_LANE_CENTER * centring * speed_ratio

            if self.ego.s >= self.finish_s:
                reached = True
                reward += config.REWARD_DESTINATION

        terminal = collided or violated or reached

        # Time limits: a generous cap, plus an "no progress for a very long time" check.
        if self.ego.s >= self.progress_mark_s + config.STUCK_MIN_PROGRESS:
            self.progress_mark_s = self.ego.s
            self.progress_mark_step = self.steps
        stuck = (not terminal) and (self.steps - self.progress_mark_step) >= config.STUCK_WINDOW_STEPS
        capped = (not terminal) and self.steps >= self.max_steps
        timeout = stuck or capped
        done = terminal or timeout
        if timeout:
            reward += config.TIMEOUT_PENALTY

        if collided:
            outcome = "collision"
        elif violated:
            outcome = "double_line"
        elif reached:
            outcome = "destination"
        elif stuck:
            outcome = "stuck"
        elif capped:
            outcome = "timeout"
        else:
            outcome = "running"

        info = {
            "scenario": self.scenario_name,
            "collision": collided,
            "destination": reached,
            "timeout": timeout,
            "terminal": terminal,
            "outcome": outcome,
            "step": self.steps,
            "lane": lane,
            "progress": self._mission_progress(),
            "front": front,
            "left": left,
            "right": right,
            "speed": speed,
            "closing": self.front_closing,
            "override": override,
            "violation": crossing,
            "violations": self.violations,
        }
        self.last_info = info
        return self._get_state(), float(reward), done, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, hud=None, flip=True):
        """Draw the current frame. With flip=False the caller draws more on top and flips itself."""
        if self.screen is None:
            return

        surface = self.screen
        self._draw_road(surface)
        self._draw_destination(surface)

        for vehicle in self.vehicles:
            vehicle.draw(surface, self.ego.s)

        self._draw_sensors(surface)
        self.ego.draw(surface)
        self._draw_panel(surface, hud)
        if flip:
            pygame.display.flip()

    def _draw_road(self, surface):
        road = self.road
        height = config.SCREEN_HEIGHT
        surface.fill(config.COLOR_GRASS)

        if road.curved:
            self._draw_curved_road(surface)
        else:
            road_rect = pygame.Rect(int(road.left), 0, int(road.width), height)
            pygame.draw.rect(surface, config.COLOR_ROAD, road_rect)

            pygame.draw.rect(surface, config.COLOR_CURB, (int(road.left) - 6, 0, 6, height))
            pygame.draw.rect(surface, config.COLOR_CURB, (int(road.right), 0, 6, height))

            period = config.DASH_LENGTH + config.DASH_GAP
            offset = int(self.ego.s % period)
            for i in range(1, road.num_lanes):
                if i == road.lanes_per_side:
                    continue
                x = int(road.left + i * road.lane_width)
                y = -period + offset
                while y < height:
                    pygame.draw.rect(surface, config.COLOR_LANE_LINE, (x - 1, y, 3, config.DASH_LENGTH))
                    y += period

            center_x = int(road.divider_x)
            pygame.draw.rect(surface, config.COLOR_CENTER_LINE, (center_x - 4, 0, 3, height))
            pygame.draw.rect(surface, config.COLOR_CENTER_LINE, (center_x + 1, 0, 3, height))

        if self.font_small is not None:
            top_s = self.ego.s + config.EGO_SCREEN_Y - 12
            centre = int(road.centre_at(top_s))
            forward_label = self.font_small.render("FORWARD", True, config.COLOR_TEXT_DIM)
            oncoming_label = self.font_small.render("ONCOMING", True, config.COLOR_TEXT_DIM)
            surface.blit(forward_label, (int(road.left_at(top_s)) + 8, 6))
            surface.blit(oncoming_label, (centre + 8, 6))

    def _draw_curved_road(self, surface):
        """Draw a winding road as strips: one point every few pixels down the screen."""
        road = self.road
        height = config.SCREEN_HEIGHT
        step = 6
        ys = list(range(-step, height + step * 2, step))
        centres = [road.centre_at(self.ego.s + (config.EGO_SCREEN_Y - y)) for y in ys]
        half = road.half_width

        def edge(delta):
            return [(int(c + delta), y) for c, y in zip(centres, ys)]

        left_edge = edge(-half)
        right_edge = edge(half)
        pygame.draw.polygon(surface, config.COLOR_ROAD, left_edge + right_edge[::-1])
        pygame.draw.lines(surface, config.COLOR_CURB, False, edge(-half - 3), 6)
        pygame.draw.lines(surface, config.COLOR_CURB, False, edge(half + 3), 6)
        pygame.draw.lines(surface, config.COLOR_CENTER_LINE, False, edge(-2.5), 3)
        pygame.draw.lines(surface, config.COLOR_CENTER_LINE, False, edge(2.5), 3)

        # dashed white lines between lanes on the same side (only when a side has several lanes)
        period = config.DASH_LENGTH + config.DASH_GAP
        for i in range(1, road.num_lanes):
            if i == road.lanes_per_side:
                continue
            delta = -half + i * road.lane_width
            for k in range(-1, height // period + 2):
                y0 = -period + int(self.ego.s % period) + k * period
                y1 = y0 + config.DASH_LENGTH
                pts = [(int(road.centre_at(self.ego.s + (config.EGO_SCREEN_Y - yy)) + delta), yy)
                       for yy in range(y0, y1 + 1, step)]
                if len(pts) >= 2:
                    pygame.draw.lines(surface, config.COLOR_LANE_LINE, False, pts, 3)

    def _draw_destination(self, surface):
        road = self.road
        y = int(world_to_screen_y(self.finish_s, self.ego.s))
        if y < -30 or y > config.SCREEN_HEIGHT + 30:
            return
        left = int(road.left_at(self.finish_s))
        right = int(road.right_at(self.finish_s))
        square = 13
        columns = int(road.width) // square + 1
        for row in range(2):
            for col in range(columns):
                color = (245, 245, 245) if (row + col) % 2 == 0 else (20, 20, 20)
                rect = pygame.Rect(left + col * square, y - row * square, square, square)
                rect.width = min(square, right - rect.left)
                if rect.width > 0:
                    pygame.draw.rect(surface, color, rect)
        if self.font is not None:
            label = self.font.render("FINISH", True, (255, 255, 255))
            surface.blit(label, ((left + right) // 2 - label.get_width() // 2, y - 46))

    def _draw_sensors(self, surface):
        front, left, right = self.sensors
        ego = self.ego
        top = config.EGO_SCREEN_Y - ego.length / 2.0
        cx = int(ego.x)
        cy = config.EGO_SCREEN_Y

        if front < config.VERY_CLOSE_FRONT:
            front_color = config.COLOR_BAD
        elif front < config.CLEAR_FRONT_DISTANCE:
            front_color = config.COLOR_WARN
        else:
            front_color = config.COLOR_GOOD
        pygame.draw.line(surface, front_color, (cx, int(top)), (cx, int(top - front)), 2)
        pygame.draw.circle(surface, front_color, (cx, int(top - front)), 4)

        left_edge = int(ego.x - ego.width / 2.0)
        right_edge = int(ego.x + ego.width / 2.0)
        pygame.draw.line(surface, config.COLOR_WARN, (left_edge, cy), (int(left_edge - left), cy), 1)
        pygame.draw.line(surface, config.COLOR_WARN, (right_edge, cy), (int(right_edge + right), cy), 1)

    def _draw_panel(self, surface, hud):
        panel = pygame.Rect(config.PANEL_X - 12, 10, config.SCREEN_WIDTH - config.PANEL_X, config.SCREEN_HEIGHT - 20)
        pygame.draw.rect(surface, config.COLOR_PANEL, panel, border_radius=10)

        x = config.PANEL_X
        y = 26
        title = self.font_big.render("Intelli-Car DQN", True, config.COLOR_TEXT)
        surface.blit(title, (x, y))
        y += 40

        lines = []
        if hud:
            for key, value in hud.items():
                lines.append((str(key), str(value)))

        front, left, right = self.sensors
        info = self.last_info
        lines.extend([
            ("Scenario", self.scenario_name.replace("_", " ")),
            ("Road", "winding" if self.road.curved else "straight"),
            ("Step", "{}".format(self.steps)),
            ("Lane", "{}".format(self.ego.lane)),
            ("Action", config.ACTION_NAMES.get(self.ego.last_action, "-")),
            ("Speed", "{:.1f}".format(self.ego.speed)),
            ("Safety", "AUTO-BRAKE" if self.last_override else "ok"),
            ("Line", "CROSSED" if self.was_crossing else "ok"),
            ("Front", "{:.0f}".format(front)),
            ("Left gap", "{:.0f}".format(left)),
            ("Right gap", "{:.0f}".format(right)),
            ("Status", info.get("outcome", "running")),
        ])

        for key, value in lines:
            key_surface = self.font_small.render(key, True, config.COLOR_TEXT_DIM)
            value_surface = self.font.render(value, True, config.COLOR_TEXT)
            surface.blit(key_surface, (x, y))
            surface.blit(value_surface, (x + 96, y - 2))
            y += 24

        y += 6
        bar_width = config.SCREEN_WIDTH - config.PANEL_X - 30
        progress = self._mission_progress()
        pygame.draw.rect(surface, (60, 64, 74), (x, y, bar_width, 14), border_radius=7)
        pygame.draw.rect(surface, config.COLOR_GOOD, (x, y, int(bar_width * progress), 14), border_radius=7)
        label = self.font_small.render("Trip progress", True, config.COLOR_TEXT_DIM)
        surface.blit(label, (x, y + 20))

        hint_y = config.SCREEN_HEIGHT - 28 - 22 * len(self.hints)
        for hint in self.hints:
            surface.blit(self.font_small.render(hint, True, config.COLOR_TEXT_DIM), (x, hint_y))
            hint_y += 22

    def close(self):
        self.screen = None