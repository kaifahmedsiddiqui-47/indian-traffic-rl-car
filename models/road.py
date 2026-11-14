"""Road geometry shared by the environment and every vehicle.

A road has `lanes_per_side` lanes in each direction, centred on config.ROAD_CENTER_X:

    lanes 0 .. n-1   forward lanes (the ego car's direction), left half
    divider_x        double yellow line
    lanes n .. 2n-1  oncoming lanes, right half
"""

import random

import config


class Road:
    def __init__(self, lanes_per_side, lane_width, jitter_scale=1.0, center_x=None):
        self.lanes_per_side = int(lanes_per_side)
        self.num_lanes = 2 * self.lanes_per_side
        self.forward_lanes = tuple(range(self.lanes_per_side))
        self.opposite_lanes = tuple(range(self.lanes_per_side, self.num_lanes))
        self.lane_width = float(lane_width)
        self.jitter_scale = float(jitter_scale)

        centre = config.ROAD_CENTER_X if center_x is None else center_x
        self.width = self.num_lanes * self.lane_width
        self.left = centre - self.width / 2.0
        self.right = self.left + self.width
        self.divider_x = self.left + self.lanes_per_side * self.lane_width

    # ------------------------------------------------------------------
    # Lane helpers
    # ------------------------------------------------------------------
    def lane_center_x(self, lane):
        return self.left + (lane + 0.5) * self.lane_width

    def lane_from_x(self, x):
        lane = int((x - self.left) // self.lane_width)
        return max(0, min(self.num_lanes - 1, lane))

    def lane_direction(self, lane):
        """+1 for forward lanes, -1 for oncoming lanes."""
        return 1 if lane in self.forward_lanes else -1

    def lanes_in_same_group(self, lane):
        return self.forward_lanes if lane in self.forward_lanes else self.opposite_lanes

    def mirror_lane(self, lane):
        return self.num_lanes - 1 - lane

    def mirror_x(self, x):
        return self.left + self.right - x

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------
    def random_jitter(self, width):
        room = max(0.0, (self.lane_width - width) / 2.0 - 4.0)
        return random.uniform(-1.0, 1.0) * room * 0.5 * self.jitter_scale

    def jittered_lane_x(self, lane, width):
        return self.lane_center_x(lane) + self.random_jitter(width)

    def parked_x(self, lane, width):
        """x of a vehicle parked against the left kerb (lane 0) or the right kerb (last lane)."""
        if lane == 0:
            return self.left + width / 2.0 + config.PARKED_EDGE_MARGIN
        return self.right - width / 2.0 - config.PARKED_EDGE_MARGIN