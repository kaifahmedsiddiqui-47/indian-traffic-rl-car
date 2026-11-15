"""Road geometry shared by the environment and every vehicle.

A road has `lanes_per_side` lanes in each direction, centred on config.ROAD_CENTER_X:

    lanes 0 .. n-1   forward lanes (the ego car's direction), left half
    centre line      double yellow line
    lanes n .. 2n-1  oncoming lanes, right half

Straight roads never move. A curvy road winds from side to side: its centre line is
displaced sideways by shift(s), a smooth sine wave that fades in over the first few
hundred pixels so every episode starts on a straight piece.

Positions are described in two ways:
    world x   the real horizontal screen position
    offset    the sideways distance from the road's centre line at that point of the road
              (negative = forward half, positive = oncoming half)
Everything that talks about lanes, kerbs or the double line works with offsets, so the same
code runs on straight and winding roads.
"""

import math
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

        self.center_x = float(config.ROAD_CENTER_X if center_x is None else center_x)
        self.width = self.num_lanes * self.lane_width
        self.half_width = self.width / 2.0

        # edges of the straight road (a winding road moves these sideways by shift(s))
        self.left = self.center_x - self.half_width
        self.right = self.center_x + self.half_width
        self.divider_x = self.center_x

        # winding road settings (amplitude 0 means straight)
        self.amplitude = 0.0
        self.wavelength = 1000.0
        self.phase = 0.0
        self.ramp = 300.0

    # ------------------------------------------------------------------
    # Curve
    # ------------------------------------------------------------------
    @property
    def curved(self):
        return self.amplitude > 0.0

    def set_curve(self, amplitude, wavelength, phase, ramp):
        self.amplitude = float(amplitude)
        self.wavelength = float(wavelength)
        self.phase = float(phase)
        self.ramp = max(1.0, float(ramp))

    def _fade(self, s):
        """Smooth 0..1 fade-in of the curve over the first `ramp` pixels, and its slope."""
        t = s / self.ramp
        if t <= 0.0:
            return 0.0, 0.0
        if t >= 1.0:
            return 1.0, 0.0
        return t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t) / self.ramp

    def shift(self, s):
        """Sideways displacement of the road centre line at road position s."""
        if self.amplitude <= 0.0:
            return 0.0
        fade, _ = self._fade(s)
        return self.amplitude * fade * math.sin(2.0 * math.pi * s / self.wavelength + self.phase)

    def slope(self, s):
        """How fast the centre line moves sideways per pixel of road (dx / ds)."""
        if self.amplitude <= 0.0:
            return 0.0
        fade, fade_slope = self._fade(s)
        angle = 2.0 * math.pi * s / self.wavelength + self.phase
        return self.amplitude * (
            fade_slope * math.sin(angle) + fade * math.cos(angle) * 2.0 * math.pi / self.wavelength
        )

    # ------------------------------------------------------------------
    # Offset <-> world x
    # ------------------------------------------------------------------
    def centre_at(self, s):
        return self.center_x + self.shift(s)

    def left_at(self, s):
        return self.centre_at(s) - self.half_width

    def right_at(self, s):
        return self.centre_at(s) + self.half_width

    def divider_at(self, s):
        return self.centre_at(s)

    def offset(self, x, s):
        return x - self.centre_at(s)

    def x_from_offset(self, offset, s):
        return self.centre_at(s) + offset

    # ------------------------------------------------------------------
    # Lane helpers (all in offsets)
    # ------------------------------------------------------------------
    def lane_center_off(self, lane):
        return -self.half_width + (lane + 0.5) * self.lane_width

    def lane_center_x(self, lane, s=0.0):
        return self.x_from_offset(self.lane_center_off(lane), s)

    def lane_from_offset(self, offset):
        lane = int((offset + self.half_width) // self.lane_width)
        return max(0, min(self.num_lanes - 1, lane))

    def lane_direction(self, lane):
        """+1 for forward lanes, -1 for oncoming lanes."""
        return 1 if lane in self.forward_lanes else -1

    def lanes_in_same_group(self, lane):
        return self.forward_lanes if lane in self.forward_lanes else self.opposite_lanes

    # ------------------------------------------------------------------
    # Placement helpers (offsets)
    # ------------------------------------------------------------------
    def random_jitter(self, width):
        room = max(0.0, (self.lane_width - width) / 2.0 - 4.0)
        return random.uniform(-1.0, 1.0) * room * 0.5 * self.jitter_scale

    def jittered_lane_off(self, lane, width):
        return self.lane_center_off(lane) + self.random_jitter(width)

    def parked_off(self, lane, width):
        """Offset of a vehicle parked against the left kerb (lane 0) or the right kerb (last lane)."""
        if lane == 0:
            return -self.half_width + width / 2.0 + config.PARKED_EDGE_MARGIN
        return self.half_width - width / 2.0 - config.PARKED_EDGE_MARGIN