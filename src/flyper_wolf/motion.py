"""Versioned seeded planar trajectories and resolved motion provenance."""

from bisect import bisect_left, bisect_right

import numpy as np


MOTION_VERSIONS = ("smooth-sines-v2", "random-waypoints-v1", "constant-speed-waypoints-v1",
                   "smooth-reactive-plane-v1")
REACTIVE_PLANE_SETTINGS = {
    "profile": "correlated-random-steering-with-soft-reflections",
    "reference_speed_m_s": 3.66,
    "speed_semantics": "latent cruising reference; world speed varies and slows smoothly at reflections",
    "speed_multiplier_range": [0.65, 1.4],
    "turn_rate_limit_rad_s": 4.5,
    "command_hold_seconds": [0.3, 0.9],
    "turn_response_seconds": 0.18,
    "speed_response_seconds": 0.3,
    "integration_dt_seconds": 0.005,
    "interpolation": "cubic Hermite position with continuous velocity",
    "boundary_rule": "coordinate-wise sine soft reflection, no position clipping",
    "depth_motion": False,
}
CONSTANT_SPEED_SETTINGS = {
    "profile": "piecewise-linear-constant-speed",
    "speed_m_s": 3.66,
    "destination_distribution": "uniform across full room plane, sphere kept inside walls",
    "initial_position": "seeded sine phases scaled to full room plane",
    "turns": "instantaneous direction changes, no slowing or stopping",
    "depth_motion": False,
}
RANDOM_WAYPOINT_SETTINGS = {
    "profile": "quintic-rest-to-rest",
    "normalized_waypoint_range": [-1.0, 1.0],
    "sampled_duration_seconds": [0.5, 1.25],
    "reference_periods_seconds": [4.0, 3.0],
    "duration_rule": "max(sampled, 1.875*abs(delta)/omega, sqrt((10/sqrt(3))*abs(delta)/omega**2))",
    "depth_motion": False,
}


def motion_settings_for(version, scene):
    """Resolved settings for provenance; preserve old motion metadata."""
    if version == "random-waypoints-v1":
        return RANDOM_WAYPOINT_SETTINGS.copy()
    if version == "constant-speed-waypoints-v1":
        w, d, h = scene.room_dimensions
        r = scene.target_radius
        return {**CONSTANT_SPEED_SETTINGS, "x_bounds_m": [-w/2+r, w/2-r],
                "z_bounds_m": [r, h-r], "depth_y_m": d/8}
    if version == "smooth-reactive-plane-v1":
        w, d, h = scene.room_dimensions
        r = scene.target_radius
        return {**REACTIVE_PLANE_SETTINGS, "target_radius_m": r,
                "x_bounds_m": [-w/2+r, w/2-r],
                "z_bounds_m": [r, h-r], "depth_y_m": d/8}
    if version == "smooth-sines-v2":
        return None
    raise ValueError("Unknown trajectory version")


class SmoothReactivePlane:
    """Seeded curved travel with varying speed and smooth wall reversals.

    A latent, unbounded point follows a smoothly filtered random turn rate and
    cruising speed. Mapping its coordinates through sine folds that travel into
    the room: normal velocity fades to zero and reverses at a wall, while the
    other axis keeps moving. This is a spatial boundary map, not a periodic sine
    trajectory. There are no destinations or mandatory stops.

    Fixed-grid cached integration makes arbitrary fractional-time queries and
    out-of-order replay independent of the caller's sampling cadence. Hermite
    interpolation gives continuous position and velocity; acceleration is
    piecewise continuous. Returned kinematics are normalized room coordinates.
    """

    def __init__(self, phase, rng, half_extent):
        self.rng = rng
        self.half_extent = np.asarray(half_extent, dtype=float)
        if self.half_extent.shape != (2,) or not np.isfinite(self.half_extent).all() or np.any(self.half_extent <= 0):
            raise ValueError("Target plane must have positive finite extents")
        self.dt = REACTIVE_PLANE_SETTINGS["integration_dt_seconds"]
        self.reference_speed = REACTIVE_PLANE_SETTINGS["reference_speed_m_s"]
        self.heading = float(rng.uniform(-np.pi, np.pi))
        self.turn_rate = 0.0
        self.speed = self.reference_speed
        self.next_command_time = 0.0
        velocity = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.knots = [(self.half_extent * phase, velocity)]

    def _append(self):
        time = (len(self.knots)-1)*self.dt
        if time >= self.next_command_time:
            limit = REACTIVE_PLANE_SETTINGS["turn_rate_limit_rad_s"]
            self.desired_turn_rate = float(self.rng.uniform(-limit, limit))
            self.desired_speed = self.reference_speed * float(
                self.rng.uniform(*REACTIVE_PLANE_SETTINGS["speed_multiplier_range"]))
            self.next_command_time = time + float(
                self.rng.uniform(*REACTIVE_PLANE_SETTINGS["command_hold_seconds"]))
        previous_turn_rate = self.turn_rate
        turn_decay = np.exp(-self.dt/REACTIVE_PLANE_SETTINGS["turn_response_seconds"])
        speed_decay = np.exp(-self.dt/REACTIVE_PLANE_SETTINGS["speed_response_seconds"])
        self.turn_rate = self.desired_turn_rate + (self.turn_rate-self.desired_turn_rate)*turn_decay
        self.speed = self.desired_speed + (self.speed-self.desired_speed)*speed_decay
        self.heading += (previous_turn_rate+self.turn_rate)*self.dt/2
        velocity = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        position, previous_velocity = self.knots[-1]
        self.knots.append((position + (previous_velocity+velocity)*self.dt/2, velocity))

    def kinematics(self, seconds):
        """Normalized position, velocity and acceleration at any nonnegative time."""
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Motion time must be finite and nonnegative")
        index = int(np.floor(seconds/self.dt))
        while len(self.knots) <= index+1:
            self._append()
        p0, v0 = self.knots[index]
        p1, v1 = self.knots[index+1]
        u = (seconds-index*self.dt)/self.dt
        # Cubic Hermite coefficients; trapezoidal integration makes c3 zero
        # up to roundoff, with endpoint velocities shared between segments.
        c1 = self.dt*v0
        c2 = 3*(p1-p0)-self.dt*(2*v0+v1)
        c3 = 2*(p0-p1)+self.dt*(v0+v1)
        latent = p0 + u*(c1+u*(c2+u*c3))
        latent_velocity = (c1+u*(2*c2+3*u*c3))/self.dt
        latent_acceleration = (2*c2+6*u*c3)/self.dt**2
        angle = latent/self.half_extent
        sine, cosine = np.sin(angle), np.cos(angle)
        velocity = cosine*latent_velocity/self.half_extent
        acceleration = (cosine*latent_acceleration/self.half_extent
                        - sine*(latent_velocity/self.half_extent)**2)
        return sine, velocity, acceleration


class ConstantSpeedWaypoints:
    """Linear random destinations with fixed physical speed and sharp turns.

    Position is continuous. Velocity changes instantly at destinations; there is
    no finite acceleration bound at a turn. Coordinates are normalized so the
    simulation's full-room amplitudes map them into physical meters.
    """

    def __init__(self, phase, rng, half_extent):
        self.rng = rng
        self.initial = np.sin(phase)
        self.half_extent = np.asarray(half_extent, dtype=float)
        if self.half_extent.shape != (2,) or not np.isfinite(self.half_extent).all() or np.any(self.half_extent <= 0):
            raise ValueError("Target plane must have positive finite extents")
        self.speed = CONSTANT_SPEED_SETTINGS["speed_m_s"]
        self.segments = []
        self.end_times = []

    def _append(self):
        start_time = self.end_times[-1] if self.end_times else 0.0
        start = self.segments[-1][2] if self.segments else self.initial
        while True:
            end = self.rng.uniform(-1, 1, 2)
            distance = np.linalg.norm((end-start)*self.half_extent)
            if distance > 1e-12:
                break
        duration = float(distance/self.speed)
        self.segments.append((start_time, start, end, duration))
        self.end_times.append(start_time+duration)

    def kinematics(self, seconds):
        """Normalized position and right-hand velocity, including at turns."""
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Motion time must be finite and nonnegative")
        while not self.end_times or seconds >= self.end_times[-1]:
            self._append()
        start_time, start, end, duration = self.segments[bisect_right(self.end_times, seconds)]
        u = np.clip((seconds-start_time)/duration, 0, 1)
        return start+(end-start)*u, (end-start)/duration


class RandomWaypoints:
    """Two normalized axes; cached segments make evaluation order independent.

    A quintic segment stops with zero velocity and acceleration at each random
    waypoint. Durations enforce each axis's reference sine velocity/acceleration
    ceilings. Bounds on normalized position also keep every segment in the room.
    """

    def __init__(self, phase, rng):
        self.rng = rng
        self.initial = np.sin(phase)
        self.omega = 2 * np.pi / np.array(RANDOM_WAYPOINT_SETTINGS["reference_periods_seconds"])
        self.segments = []
        self.end_times = []

    def _append(self):
        start_time = self.end_times[-1] if self.end_times else 0.0
        start = self.segments[-1][2] if self.segments else self.initial
        end = self.rng.uniform(-1, 1, 2)
        delta = np.abs(end - start)
        duration = max(self.rng.uniform(*RANDOM_WAYPOINT_SETTINGS["sampled_duration_seconds"]),
                       float(np.max(1.875 * delta / self.omega)),
                       float(np.max(np.sqrt((10 / np.sqrt(3)) * delta / self.omega**2))))
        self.segments.append((start_time, start, end, duration))
        self.end_times.append(start_time + duration)

    def kinematics(self, seconds):
        """Return normalized position, velocity and acceleration at any time."""
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Motion time must be finite and nonnegative")
        while not self.end_times or seconds > self.end_times[-1]:
            self._append()
        start_time, start, end, duration = self.segments[bisect_left(self.end_times, seconds)]
        u = np.clip((seconds - start_time) / duration, 0, 1)
        delta = end - start
        position = start + delta * (10*u**3 - 15*u**4 + 6*u**5)
        velocity = delta * (30*u**2 - 60*u**3 + 30*u**4) / duration
        acceleration = delta * (60*u - 180*u**2 + 120*u**3) / duration**2
        return position, velocity, acceleration
