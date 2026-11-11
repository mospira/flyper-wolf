"""An explicitly stepped kinematic scene, without a physics solver."""

from dataclasses import dataclass

import numpy as np

from flyper_wolf.scene import SceneConfig
from flyper_wolf.motion import MOTION_VERSIONS, RandomWaypoints, ConstantSpeedWaypoints, SmoothReactivePlane


@dataclass(frozen=True)
class WorldFrame:
    tick: int
    target_position: tuple[float, float, float]
    yaw: float
    pitch: float


class Simulation:
    """X right, Y forward, Z up; positive yaw right, positive pitch up.

    reset() establishes observation 0. step(action) holds angular velocity for
    dt, advances the target to the new tick, then returns the next world frame.
    Returned world frames are for the renderer/logger, never the visual adapter.
    """

    trajectory_version = "smooth-sines-v2"

    def __init__(self, config: SceneConfig, seed: int = 7, target_speed: float = 1.0,
                 trajectory_version=None):
        if not np.isfinite(target_speed) or target_speed <= 0:
            raise ValueError("Target speed multiplier must be finite and positive")
        self.config = config
        self.seed = seed
        self.target_speed = float(target_speed)
        self.trajectory_version = type(self).trajectory_version if trajectory_version is None else trajectory_version
        if self.trajectory_version not in MOTION_VERSIONS:
            raise ValueError(f"Unknown trajectory version: {self.trajectory_version}")
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(-np.pi, np.pi, 2)
        w, d, h = config.room_dimensions
        # Entire trajectory remains inside the room, including the sphere surface.
        self.center = np.array([0.0, d / 8, h / 2])
        self.amplitude = np.array([w * 0.24, 0.0, h * 0.18])
        if self.trajectory_version == "constant-speed-waypoints-v1":
            self.amplitude = np.array([w/2-config.target_radius, 0.0, h/2-config.target_radius])
        elif self.trajectory_version == "smooth-reactive-plane-v1":
            self.amplitude = np.array([w/2-config.target_radius, 0.0, h/2-config.target_radius])
        room_margin = np.array([w / 2, d / 2 - abs(self.center[1]), h / 2])
        if (np.any(self.amplitude + config.target_radius > room_margin)
                or np.any(self.amplitude[[0, 2]] <= 0)):
            raise ValueError("Target trajectory does not fit inside the room")
        self.random_motion = None
        if self.trajectory_version == "random-waypoints-v1":
            self.random_motion = RandomWaypoints(self.phase, rng)
        elif self.trajectory_version == "constant-speed-waypoints-v1":
            self.random_motion = ConstantSpeedWaypoints(self.phase, rng, self.amplitude[[0, 2]])
        elif self.trajectory_version == "smooth-reactive-plane-v1":
            self.random_motion = SmoothReactivePlane(self.phase, rng, self.amplitude[[0, 2]])
        self.reset()

    def reset(self) -> WorldFrame:
        self.tick = 0
        self.yaw = self.pitch = 0.0
        return self.snapshot()

    def snapshot(self) -> WorldFrame:
        return self.frame_at(self.tick, self.yaw, self.pitch)

    def frame_at(self, tick, yaw, pitch) -> WorldFrame:
        """Evaluate the shared trajectory at a possibly fractional diagnostic tick."""
        # Scale only target trajectory time; observation and camera clocks stay fixed.
        t = tick * self.config.observation_dt * self.target_speed
        if self.random_motion is None:
            motion = [np.sin(2 * np.pi * t / 4 + self.phase[0]), 0.0,
                      np.sin(2 * np.pi * t / 3 + self.phase[1])]
        else:
            position = self.random_motion.kinematics(t)[0]
            motion = [position[0], 0.0, position[1]]
        pos = self.center + self.amplitude * np.asarray(motion)
        return WorldFrame(tick, tuple(pos), yaw, pitch)

    def step(self, action) -> WorldFrame:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Action must contain two finite values in [-1, 1]")
        delta = action * self.config.max_angular_speed * self.config.observation_dt
        self.yaw = float((self.yaw + delta[0] + 180) % 360 - 180)
        self.pitch = float(np.clip(self.pitch + delta[1], -self.config.pitch_limit, self.config.pitch_limit))
        self.tick += 1
        return self.snapshot()
