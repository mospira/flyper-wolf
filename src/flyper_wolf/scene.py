"""Validated, versioned settings shared by capture and replay."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    schema_version: int = 1
    observation_dt: float = 0.01
    room_dimensions: tuple[float, float, float] = (12.0, 16.0, 8.0)
    wall_gray: tuple[float, ...] = (0.20, 0.25, 0.28, 0.32, 0.16, 0.36)
    camera_position: tuple[float, float, float] = (0.0, -6.0, 4.0)
    horizontal_fov: float = 90.0
    image_size: tuple[int, int] = (384, 384)
    target_radius: float = 0.288
    target_gray: float = 1.0
    max_angular_speed: float = 60.0
    pitch_limit: float = 80.0
    receptor_extent: int = 15
    pooling_kernel: int = 11

    def __post_init__(self):
        for name in ("room_dimensions", "wall_gray", "camera_position", "image_size"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        numbers = [self.observation_dt, self.horizontal_fov, self.target_radius,
                   self.target_gray, self.max_angular_speed, self.pitch_limit,
                   *self.room_dimensions, *self.wall_gray, *self.camera_position]
        if not np.isfinite(numbers).all():
            raise ValueError("Scene settings must be finite")
        if self.schema_version != 1 or self.observation_dt != 0.01:
            raise ValueError("Schema 1 uses the project-selected 10 ms observation interval")
        if len(self.room_dimensions) != 3 or min(self.room_dimensions) <= 0:
            raise ValueError("Room dimensions must be three positive lengths")
        if len(self.camera_position) != 3:
            raise ValueError("Camera position must have three coordinates")
        if len(self.wall_gray) != 6 or not all(0 <= x <= 1 for x in self.wall_gray):
            raise ValueError("Six wall intensities in [0, 1] are required")
        if not 0 < self.horizontal_fov < 170 or not 0 < self.pitch_limit < 90:
            raise ValueError("Invalid camera FOV or pitch limit")
        if not 0 <= self.target_gray <= 1 or self.target_radius <= 0 or self.max_angular_speed <= 0:
            raise ValueError("Invalid target or angular-speed settings")
        if len(self.image_size) != 2 or any(type(x) is not int or x < 1 for x in self.image_size):
            raise ValueError("Image dimensions must be positive integers")
        if self.receptor_extent != 15:
            raise ValueError("This adapter targets the 721-element FlyVis input")
        if type(self.pooling_kernel) is not int or self.pooling_kernel < 1 or self.pooling_kernel % 2 != 1:
            raise ValueError("Pooling kernel must be a positive odd integer")
        footprint = (2 * self.receptor_extent + 1) * self.pooling_kernel
        if min(self.image_size) < footprint:
            raise ValueError("Image must contain the complete sampling footprint without padding")
        width, depth, height = self.room_dimensions
        x, y, z = self.camera_position
        if not (-width / 2 < x < width / 2 and -depth / 2 < y < depth / 2 and 0 < z < height):
            raise ValueError("Camera must be inside the room")
