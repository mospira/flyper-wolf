"""Plain proportional steering from estimated camera-local bearing only."""

import numpy as np


class ProportionalController:
    def __init__(self, gain_yaw=24.0, gain_pitch=24.0, max_angular_speed=60.0):
        self.gains = np.asarray([gain_yaw, gain_pitch], dtype=np.float64)
        if not np.isfinite(self.gains).all() or np.any(self.gains <= 0):
            raise ValueError("Controller gains must be finite and positive")
        if not np.isfinite(max_angular_speed) or max_angular_speed <= 0:
            raise ValueError("Maximum angular speed must be finite and positive")
        self.max_angular_speed = float(max_angular_speed)
        self.last_saturation = np.zeros(2, dtype=bool)

    def action(self, bearing_deg):
        bearing = np.asarray(bearing_deg, dtype=np.float64).copy()
        if bearing.shape != (2,) or not np.isfinite(bearing).all():
            raise ValueError("Expected two finite predicted bearing angles")
        # Match the retained source plain controller's corrected-bearing contract.
        if not -180 <= bearing[0] <= 180:
            original = bearing[0]
            bearing[0] = (original+180) % 360-180
            if bearing[0] == -180 and original > 0:
                bearing[0] = 180
        bearing[1] = np.clip(bearing[1], -90, 90)
        raw = self.gains * bearing
        self.last_saturation = np.abs(raw/self.max_angular_speed) > 1
        return np.clip(raw, -self.max_angular_speed, self.max_angular_speed)/self.max_angular_speed
