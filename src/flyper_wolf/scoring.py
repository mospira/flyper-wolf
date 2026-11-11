"""Shared geometric score, integrated in simulation time without rendering."""

import numpy as np


SCORING_VERSION = "angular-ray-midpoint-v1"


def aim_geometry(frame, config):
    """Angular center error (degrees) and whether the forward ray hits the sphere."""
    direction = np.asarray(frame.target_position) - np.asarray(config.camera_position)
    distance = np.linalg.norm(direction)
    yaw, pitch = np.radians([frame.yaw, frame.pitch])
    forward = np.array([np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    angle = np.arccos(np.clip(np.dot(forward, direction) / distance, -1, 1))
    radius = np.arcsin(min(1.0, config.target_radius / distance))
    return float(np.degrees(angle)), bool(angle <= radius)


def score_interval(simulation, before, action, substeps=10):
    """Midpoint quadrature over the action interval, 1 ms by default.

    The camera turns at the commanded angular velocity, respecting pitch limits.
    This scores interval interiors, not display frames or just endpoint hits.
    """
    if type(substeps) is not int or substeps < 1:
        raise ValueError("Scoring substeps must be a positive integer")
    config = simulation.config
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("Invalid scoring action")
    errors, hits = [], []
    for fraction in (np.arange(substeps) + 0.5) / substeps:
        delta = action * config.max_angular_speed * config.observation_dt * fraction
        frame = simulation.frame_at(before.tick + fraction, before.yaw + delta[0],
                                    np.clip(before.pitch + delta[1], -config.pitch_limit, config.pitch_limit))
        error, hit = aim_geometry(frame, config)
        errors.append(error)
        hits.append(hit)
    return {"mean_error_deg": float(np.mean(errors)),
            "on_target_seconds": float(np.mean(hits) * config.observation_dt)}
