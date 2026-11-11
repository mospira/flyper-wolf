"""Display-only geometry and full-duration midpoint tracking metrics."""

import numpy as np

from flyper_wolf.scoring import aim_geometry

def camera_local_bearing(frame, scene):
    """Azimuth right / elevation up in the camera basis, in signed degrees.

    Elevation is atan2(up, horizontal length), so vector conversion remains
    meaningful for behind-camera targets and avoids subtracting world angles.
    """
    direction = np.asarray(frame.target_position) - np.asarray(scene.camera_position)
    yaw, pitch = np.radians([frame.yaw, frame.pitch])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    forward = np.array([np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)])
    up = np.cross(right, forward)
    x, y, z = np.array([right, forward, up]) @ direction
    if not np.isfinite([x, y, z]).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Bearing requires a finite nonzero camera-to-target vector")
    return np.degrees([np.arctan2(x, y), np.arctan2(z, np.hypot(x, y))])

def bearing_vectors(bearings):
    values = np.asarray(bearings, dtype=np.float64)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ValueError("Bearings must be finite [..., 2] degrees")
    horizontal, vertical = np.moveaxis(np.radians(values), -1, 0)
    return np.stack([np.sin(horizontal)*np.cos(vertical),
                     np.cos(horizontal)*np.cos(vertical), np.sin(vertical)], axis=-1)

def angular_error_deg(predicted_bearings, true_bearings):
    predicted, target = bearing_vectors(predicted_bearings), bearing_vectors(true_bearings)
    # atan2 is better behaved than arccos near exact agreement.
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(predicted, target), axis=-1),
                                np.sum(predicted * target, axis=-1)))

def summarize_tracking(errors, hits, sample_dt, estimator_errors=(), zero_norm_fallbacks=0):
    """Full-duration midpoint metrics; reacquisition is a completed natural miss.

    Hit events use scoring-bin starts (1 ms resolution in this protocol).
    Initial acquisition and trailing, censored misses are not reacquisitions.
    """
    errors = np.asarray(errors, dtype=float)
    hits = np.asarray(hits)
    estimated = np.asarray(estimator_errors, dtype=float)
    if (errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all()
            or hits.shape != errors.shape or hits.dtype != np.bool_
            or not np.isfinite(sample_dt) or sample_dt <= 0
            or estimated.ndim != 1 or not np.isfinite(estimated).all()):
        raise ValueError("Expected aligned finite scoring samples, boolean hits and positive sample_dt")
    first = np.flatnonzero(hits)
    seen_hit, miss_start, longest = False, None, 0
    reacquisitions = []
    for index, hit in enumerate(hits):
        if hit:
            if miss_start is not None:
                length = index - miss_start
                longest = max(longest, length)
                if seen_hit:
                    reacquisitions.append(float(length * sample_dt))
                miss_start = None
            seen_hit = True
        elif miss_start is None:
            miss_start = index
    censored = 0 if miss_start is None else len(hits) - miss_start
    longest = max(longest, censored)
    return {"duration_seconds": float(len(hits)*sample_dt),
            "on_target_seconds": float(hits.sum()*sample_dt), "hit_fraction": float(hits.mean()),
            "mean_error_deg": float(errors.mean()), "p90_error_deg": float(np.percentile(errors, 90)),
            "over_45_deg_fraction": float(np.mean(errors > 45)),
            "first_hit_seconds": float(first[0]*sample_dt) if len(first) else None,
            "longest_miss_seconds": float(longest*sample_dt),
            "reacquisition_seconds": reacquisitions, "reacquisition_count": len(reacquisitions),
            "mean_reacquisition_seconds": float(np.mean(reacquisitions)) if reacquisitions else None,
            "trailing_miss_seconds": float(censored*sample_dt),
            "estimator_mean_error_deg": float(estimated.mean()) if len(estimated) else None,
            "estimator_p90_error_deg": float(np.percentile(estimated, 90)) if len(estimated) else None,
            "estimator_samples": len(estimated), "zero_norm_fallbacks": zero_norm_fallbacks}

def _interval_samples(simulation, before, action, substeps):
    errors, hits = [], []
    scene = simulation.config
    for fraction in (np.arange(substeps) + 0.5)/substeps:
        delta = action*scene.max_angular_speed*scene.observation_dt*fraction
        frame = simulation.frame_at(before.tick+fraction, before.yaw+delta[0],
                                    np.clip(before.pitch+delta[1], -scene.pitch_limit, scene.pitch_limit))
        error, hit = aim_geometry(frame, scene)
        errors.append(error)
        hits.append(hit)
    return errors, hits
