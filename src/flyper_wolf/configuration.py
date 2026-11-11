"""Small YAML interface with explicit overrides and no hidden development paths."""

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import yaml

from .scene import SceneConfig
from .motion import MOTION_VERSIONS


DEFAULTS = {
    "schema_version": 1,
    "model_bundle": "../models/release",
    "scene": {"target_radius": 0.288},
    "episode": {"duration_seconds": 10.0, "target_seed": 302000001,
                "camera_seed": 303000001, "target_speed": 1.1,
                "trajectory_version": "smooth-reactive-plane-v1",
                "initial_yaw": None, "initial_pitch": None},
    "runtime": {"device": "cpu", "cpu_threads": 4, "seed": 74, "integration_dt": 0.005, "warmup_seconds": 1.0,
                "initial_zero_ticks": 20, "scoring_substeps": 10,
                "video": True, "video_fps": 50, "record_neural": False},
    "controller": {"gain_yaw": 24.0, "gain_pitch": 24.0},
}


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def load_config(path, overrides=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError(f"Config must be a mapping with keys {sorted(DEFAULTS)}")
    config = deepcopy(DEFAULTS)
    for name, value in raw.items():
        if isinstance(config[name], dict):
            allowed = set(asdict(SceneConfig())) if name == "scene" else set(config[name])
            if not isinstance(value, dict) or set(value) - allowed:
                raise ValueError(f"Unknown or invalid {name} settings; allowed: {sorted(allowed)}")
            config[name].update(value)
        else:
            config[name] = value
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        group, key = dotted.split(".")
        config[group][key] = value
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Only config schema_version 1 is supported")
    bundle = config["model_bundle"]
    if not isinstance(bundle, str) or not bundle.strip():
        raise ValueError("model_bundle must be a nonempty path")
    config["model_bundle"] = str((path.parent / bundle).resolve())
    scene = SceneConfig(**config["scene"])
    config["scene"] = asdict(scene)
    episode, runtime = config["episode"], config["runtime"]
    for key in ("target_seed", "camera_seed"):
        if type(episode[key]) is not int or not 0 <= episode[key] < 2**32:
            raise ValueError(f"{key} must be a uint32 integer")
    if episode["trajectory_version"] not in MOTION_VERSIONS:
        raise ValueError(f"trajectory_version must be one of {MOTION_VERSIONS}")
    for name in ("duration_seconds", "target_speed"):
        positive(episode[name], name)
    ticks = episode["duration_seconds"] / scene.observation_dt
    if not math.isclose(ticks, round(ticks), abs_tol=1e-8):
        raise ValueError("duration_seconds must be a multiple of 0.01")
    poses = [episode["initial_yaw"], episode["initial_pitch"]]
    if any(v is None for v in poses) != all(v is None for v in poses):
        raise ValueError("Set initial_yaw and initial_pitch together, or omit both for seeded pose")
    if poses[0] is not None:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in poses):
            raise ValueError("Initial angles must be finite numbers")
        if abs(poses[1]) > scene.pitch_limit:
            raise ValueError("initial_pitch exceeds the scene pitch limit")
    for key in ("initial_zero_ticks", "scoring_substeps", "video_fps"):
        if type(runtime[key]) is not int or runtime[key] < (0 if key == "initial_zero_ticks" else 1):
            raise ValueError(f"{key} must be a valid integer")
    if runtime["initial_zero_ticks"] >= round(ticks):
        raise ValueError("Episode must extend beyond the initial zero-action period")
    if runtime["initial_zero_ticks"] != 20 or runtime["scoring_substeps"] != 10:
        raise ValueError("This inference contract fixes initial_zero_ticks=20 and scoring_substeps=10")
    if type(runtime["cpu_threads"]) is not int or runtime["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if type(runtime["seed"]) is not int or not 0 <= runtime["seed"] < 2**32:
        raise ValueError("runtime seed must be a uint32 integer")
    if 100 % runtime["video_fps"]:
        raise ValueError("video_fps must divide the 100 Hz observation rate")
    if round(ticks) % (100 // runtime["video_fps"]):
        raise ValueError("Episode duration must contain an exact number of video frames")
    if type(runtime["video"]) is not bool:
        raise ValueError("video must be true or false")
    if type(runtime["record_neural"]) is not bool:
        raise ValueError("record_neural must be true or false")
    if runtime["video"] and any(size % 2 for size in scene.image_size):
        raise ValueError("Video output requires even image dimensions for yuv420p encoding")
    import re
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"]):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    for key in ("integration_dt", "warmup_seconds"):
        positive(runtime[key], key)
    dt = runtime["integration_dt"]
    if dt > scene.observation_dt or any(not math.isclose(v / dt, round(v / dt), abs_tol=1e-8)
                                       for v in (scene.observation_dt, runtime["warmup_seconds"])):
        raise ValueError("integration_dt must divide observation and warmup intervals")
    for key, gain in config["controller"].items():
        positive(gain, key)
    return config
