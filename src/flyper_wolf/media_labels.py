"""Presentation names derived from the actual estimator and controller settings."""

import json
from pathlib import Path

import yaml

from .neural_recording import _hash


def pair_label(estimator, controller, max_angular_speed=60):
    known = "858eaff1cbc17bf021fd66dec0509491da007f322a7471930dd83926c5d771d5"
    estimator_name = "fw-est342" if estimator.get("checkpoint_sha256") == known else "estimator"
    controller_name = "fw-cont24" if (
        controller.get("gain_yaw", 24) == 24 and controller.get("gain_pitch", 24) == 24
        and max_angular_speed == 60) else "custom proportional controller"
    return f"{estimator_name} + {controller_name}"


def recording_label(run, recording):
    run = Path(run)
    config_path, metadata_path = run / "resolved-config.yaml", run / "metadata.json"
    if not config_path.exists() or not metadata_path.exists():
        return "Recorded estimator + controller"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("output_sha256", {}).get("resolved-config.yaml") != _hash(config_path):
        raise ValueError("Cannot label media: episode configuration checksum mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return pair_label(recording["manifest"]["estimator"], config.get("controller", {}),
                      config.get("scene", {}).get("max_angular_speed", 60))
