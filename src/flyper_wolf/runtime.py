"""Closed-loop inference and episode exports, without training infrastructure."""

import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from flyper_wolf.controller import ProportionalController
from flyper_wolf.estimator import direction_bearings, load_frozen_estimator
from flyper_wolf.metrics import _interval_samples, angular_error_deg, camera_local_bearing, summarize_tracking
from flyper_wolf.scene import SceneConfig
from flyper_wolf.scoring import SCORING_VERSION, score_interval
from flyper_wolf.simulation import Simulation


def _configure_runtime(settings):
    # Set before importing Torch, and construct Torch before Panda3D on Windows.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    device = settings.get("device", "cpu")
    seed = settings.get("seed", 74)
    torch.set_num_threads(settings.get("cpu_threads", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return {"device": device, "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": True, "tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _display(rgb, row, hit_seconds, dt, label=None):
    """Annotate a copied RGB observation after inference; never feed it back."""
    from PIL import Image, ImageDraw

    display = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(display)
    width, height = display.size
    x, y = width//2, height//2
    draw.line((x-9, y, x+9, y), fill=(255, 80, 80), width=2)
    draw.line((x, y-9, x, y+9), fill=(255, 80, 80), width=2)
    draw.rectangle((0, 0, width, 51 if label else 31), fill=(12, 16, 23))
    percent = 100*hit_seconds/((row["tick"]+1)*dt)
    draw.text((8, 9), f"{row['time_seconds']:.2f}s  |  time on target {percent:.1f}%", fill="white")
    if label:
        draw.text((8, 30), label, fill="white")
    return np.asarray(display)


def run_episode(config: dict, output: Path):
    """Generate one real FlyVis episode; return metrics and runtime provenance.

    The config contains scene, episode, runtime and controller groups plus
    verified artifacts (estimator_manifest and flyvis_model_dir). Each circuit
    observation advances 10 ms; the first 20 ticks have zero commands and no
    estimator calls. Truth is used only after commands, for scoring and logging.
    """
    output = Path(output)
    settings, episode = config.get("runtime", {}), config["episode"]
    scene = SceneConfig(**config.get("scene", {}))
    ticks_float = episode["duration_seconds"] / scene.observation_dt
    ticks = round(ticks_float)
    startup, substeps = settings.get("initial_zero_ticks", 20), settings.get("scoring_substeps", 10)
    if (not math.isfinite(ticks_float) or not math.isclose(ticks, ticks_float, abs_tol=1e-9)
            or ticks <= startup or startup != 20 or substeps != 10):
        raise ValueError("Episodes require whole 10-ms ticks, duration > 0.2s, 20 startup ticks and 10 scoring substeps")
    video_enabled, fps = settings.get("video", True), settings.get("video_fps", 50)
    if type(fps) is not int or fps < 1 or 100 % fps:
        raise ValueError("Video FPS must be a positive integer divisor of 100")
    artifacts = config["artifacts"]
    started = time.perf_counter()
    runtime_identity = _configure_runtime(settings)
    from flyper_wolf.fly import FrozenFly
    from flyper_wolf.preprocessing import VisualInput
    from flyper_wolf.rendering import SceneRenderer

    fly = FrozenFly(artifacts["flyvis_model_dir"], device=runtime_identity["device"],
                    observation_dt=scene.observation_dt, integration_dt=settings.get("integration_dt", .005),
                    warmup_seconds=settings.get("warmup_seconds", 1.0))
    predictor = load_frozen_estimator(artifacts["estimator_manifest"], device=runtime_identity["device"])
    predictor.validate_output_order(fly)
    frozen_hash, estimator_hash = fly.parameter_hash(), predictor.parameter_hash()
    policy = ProportionalController(**config.get("controller", {}), max_angular_speed=scene.max_angular_speed)
    from .media_labels import pair_label
    display_label = pair_label(predictor.identity, config.get("controller", {}), scene.max_angular_speed)
    adapter = VisualInput(scene.image_size, scene.receptor_extent, scene.pooling_kernel)
    simulation = Simulation(scene, seed=episode["target_seed"], target_speed=episode.get("target_speed", 1.1),
                            trajectory_version=episode.get("trajectory_version", "smooth-reactive-plane-v1"))
    rng = np.random.default_rng(episode["camera_seed"])
    seeded_yaw, seeded_pitch = float(rng.uniform(-25, 25)), float(rng.uniform(-16, 16))
    simulation.yaw = seeded_yaw if episode.get("initial_yaw") is None else episode["initial_yaw"]
    simulation.pitch = seeded_pitch if episode.get("initial_pitch") is None else episode["initial_pitch"]
    if not np.isfinite([simulation.yaw, simulation.pitch]).all() or abs(simulation.pitch) > scene.pitch_limit:
        raise ValueError("Initial camera pose must be finite and respect pitch limits")
    resolved_episode = {**episode, "initial_yaw": simulation.yaw, "initial_pitch": simulation.pitch,
                        "episode_ticks": ticks, "target_speed": simulation.target_speed,
                        "trajectory_version": simulation.trajectory_version}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("trace.csv", "metrics.json", "episode.mp4"):
        if (output/name).exists():
            raise FileExistsError(f"Refusing to overwrite episode artifact: {output/name}")
    renderer, video, recorder = None, None, None
    recording_complete, recording_error = False, None
    recording_exception = None
    errors, hits, estimator_errors, records, velocities = [], [], [], [], []
    zero_count, saturated_updates = 0, 0
    saturated_axis_updates = np.zeros(2, dtype=np.int64)
    video_frames, cumulative_hit_seconds = 0, 0.0
    try:
        renderer = SceneRenderer(scene)
        if video_enabled:
            import av
            video = av.open(str(output / "episode.mp4"), mode="w")
            stream = video.add_stream("libx264", rate=fps)
            stream.width, stream.height = scene.image_size
            stream.pix_fmt = "yuv420p"
        baseline = fly.reset()
        if settings.get("record_neural", False):
            from flyper_wolf.neural_recording import NeuralRecorder
            recorder = NeuralRecorder(output, fly, baseline, ticks=ticks, observation_dt=scene.observation_dt,
                                      estimator_identity=predictor.identity, episode=resolved_episode,
                                      frozen_hash=frozen_hash, estimator_hash=estimator_hash)
        frame = simulation.snapshot()
        action, prediction = np.zeros(2, dtype=np.float64), None
        for tick in range(ticks):
            causal_time = tick*scene.observation_dt
            rgb = renderer.capture(frame)
            receptors = adapter(rgb)
            full_activity = fly.step(receptors)
            activity = fly.outputs(full_activity)
            updated, estimator_error = tick >= startup, None
            if updated:
                predictions, zeros = direction_bearings(predictor.predict(activity.detach().cpu().numpy()), predictor.fallback)
                if predictions.shape != (1, 2):
                    raise ValueError("One circuit observation must produce exactly one bearing")
                prediction = predictions[0]
                action = policy.action(prediction)
                zero_count += zeros
                saturated_axis_updates += policy.last_saturation.astype(np.int64)
                saturated_updates += int(policy.last_saturation.any())
                # Simulator truth is first consulted after prediction and action.
                truth = np.asarray([camera_local_bearing(frame, scene)])
                estimator_error = float(angular_error_deg(prediction[None], truth)[0])
                estimator_errors.append(estimator_error)
            if recorder is not None:
                recorder.append(full_activity, receptors)
            score = score_interval(simulation, frame, action, substeps)
            interval_errors, interval_hits = _interval_samples(simulation, frame, action, substeps)
            if not (np.isclose(np.mean(interval_errors), score["mean_error_deg"])
                    and np.isclose(np.mean(interval_hits)*scene.observation_dt, score["on_target_seconds"])):
                raise RuntimeError("Scoring samples disagree with shared score_interval")
            errors.extend(interval_errors)
            hits.extend(interval_hits)
            velocities.append(action*scene.max_angular_speed)
            row = {"tick": tick, "time_seconds": causal_time, "prediction_updated": updated, "action_updated": updated,
                   "predicted_horizontal_deg": None if prediction is None else float(prediction[0]),
                   "predicted_vertical_deg": None if prediction is None else float(prediction[1]),
                   "estimator_error_deg": estimator_error, "action_yaw": float(action[0]),
                   "action_pitch": float(action[1]), "yaw_deg": frame.yaw, "pitch_deg": frame.pitch,
                   "true_mean_error_deg": score["mean_error_deg"], "on_target_seconds": score["on_target_seconds"],
                   "raw_saturation_yaw": bool(policy.last_saturation[0]) if updated else None,
                   "raw_saturation_pitch": bool(policy.last_saturation[1]) if updated else None}
            records.append(row)
            cumulative_hit_seconds += score["on_target_seconds"]
            if video is not None and tick % (100//fps) == 0:
                image = _display(rgb, row, cumulative_hit_seconds, scene.observation_dt, display_label)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    video.mux(packet)
                video_frames += 1
            frame = simulation.step(action)
        if video is not None:
            for packet in stream.encode():
                video.mux(packet)
        if fly.parameter_hash() != frozen_hash or predictor.parameter_hash() != estimator_hash:
            raise RuntimeError("Frozen circuit or estimator parameters changed during inference")
        recording_complete = True
    except Exception as error:
        recording_error = str(error)
        recording_exception = error
        raise
    finally:
        try:
            if recorder is not None:
                try:
                    recorder.close(complete=recording_complete, error=recording_error)
                except Exception as close_error:
                    if recording_exception is None:
                        raise
                    recording_exception.add_note(f"Recording cleanup also failed: {close_error}")
        finally:
            try:
                if video is not None:
                    video.close()
            finally:
                if renderer is not None:
                    renderer.close()
    metrics = summarize_tracking(errors, np.asarray(hits, dtype=bool), scene.observation_dt/substeps,
                                 estimator_errors, zero_count)
    updates = ticks-startup
    velocity = np.asarray(velocities, dtype=np.float64)
    metrics.update(time_on_target_pct=100*metrics["hit_fraction"], time_on_target_percent=100*metrics["hit_fraction"],
                   circuit_steps=ticks, prediction_updates=updates, action_updates=updates,
                   action_saturation_fraction=saturated_updates/updates,
                   yaw_saturation_fraction=float(saturated_axis_updates[0]/updates),
                   pitch_saturation_fraction=float(saturated_axis_updates[1]/updates),
                   mean_yaw_velocity_deg_per_second=float(velocity[:, 0].mean()),
                   mean_pitch_velocity_deg_per_second=float(velocity[:, 1].mean()),
                   rms_yaw_velocity_deg_per_second=float(np.sqrt(np.mean(velocity[:, 0]**2))),
                   rms_pitch_velocity_deg_per_second=float(np.sqrt(np.mean(velocity[:, 1]**2))))
    with (output/"trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _write_json(output/"metrics.json", metrics)
    return {"metrics": metrics, "runtime": runtime_identity, "flyvis": fly.identity,
            "estimator": predictor.identity, "scene": asdict(scene), "episode": resolved_episode,
            "scoring_version": SCORING_VERSION, "frozen_parameters_sha256": frozen_hash,
            "estimator_parameters_sha256": estimator_hash, "frozen_parameters_unchanged": True,
            "neural_recording": None if recorder is None else {
                "manifest": "neural/manifest.json", "frames": recorder.frames,
                "complete": recorder.manifest["complete"]},
            "video_frames": video_frames, "wall_seconds": time.perf_counter()-started}
