"""Commands for downloading models, checking setup, and generating episodes."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import yaml

from .configuration import load_config


def _json(value):
    print(json.dumps(value, indent=2, default=str, allow_nan=False))


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def code_identity():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"revision": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
            "source_sha256": {p.relative_to(package).as_posix(): _hash(p)
                              for p in sorted(package.rglob("*.py"))}}


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("flyper-wolf", "numpy", "PyYAML", "torch", "torchvision", "flyvis", "panda3d", "pillow", "imageio", "av"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _config(args):
    overrides = {}
    for name, group, key in (("duration", "episode", "duration_seconds"),
                             ("target_seed", "episode", "target_seed"),
                             ("camera_seed", "episode", "camera_seed"),
                             ("target_speed", "episode", "target_speed"),
                             ("device", "runtime", "device")):
        overrides[f"{group}.{key}"] = getattr(args, name, None)
    if getattr(args, "no_video", False):
        overrides["runtime.video"] = False
    if getattr(args, "record_neural", False):
        overrides["runtime.record_neural"] = True
    config = load_config(args.config, overrides)
    if getattr(args, "model_bundle", None):
        config["model_bundle"] = str(Path(args.model_bundle).resolve())
    return config


def _bundle(config):
    from .artifacts import verify_bundle
    bundle = verify_bundle(config["model_bundle"])
    config["artifacts"] = {"estimator_manifest": str(bundle.estimator_manifest),
                           "flyvis_model_dir": str(bundle.flyvis_dir)}
    return bundle


def doctor(args):
    config = _config(args)
    report = {"config": str(Path(args.config).resolve()), "versions": versions(), "problems": []}
    for module in ("torch", "flyvis", "panda3d", "imageio", "av"):
        if importlib.util.find_spec(module) is None:
            report["problems"].append(f"Missing {module}; install the inference extra")
    try:
        bundle = _bundle(config)
        report["bundle"] = {"release_id": bundle.release_id, "manifest_sha256": _hash(bundle.manifest_path)}
    except (ValueError, OSError) as error:
        report["problems"].append(str(error))
    if importlib.util.find_spec("torch") is not None:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if config["runtime"]["device"].startswith("cuda") and not report["cuda_available"]:
            report["problems"].append("CUDA requested but unavailable; select cpu or install a compatible CUDA build")
    report["status"] = "ready" if not report["problems"] else "incomplete"
    report["scope"] = "Dependency and artifact preflight; OpenGL rendering and model execution are checked by run"
    _json(report)
    return 0 if not report["problems"] else 1


def run(args):
    config = _config(args)
    bundle = _bundle(config)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    if args.dry_run:
        _json({"status": "validated", "inference_executed": False, "output": output,
               "release_id": bundle.release_id, "resolved_config": config})
        return 0
    from .runtime import run_episode
    started = time.perf_counter()
    code = code_identity()
    try:
        result = run_episode(config, output)
        (output / "resolved-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        metadata = {"schema_version": 1, "status": "complete", "purpose": "inference-demonstration",
                    "elapsed_seconds": time.perf_counter() - started, "versions": versions(), "code": code,
                    "model_bundle": {"release_id": bundle.release_id,
                                     "manifest_sha256": _hash(bundle.manifest_path),
                                     "manifest": bundle.manifest}, "runtime": result,
                    "output_sha256": {p.relative_to(output).as_posix(): _hash(p) for p in sorted(output.rglob("*"))
                                      if p.is_file() and p.name not in {"metadata.json", "failure.json"}}}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
        _json({"status": "complete", "output": output, "metrics": result.get("metrics", result)})
    except Exception as error:
        if output.is_dir():
            (output / "failure.json").write_text(json.dumps({"status": "failed", "error": str(error), "code": code}, indent=2) + "\n", encoding="utf-8")
        raise
    return 0


def parser():
    result = argparse.ArgumentParser(description="Flyper Wolf: frozen visual-circuit tracking")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "doctor"):
        command = sub.add_parser(name, help="Generate an episode" if name == "run" else "Check dependencies, configuration and models")
        command.add_argument("--config", default="configs/episode.yaml")
        command.add_argument("--model-bundle")
        command.add_argument("--device")
        if name == "run":
            command.add_argument("--output", required=True)
            command.add_argument("--duration", type=float)
            command.add_argument("--target-seed", type=int)
            command.add_argument("--camera-seed", type=int)
            command.add_argument("--target-speed", type=float)
            command.add_argument("--no-video", action="store_true")
            command.add_argument("--record-neural", action="store_true",
                                 help="Archive every actual circuit unit and receptor observation for offline figures and GIFs")
            command.add_argument("--dry-run", action="store_true")
    command = sub.add_parser("download", help="Download a pinned, checksum-verified Hugging Face model bundle")
    command.add_argument("--manifest", required=True, help="Download descriptor JSON with repo_id, revision and manifest_sha256")
    command.add_argument("--output", required=True)
    command = sub.add_parser("export", help="Make a portable local bundle from existing model artifacts")
    command.add_argument("--estimator-manifest", required=True)
    command.add_argument("--flyvis-dir", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--release-id", default="local-inference")
    command.add_argument("--config", default="configs/episode.yaml", help="Inference defaults to include with the exported models")
    command = sub.add_parser("neural-video", help="Render recorded neural activity to MP4 offline")
    command.add_argument("--run", required=True, help="Episode directory generated with --record-neural")
    command.add_argument("--output", required=True, help="New output .mp4 file")
    command.add_argument("--fps", type=int, default=20, help="Display frame rate; archival activity is unchanged")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        if args.command == "doctor":
            return doctor(args)
        if args.command == "neural-video":
            from .neural_video import export_video
            report = export_video(args.run, args.output, fps=args.fps)
            _json({"status": "complete", "output": str(Path(args.output).resolve()), "video": report})
            return 0
        from .artifacts import download_bundle, export_local_bundle
        if args.command == "download":
            bundle = download_bundle(args.manifest, args.output)
        else:
            inference_config = load_config(args.config)
            inference_config["model_bundle"] = "."
            bundle = export_local_bundle(args.estimator_manifest, args.flyvis_dir, args.output,
                                         release_id=args.release_id,
                                         compatible_code_revision=code_identity()["revision"],
                                         inference_config=inference_config)
        _json({"status": "verified", "release_id": bundle.release_id, "bundle": bundle.root})
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"flyper-wolf: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
