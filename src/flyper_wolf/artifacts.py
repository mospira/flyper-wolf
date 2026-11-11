"""Portable, hash-verified inference bundles; no training repository dependency."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any

import yaml


class BundleError(ValueError):
    """A bundle is incomplete, unsafe, or differs from its recorded identity."""


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest_path: Path
    release_id: str
    estimator_manifest: Path
    estimator_checkpoint: Path
    output_order: Path
    flyvis_dir: Path
    hashes: dict[str, str]
    manifest: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise BundleError("Expected a configured lowercase SHA256 checksum")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise BundleError(f"Unsafe bundle path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise BundleError(f"Unsafe bundle path: {value!r}")
    return value


def _linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _file(root: Path, relative: str) -> Path:
    relative = _relative(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if _linked(path):
            raise BundleError(f"Bundle paths must not contain symlinks: {relative}")
    if not path.is_file():
        raise BundleError(f"Missing bundle artifact: {relative}")
    return path


def _manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"Cannot read bundle manifest: {path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise BundleError("Unsupported bundle schema_version; expected 1")
    if not isinstance(data.get("release_id"), str) or not data["release_id"].strip():
        raise BundleError("Bundle requires a nonempty release_id")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise BundleError("Bundle requires an artifact checksum mapping")
    for name, digest in artifacts.items():
        _relative(name)
        _hash(digest)
        if name.casefold() == "manifest.json":
            raise BundleError("Bundle manifest cannot hash itself")
    if len({name.casefold() for name in artifacts}) != len(artifacts):
        raise BundleError("Artifact names collide on case-insensitive filesystems")
    _relative(data.get("estimator_manifest"))
    _relative(data.get("flyvis_dir"))
    return data


def verify_bundle(bundle_dir: str | Path) -> VerifiedBundle:
    """Verify every artifact before returning any paths usable by inference.

    Hashes establish identity and integrity relative to this manifest. Downloaded
    releases additionally pin the manifest checksum in their download config.
    """
    root = Path(bundle_dir).absolute()
    if _linked(root) or not root.is_dir():
        raise BundleError(f"Expected a real bundle directory: {root}")
    manifest_path = _file(root, "manifest.json")
    manifest = _manifest(manifest_path)
    hashes = manifest["artifacts"]
    for name, digest in hashes.items():
        if sha256_file(_file(root, name)) != digest:
            raise BundleError(f"SHA256 mismatch for bundle artifact: {name}")
    estimator_relative = manifest["estimator_manifest"]
    flyvis_relative = manifest["flyvis_dir"]
    required = [estimator_relative, f"{flyvis_relative}/best_chkpt", f"{flyvis_relative}/_meta.yaml"]
    if any(name not in hashes for name in required):
        raise BundleError("Manifest must hash the estimator manifest and both FlyVis artifacts")
    if "inference_config" in manifest:
        relative = _relative(manifest["inference_config"])
        if relative not in hashes:
            raise BundleError("Manifest must hash its inference configuration")
        _validate_inference_config(_file(root, relative))
    estimator_path = _file(root, estimator_relative)
    try:
        estimator = yaml.safe_load(estimator_path.read_text(encoding="utf-8"))
        if estimator["schema_version"] != 1 or not isinstance(estimator["identity"], dict):
            raise BundleError("Unsupported estimator manifest")
        paths = []
        for key in ("checkpoint", "output_order"):
            relative = str(PurePosixPath(estimator_relative).parent / _relative(estimator[key]["path"]))
            if relative not in hashes or hashes[relative] != _hash(estimator[key]["sha256"]):
                raise BundleError(f"Estimator {key} differs from the bundle artifact index")
            paths.append(_file(root, relative))
    except (TypeError, KeyError, yaml.YAMLError) as exc:
        raise BundleError("Malformed estimator manifest") from exc
    return VerifiedBundle(root, manifest_path, manifest["release_id"], estimator_path,
                          paths[0], paths[1], root / flyvis_relative, dict(hashes), manifest)


def _validate_inference_config(path: Path) -> dict[str, Any]:
    from .configuration import load_config

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("model_bundle") != ".":
            raise ValueError("model_bundle must be '.' relative to the bundle directory")
        return load_config(path)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise BundleError(f"Invalid bundled inference configuration: {exc}") from exc


def _staging(destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing model bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))


def _publish(stage: Path, destination: Path) -> VerifiedBundle:
    verify_bundle(stage)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination appeared during preparation: {destination}")
    os.rename(stage, destination)
    return verify_bundle(destination)


def export_local_bundle(estimator_manifest: str | Path, flyvis_dir: str | Path,
                        destination: str | Path, *, release_id: str = "local-inference",
                        compatible_code_revision: str | None = None,
                        inference_config: Mapping[str, Any] | None = None) -> VerifiedBundle:
    """Copy an explicitly selected local model into an unpublished local inference bundle.

    Only estimator inference identity is retained from its YAML; development
    paths and private provenance are removed. Weights and output order retain
    their original byte hashes. This does not choose or publish a release.
    """
    source = Path(estimator_manifest).resolve()
    try:
        estimator = yaml.safe_load(source.read_text(encoding="utf-8"))
        if estimator["schema_version"] != 1 or not isinstance(estimator["identity"], dict):
            raise BundleError("Unsupported source estimator manifest")
        portable = {"schema_version": 1, "identity": estimator["identity"]}
        source_paths = {}
        for key, filename in (("checkpoint", "best.pt"), ("output_order", "output-order.npz")):
            section = estimator[key]
            path = (source.parent / section["path"]).resolve()
            if sha256_file(path) != _hash(section["sha256"]):
                raise BundleError(f"Source estimator {key} SHA256 mismatch")
            source_paths[key] = path
            portable[key] = {"path": filename, "sha256": section["sha256"]}
    except (TypeError, KeyError, yaml.YAMLError) as exc:
        raise BundleError("Malformed source estimator manifest") from exc
    flyvis = Path(flyvis_dir)
    for name in ("best_chkpt", "_meta.yaml"):
        _file(flyvis, name)
    destination = Path(destination).absolute()
    stage = _staging(destination)
    try:
        (stage / "estimator").mkdir()
        (stage / "flyvis").mkdir()
        (stage / "estimator/manifest.yaml").write_text(yaml.safe_dump(portable, sort_keys=False), encoding="utf-8")
        for key in source_paths:
            shutil.copyfile(source_paths[key], stage / "estimator" / portable[key]["path"])
        for name in ("best_chkpt", "_meta.yaml"):
            shutil.copyfile(flyvis / name, stage / "flyvis" / name)
        if inference_config is not None:
            if not isinstance(inference_config, Mapping):
                raise BundleError("inference_config must be a settings mapping")
            config = deepcopy(dict(inference_config))
            config.pop("artifacts", None)
            config["model_bundle"] = "."
            config_path = stage / "inference-config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            # Record resolved defaults now so later code-default changes cannot
            # silently change this bundle's simulation or timing settings.
            config = _validate_inference_config(config_path)
            config["model_bundle"] = "."
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        artifacts = {path.relative_to(stage).as_posix(): sha256_file(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"schema_version": 1, "release_id": release_id,
                    "release_status": "unpublished-local",
                    "compatible_code_revision": compatible_code_revision,
                    "estimator_manifest": "estimator/manifest.yaml", "flyvis_dir": "flyvis",
                    "source_estimator_manifest_sha256": sha256_file(source),
                    "architecture": {"visual_circuit": "FlyVis-1.2.0-frozen", "estimator": estimator["identity"]},
                    "artifacts": artifacts}
        if inference_config is not None:
            manifest["inference_config"] = "inference-config.yaml"
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return _publish(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def download_bundle(config_path: str | Path, destination: str | Path) -> VerifiedBundle:
    """Fetch a release from a pinned HF commit, verify, then publish atomically."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        repo_id, revision = config["repo_id"], config["revision"]
        if (config.get("schema_version") != 1 or not isinstance(repo_id, str)
                or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id)
                or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise BundleError("Configure a real Hugging Face repo_id and immutable 40-character commit revision")
        expected_hash = _hash(config["manifest_sha256"])
        manifest_file = _relative(config.get("manifest_file", "manifest.json"))
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(f"Release download is not configured: {exc}") from exc
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        verified = verify_bundle(destination)
        if sha256_file(verified.manifest_path) != expected_hash:
            raise BundleError("Existing bundle belongs to a different pinned release")
        return verified
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub to download released model bundles") from exc
    stage = _staging(destination)
    try:
        downloaded = Path(hf_hub_download(repo_id=repo_id, revision=revision, filename=manifest_file))
        shutil.copyfile(downloaded, stage / "manifest.json")
        if sha256_file(stage / "manifest.json") != expected_hash:
            raise BundleError("Downloaded release manifest SHA256 mismatch")
        manifest = _manifest(stage / "manifest.json")
        for relative, digest in manifest["artifacts"].items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(hf_hub_download(repo_id=repo_id, revision=revision, filename=relative))
            shutil.copyfile(downloaded, target)
            if sha256_file(target) != digest:
                raise BundleError(f"Downloaded artifact SHA256 mismatch: {relative}")
        return _publish(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
