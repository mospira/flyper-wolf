"""Integrity and publication checks; fixture bytes are never used as models."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from flyper_wolf.artifacts import (
    BundleError, download_bundle, export_local_bundle, sha256_file, verify_bundle,
)


@pytest.fixture
def source(tmp_path):
    estimator = tmp_path / "source-estimator"
    flyvis = tmp_path / "source-flyvis"
    estimator.mkdir()
    flyvis.mkdir()
    for path in (estimator / "best.pt", estimator / "output-order.npz", flyvis / "best_chkpt"):
        path.write_bytes(b"integrity-test-only:" + path.name.encode())
    (flyvis / "_meta.yaml").write_text("config: {network: {}, task: {dataset: test}}\n")
    identity = {"kind": "direction-search-mlp", "selected_epoch": 342}
    manifest = {"schema_version": 1, "identity": identity,
                "provenance": {"private_path": "C:/private/training"}}
    for key, name in (("checkpoint", "best.pt"), ("output_order", "output-order.npz")):
        manifest[key] = {"path": name, "sha256": sha256_file(estimator / name)}
    (estimator / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return estimator / "manifest.yaml", flyvis


def export(source, tmp_path):
    return export_local_bundle(*source, tmp_path / "bundle", release_id="test-unpublished")


def write_manifest(bundle, manifest):
    bundle.manifest_path.write_text(json.dumps(manifest))


def test_export_preserves_identity_and_is_portable(source, tmp_path):
    bundle = export(source, tmp_path)
    portable = yaml.safe_load(bundle.estimator_manifest.read_text())
    assert portable["identity"] == {"kind": "direction-search-mlp", "selected_epoch": 342}
    assert "provenance" not in portable
    assert portable["checkpoint"]["path"] == "best.pt"
    assert sha256_file(bundle.estimator_checkpoint) == portable["checkpoint"]["sha256"]
    assert "C:/private" not in bundle.manifest_path.read_text()
    moved = tmp_path / "moved"
    bundle.root.rename(moved)
    assert verify_bundle(moved).estimator_checkpoint == moved / "estimator/best.pt"


def test_export_pins_validated_inference_config_and_preserves_input(source, tmp_path):
    config = {"schema_version": 1, "model_bundle": "C:/private/model",
              "artifacts": {"estimator_manifest": "C:/private/manifest"},
              "controller": {"gain_yaw": 22.0, "gain_pitch": 23.0},
              "runtime": {"video": False}, "scene": {"target_radius": 0.29}}
    bundle = export_local_bundle(*source, tmp_path / "bundle", inference_config=config)
    relative = bundle.manifest["inference_config"]
    path = bundle.root / relative
    pinned = yaml.safe_load(path.read_text())
    assert bundle.hashes[relative] == sha256_file(path)
    assert pinned["model_bundle"] == "."
    assert "artifacts" not in pinned
    assert pinned["controller"] == config["controller"]
    assert pinned["scene"]["target_radius"] == 0.29
    assert pinned["runtime"]["integration_dt"] == 0.005
    assert pinned["runtime"]["video"] is False
    assert "C:/private" not in path.read_text()
    assert config["model_bundle"] == "C:/private/model"
    assert config["artifacts"]["estimator_manifest"] == "C:/private/manifest"
    path.write_text(path.read_text().replace("gain_yaw: 22.0", "gain_yaw: 25.0"))
    with pytest.raises(BundleError, match="SHA256 mismatch"):
        verify_bundle(bundle.root)


def test_invalid_inference_config_is_not_published(source, tmp_path):
    destination = tmp_path / "bundle"
    with pytest.raises(BundleError, match="Invalid bundled inference configuration"):
        export_local_bundle(*source, destination,
                            inference_config={"schema_version": 1, "controller": {"gain_yaw": -1}})
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_corruption_and_incomplete_existing_destination_rejected(source, tmp_path):
    bundle = export(source, tmp_path)
    bundle.estimator_checkpoint.write_bytes(b"corrupt")
    with pytest.raises(BundleError, match="SHA256 mismatch"):
        verify_bundle(bundle.root)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_local_bundle(*source, bundle.root)
    bundle.estimator_checkpoint.unlink()
    with pytest.raises(BundleError, match="Missing"):
        verify_bundle(bundle.root)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/absolute", "a\\b", "a/../b", "a//b"])
def test_unsafe_manifest_paths_rejected(source, tmp_path, path):
    bundle = export(source, tmp_path)
    manifest = bundle.manifest
    manifest["artifacts"][path] = "0" * 64
    write_manifest(bundle, manifest)
    with pytest.raises(BundleError, match="Unsafe bundle path"):
        verify_bundle(bundle.root)


def test_estimator_reference_must_be_in_hash_index(source, tmp_path):
    bundle = export(source, tmp_path)
    estimator = yaml.safe_load(bundle.estimator_manifest.read_text())
    estimator["checkpoint"]["path"] = "../other.pt"
    bundle.estimator_manifest.write_text(yaml.safe_dump(estimator))
    manifest = bundle.manifest
    manifest["artifacts"]["estimator/manifest.yaml"] = sha256_file(bundle.estimator_manifest)
    write_manifest(bundle, manifest)
    with pytest.raises(BundleError, match="Unsafe bundle path"):
        verify_bundle(bundle.root)


def test_symlinked_artifact_rejected(source, tmp_path):
    bundle = export(source, tmp_path)
    original = bundle.estimator_checkpoint
    contents = tmp_path / "outside.pt"
    original.rename(contents)
    try:
        original.symlink_to(contents)
    except OSError:
        pytest.skip("Creating symlinks requires platform permissions")
    with pytest.raises(BundleError, match="symlinks"):
        verify_bundle(bundle.root)


def test_bad_source_hash_does_not_publish_partial_bundle(source, tmp_path):
    (source[0].parent / "best.pt").write_bytes(b"changed")
    destination = tmp_path / "bundle"
    with pytest.raises(BundleError, match="Source estimator checkpoint SHA256 mismatch"):
        export_local_bundle(*source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_download_is_pinned_verified_and_atomic(source, tmp_path, monkeypatch):
    upstream = export(source, tmp_path)
    config = {"schema_version": 1, "repo_id": "owner/test", "revision": "a" * 40,
              "manifest_sha256": sha256_file(upstream.manifest_path)}
    config_path = tmp_path / "release.json"
    config_path.write_text(json.dumps(config))
    calls = []

    def fake_download(*, repo_id, revision, filename):
        calls.append((repo_id, revision, filename))
        return str(upstream.root / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    destination = tmp_path / "download"
    bundle = download_bundle(config_path, destination)
    assert bundle.release_id == upstream.release_id
    assert len(calls) == 1 + len(upstream.hashes)
    assert all(repo == "owner/test" and revision == "a" * 40 for repo, revision, _ in calls)
    assert download_bundle(config_path, destination).hashes == bundle.hashes
    assert len(calls) == 1 + len(upstream.hashes)
    upstream.estimator_checkpoint.write_bytes(b"upstream corruption")
    failed = tmp_path / "failed"
    with pytest.raises(BundleError, match="Downloaded artifact SHA256 mismatch"):
        download_bundle(config_path, failed)
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_placeholder_download_fails_before_network(tmp_path):
    config = tmp_path / "placeholder.json"
    config.write_text(json.dumps({"schema_version": 1, "repo_id": None, "revision": None,
                                  "manifest_sha256": None}))
    with pytest.raises(BundleError, match="not configured"):
        download_bundle(config, tmp_path / "download")


def test_incomplete_existing_download_is_never_replaced(source, tmp_path):
    upstream = export(source, tmp_path)
    config = tmp_path / "release.json"
    config.write_text(json.dumps({"schema_version": 1, "repo_id": "owner/test", "revision": "a" * 40,
                                  "manifest_sha256": sha256_file(upstream.manifest_path)}))
    incomplete = tmp_path / "download"
    incomplete.mkdir()
    (incomplete / "keep.txt").write_text("existing user data")
    with pytest.raises(BundleError, match="Missing bundle artifact"):
        download_bundle(config, incomplete)
    assert (incomplete / "keep.txt").read_text() == "existing user data"
