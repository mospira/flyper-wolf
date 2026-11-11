"""Passive, disk-backed recording of the circuit already running an episode.

Nothing in this module advances neural state or supplies estimator/controller
inputs. Saved float32 values retain the exact inference precision; any display
subsampling belongs to offline figure generation.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _copy_float32(value, shape):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != np.float32 or array.size != int(np.prod(shape)):
        raise ValueError(f"Expected float32 neural values with shape {shape}")
    array = array.reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("Cannot record nonfinite neural values")
    return array.copy()


class NeuralRecorder:
    """Own archival copies only, with an explicit valid prefix after failure."""

    def __init__(self, output, fly, baseline, *, ticks, observation_dt,
                 estimator_identity, episode, frozen_hash, estimator_hash):
        self.directory = Path(output) / "neural"
        self.directory.mkdir(exist_ok=False)
        self.frames = 0
        self.activity = self.inputs = None
        self.closed = False
        self.manifest = {
            "schema_version": 1, "status": "incomplete", "complete": False,
            "frames_recorded": 0, "expected_frames": ticks, "source_frames": ticks,
            "units": len(fly.cell_types), "observation_dt": observation_dt,
            "input_time_offset_seconds": 0, "activity_time_offset_seconds": observation_dt,
            "baseline_kind": "post-reset neutral-gray adaptation",
            "baseline_time_seconds": 0, "recording_stride": 1,
            "state_timestamp": "end of the input observation interval",
            "input_order": "axial u ascending, then v ascending; radius 15, 721 receptors",
            "model": fly.identity, "estimator": estimator_identity, "episode": episode,
            "frozen_parameters_sha256": frozen_hash,
            "estimator_parameters_sha256": estimator_hash,
            "frozen_parameters_unchanged": None, "files": {},
        }
        self._write_manifest()
        try:
            count = len(fly.cell_types)
            np.save(self.directory / "baseline.npy", _copy_float32(baseline, (count,)), allow_pickle=False)
            types, uv = np.asarray(fly.cell_types), np.asarray(fly.cell_uv)
            if types.shape != (count,) or uv.shape != (count, 2) or not np.issubdtype(uv.dtype, np.integer):
                raise ValueError("Circuit node metadata does not match the recorded activity")
            np.savez(self.directory / "cells.npz", cell_types=types.astype(str), axial_uv=uv)
            # Read the same instantiated graph, without rebuilding or resetting it.
            graph = fly._network.connectome
            total_edges = len(graph.edges.source_index)
            selected = np.sort(np.random.default_rng(74).choice(
                total_edges, size=min(12000, total_edges), replace=False))
            # Read each HDF5 column once; thousands of fancy HDF5 selections
            # are substantially slower on the upstream graph storage.
            edges = np.column_stack((graph.edges.source_index[:][selected],
                                     graph.edges.target_index[:][selected])).astype(np.int64)
            if edges.shape != (len(selected), 2) or (edges.size and (edges.min() < 0 or edges.max() >= count)):
                raise ValueError("Circuit edges do not refer to recorded node indices")
            np.save(self.directory / "edges.npy", edges, allow_pickle=False)
            self.manifest.update(total_edges=total_edges, edges_sample={
                "seed": 74, "count": len(selected),
                "selection": "uniform without replacement; sorted original edge indices",
                "selected_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
            })
            self.activity = np.lib.format.open_memmap(self.directory / "activity.npy", mode="w+",
                                                     dtype=np.float32, shape=(ticks, count))
            self.inputs = np.lib.format.open_memmap(self.directory / "inputs.npy", mode="w+",
                                                   dtype=np.float32, shape=(ticks, 721))
            self._write_manifest()
        except Exception as error:
            try:
                self.close(complete=False, error=str(error))
            except Exception as close_error:
                error.add_note(f"Recording cleanup also failed: {close_error}")
            raise

    def _write_manifest(self):
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")

    def append(self, activity, receptors):
        if self.closed or self.frames >= self.manifest["expected_frames"]:
            raise RuntimeError("Neural recording is closed or already full")
        # Copy detached values before disk writes; the recorder owns no writable
        # alias of either private FlyVis state or causal inference input.
        values = _copy_float32(activity, (self.manifest["units"],))
        inputs = _copy_float32(receptors, (721,))
        self.activity[self.frames] = values
        self.inputs[self.frames] = inputs
        self.frames += 1
        if self.frames % 100 == 0:
            self.activity.flush()
            self.inputs.flush()
            self.manifest["frames_recorded"] = self.frames
            self._write_manifest()

    def close(self, *, complete, error=None):
        if self.closed:
            return
        for array in (self.activity, self.inputs):
            if array is not None:
                array.flush()
        complete = bool(complete and self.frames == self.manifest["expected_frames"])
        self.manifest.update(status="complete" if complete else "incomplete", complete=complete,
                             frames_recorded=self.frames, valid_prefix_frames=self.frames,
                             frozen_parameters_unchanged=True if complete else None)
        if error is not None:
            self.manifest["error"] = str(error)
        files = {}
        for path in sorted(self.directory.iterdir()):
            if path.suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                files[path.name] = {"sha256": _hash(path), "shape": list(array.shape), "dtype": str(array.dtype)}
            elif path.name == "cells.npz":
                with np.load(path, allow_pickle=False) as arrays:
                    files[path.name] = {"sha256": _hash(path), "arrays": {
                        key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)} for key in arrays.files}}
        self.manifest["files"] = files
        self._write_manifest()
        self.closed = True
        # Explicitly release file mappings, including on Windows.
        for array in (self.activity, self.inputs):
            if array is not None:
                array._mmap.close()
        self.activity = self.inputs = None


def _finite(array, name):
    # Bound validation scratch memory even for long recordings.
    if array.dtype.kind not in "iuf" or any(
        not np.isfinite(array[start:start + 128]).all()
        for start in range(0, len(array), 128)
    ):
        raise ValueError(f"{name} must contain finite numeric values")


def load_recording(run):
    """Return validated arrays and manifest from a completed run/neural capture.

    Keys: activity, inputs, baseline, cell_types, axial_uv, edges, manifest,
    neural_dir. Large .npy arrays are read-only memory maps. Hashes, completion,
    shapes, timing and values are checked before any downstream visualization.
    """
    neural = Path(run) / "neural"
    manifest = json.loads((neural / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("status") != "complete" or manifest.get("frozen_parameters_unchanged") is not True):
        raise ValueError("Neural recording is incomplete or has an unsupported schema")
    dt = manifest.get("observation_dt")
    if (isinstance(dt, bool) or not isinstance(dt, (int, float))
            or not math.isfinite(dt) or dt <= 0
            or manifest.get("input_time_offset_seconds") != 0
            or not math.isclose(manifest.get("activity_time_offset_seconds", -1), dt)):
        raise ValueError("Neural recording timing must pair source input t with activity t + dt")
    arrays = {}
    for filename in ("activity.npy", "inputs.npy", "baseline.npy", "cells.npz", "edges.npy"):
        metadata = manifest.get("files", {}).get(filename, {})
        path = neural / filename
        if not path.is_file() or metadata.get("sha256") != _hash(path):
            raise ValueError(f"Missing file or SHA-256 mismatch: {filename}")
        if filename.endswith(".npy"):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata.get("shape") or str(array.dtype) != metadata.get("dtype"):
                raise ValueError(f"Manifest shape/dtype mismatch: {filename}")
            arrays[filename[:-4]] = array
    with np.load(neural / "cells.npz", allow_pickle=False) as cells:
        types, uv = cells["cell_types"], cells["axial_uv"]
    activity, inputs, baseline, edges = (arrays[n] for n in ("activity", "inputs", "baseline", "edges"))
    if activity.ndim != 2 or min(activity.shape) < 1:
        raise ValueError("activity must have nonempty [time, units] shape")
    t, n = activity.shape
    if any(manifest.get(k) != t for k in ("source_frames", "frames_recorded", "expected_frames")):
        raise ValueError("Incomplete capture: frame counts disagree")
    if (manifest.get("units") != n or inputs.shape != (t, 721)
            or baseline.shape != (n,) or types.shape != (n,) or uv.shape != (n, 2)):
        raise ValueError("Neural arrays disagree on frame, unit or receptor dimensions")
    if (types.dtype.kind not in "US" or uv.dtype.kind not in "iu"
            or np.any(uv < -32768) or np.any(uv > 32767)):
        raise ValueError("Expected string cell types and signed-16-bit-compatible axial coordinates")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edges must contain integer source/target pairs")
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Graph edge endpoint is outside the recorded model units")
    for name in ("activity", "inputs", "baseline"):
        _finite(arrays[name], name)
    if inputs.min() < 0 or inputs.max() > 1:
        raise ValueError("Receptor inputs must lie in [0, 1]")
    total = manifest.get("total_edges", len(edges))
    if not isinstance(total, int) or total < len(edges):
        raise ValueError("Total graph edge count is smaller than its recorded sample")
    return {**arrays, "cell_types": types.astype(str), "axial_uv": uv,
            "manifest": manifest, "neural_dir": neural}
