"""Frozen pretrained FlyVis circuit with explicit episode-local neural state."""

import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flyper_wolf.compat import flyvis_initialization_device, windows_hdf5_writer
from flyper_wolf.preprocessing import VisualInput, as_tensor


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FrozenFly:
    """Accept receptor values only; return [batch, cells] activity after 10 ms.

    Reset establishes one second of neutral-gray adaptation. Each step holds
    one observation constant over the integration substeps and carries the
    complete FlyVis state into the next call. No image or world-state access.
    """

    def __init__(self, model_dir, *, device="cpu", observation_dt=0.01,
                 integration_dt=0.005, warmup_seconds=1.0):
        if version("flyvis") != "1.2.0":
            raise RuntimeError("This adapter is verified against FlyVis 1.2.0")
        for name, value in (("observation_dt", observation_dt),
                            ("integration_dt", integration_dt),
                            ("warmup_seconds", warmup_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.substeps = round(observation_dt / integration_dt)
        if (self.substeps < 1 or integration_dt > 0.02
                or not math.isclose(self.substeps * integration_dt, observation_dt)
                or not math.isclose(round(warmup_seconds / integration_dt) * integration_dt,
                                    warmup_seconds)):
            raise ValueError("Integration dt must divide observation/warmup intervals and be <= 0.02")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch installation has no available CUDA device")
        self.observation_dt = observation_dt
        self.integration_dt = integration_dt
        self.warmup_seconds = warmup_seconds
        model_dir = Path(model_dir).resolve()
        checkpoint = model_dir / "best_chkpt"
        meta_path = model_dir / "_meta.yaml"
        if not checkpoint.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Expected best_chkpt and _meta.yaml in {model_dir}; see README download command")
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        self.network_config = meta["config"]["network"]
        # The environment variable must precede FlyVis import (its connectome
        # cache root is captured by a class decorator at import time).
        os.environ.setdefault("FLYVIS_ROOT_DIR", str(Path.home() / ".cache" / "flyper-wolf" / "flyvis"))
        previous_device = torch.get_default_device()
        try:
            import flyvis

            # FlyVis allocates some indices outside registered module buffers.
            # Construct on the final device rather than calling .to() afterward.
            with (torch.device(self.device), windows_hdf5_writer() as patched,
                  flyvis_initialization_device(self.device)):
                self._network = flyvis.Network(**self.network_config)
                state = torch.load(checkpoint, map_location=self.device, weights_only=True)
                self._network.load_state_dict(state["network"], strict=True)
                self._network.requires_grad_(False)
                self._network.eval()
        finally:
            if torch.get_default_device() != previous_device:
                torch.set_default_device(previous_device)
        nodes = self._network.connectome.nodes
        self.cell_types = nodes.type[:].astype(str)
        self.cell_uv = np.column_stack((nodes.u[:], nodes.v[:]))
        # Use the model's declared readout types, never infer outputs from names.
        self.output_cell_types = self._network.connectome.output_cell_types[:].astype(str)
        self.output_indices = np.flatnonzero(np.isin(self.cell_types, self.output_cell_types))
        if not len(self.output_indices):
            raise ValueError("The loaded connectome has no labeled output cells")
        self._output_indices_tensor = torch.tensor(self.output_indices, device=self.device)
        reference_uv = VisualInput().axial_coordinates
        for name in self._network.connectome.input_cell_types[:].astype(str):
            np.testing.assert_array_equal(self.cell_uv[self.cell_types == name], reference_uv)
        self.identity = {
            "model_dir": str(model_dir), "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "model_config_sha256": file_hash(meta_path),
            "connectome_sha256": file_hash(flyvis.connectome_file),
            "flyvis_version": version("flyvis"), "torch_version": version("torch"),
            "torch_build": torch.__version__, "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_name": (torch.cuda.get_device_name(self.device)
                            if self.device.type == "cuda" else "CPU"),
            "datamate_version": version("datamate"), "windows_hdf5_workaround": patched,
            "cells": self._network.n_nodes, "edges": self._network.n_edges,
            "cell_types": len(np.unique(self.cell_types)),
            "parameter_values": sum(p.numel() for p in self._network.parameters()),
            "trainable_parameter_values": 0,
            "pretraining_task": meta["config"]["task"]["dataset"],
        }
        self._state = None
        self._batch_size = None
        self.steps = 0

    def outputs(self, activity):
        """Select labeled outputs in original node order, preserving spatial identity."""
        return activity.index_select(-1, self._output_indices_tensor)

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, parameter in self._network.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def reset(self, batch_size=1):
        """Discard episode history; re-adapt to fixed 0.5 gray, never train."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        with torch.device(self.device):
            self._state = self._network.steady_state(
                self.warmup_seconds, self.integration_dt, batch_size, value=0.5)
        self._batch_size, self.steps = batch_size, 0
        return self._state.nodes.activity.clone()

    @torch.no_grad()
    def step(self, receptors):
        """Advance exactly one observation interval without changing weights."""
        if self._state is None:
            raise RuntimeError("Call reset() at the beginning of an episode")
        if isinstance(receptors, np.ndarray):
            receptors = as_tensor(receptors, device=self.device)
        if (not isinstance(receptors, torch.Tensor)
                or receptors.shape != (self._batch_size, 1, 1, 721)
                or receptors.dtype != torch.float32
                or not torch.isfinite(receptors).all()
                or torch.any((receptors < 0) | (receptors > 1))):
            raise ValueError("Expected float32 [reset batch size, 1, 1, 721] in [0, 1]")
        with torch.device(self.device):
            stimulus = self._network.stimulus
            stimulus.zero(self._batch_size, self.substeps)
            stimulus.add_input(receptors.detach().to(self.device).expand(-1, self.substeps, -1, -1))
            states = self._network(stimulus(), self.integration_dt,
                                   state=self._state, as_states=True)
            self._state = states[-1]
            activity = self._state.nodes.activity
            if not torch.isfinite(activity).all():
                raise FloatingPointError("Nonfinite FlyVis activity; check stimulus and integration dt")
            self.steps += 1
            # Cloning prevents callers from modifying the private episode state.
            return activity.clone()
