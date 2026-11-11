"""Frozen released MLP inference; accepts ordered circuit activity only."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import yaml

TARGET_ENCODING = "camera-local-unit-vector-right-forward-up-v1"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def direction_bearings(vectors, fallback):
    """Convert predicted right/forward/up vectors into yaw and pitch degrees."""
    values, fallback = np.asarray(vectors, dtype=np.float64), np.asarray(fallback, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or fallback.shape != (3,) or not np.isfinite(fallback).all()
            or not np.isclose(np.linalg.norm(fallback), 1)):
        raise ValueError("Expected finite [N,3] predictions and unit training fallback")
    lengths = np.linalg.norm(values, axis=1)
    if not np.isfinite(lengths).all():
        raise FloatingPointError("Prediction vector norms are nonfinite")
    zeros = lengths < 1e-10
    unit = values / np.maximum(lengths[:, None], 1e-10)
    unit[zeros] = fallback
    x, y, z = unit.T
    return np.degrees(np.column_stack((np.arctan2(x, y), np.arctan2(z, np.hypot(x, y))))), int(zeros.sum())


def _network(feature_count, recipe):
    import torch

    activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU,
                  "gelu": torch.nn.GELU, "silu": torch.nn.SiLU}[recipe["activation"]]
    layers = []
    widths = [feature_count, *recipe["hidden_sizes"]]
    for before, after in zip(widths, widths[1:]):
        layers.extend([torch.nn.Linear(before, after), activation()])
        if recipe["dropout"]:
            layers.append(torch.nn.Dropout(recipe["dropout"]))
    layers.append(torch.nn.Linear(widths[-1], 3))
    return torch.nn.Sequential(*layers)


@dataclass
class FrozenEstimator:
    model: object
    mean: np.ndarray
    scale: np.ndarray
    fallback: np.ndarray
    device: str
    batch_size: int
    output_order: dict
    identity: dict

    def predict(self, activity):
        import torch

        features = np.asarray(activity)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError("Prediction features must match fitted circuit output order")
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                # Preserve source float64 standardization before the float32 handoff.
                x = (features[start:start+self.batch_size].astype(np.float64) - self.mean) / self.scale
                predictions.append(self.model(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty((0, 3), dtype=np.float32)

    def validate_output_order(self, fly):
        actual = {"node_indices": fly.output_indices, "cell_types": fly.cell_types[fly.output_indices],
                  "axial_uv": fly.cell_uv[fly.output_indices]}
        for name in actual:
            if not np.array_equal(self.output_order[name], actual[name]):
                raise ValueError(f"Frozen circuit output order differs: {name}")

    def parameter_hash(self):
        digest = hashlib.sha256()
        arrays = {name: getattr(self, name) for name in ("mean", "scale", "fallback")}
        arrays.update({f"model.{name}": value.detach().cpu().numpy()
                       for name, value in self.model.state_dict().items()})
        for name, value in sorted(arrays.items()):
            array = np.asarray(value)
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def load_frozen_estimator(manifest_path, *, device="cpu", batch_size=128):
    """Verify released artifact hashes, architecture, scaling and semantic order."""
    import torch

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    path = Path(manifest_path).resolve()
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "checkpoint", "output_order", "identity"}
    if (not isinstance(manifest, dict) or not required <= set(manifest)
            or set(manifest) - required - {"provenance"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1):
        raise ValueError("Unsupported frozen estimator manifest")
    identity = manifest["identity"]
    fields = {"kind", "target_encoding", "recipe", "input_features", "history_lags",
              "eligibility_lags", "seed", "selected_epoch"}
    if (not isinstance(identity, dict) or set(identity) != fields
            or identity["kind"] != "direction-search-mlp" or identity["target_encoding"] != TARGET_ENCODING
            or type(identity["input_features"]) is not int or identity["input_features"] <= 0
            or type(identity["seed"]) is not int or not 0 <= identity["seed"] < 2**32
            or type(identity["selected_epoch"]) is not int or identity["selected_epoch"] <= 0
            or identity["history_lags"] != [0] or identity["eligibility_lags"] != [0, 10, 20]
            or not isinstance(identity["recipe"], dict) or identity["recipe"].get("feature_mode") != "current"):
        raise ValueError("Unsupported frozen estimator identity")
    artifacts = {}
    for name in ("checkpoint", "output_order"):
        spec = manifest[name]
        if (not isinstance(spec, dict) or set(spec) != {"path", "sha256"}
                or not isinstance(spec["path"], str) or not spec["path"]
                or not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in spec["sha256"])):
            raise ValueError(f"Invalid {name} artifact specification")
        artifact = (path.parent / spec["path"]).resolve()
        if not artifact.is_relative_to(path.parent):
            raise ValueError("Estimator artifacts must remain within the manifest directory")
        if file_hash(artifact) != spec["sha256"]:
            raise ValueError(f"Frozen estimator {name} SHA256 mismatch")
        artifacts[name] = artifact
    with np.load(artifacts["output_order"], allow_pickle=False) as order:
        if set(order.files) != {"node_indices", "cell_types", "axial_uv"}:
            raise ValueError("Invalid output-order fields")
        output_order = {name: order[name].copy() for name in order.files}
    nodes, types, uv = (output_order[name] for name in ("node_indices", "cell_types", "axial_uv"))
    count = identity["input_features"]
    if (nodes.shape != (count,) or nodes.dtype.kind not in "iu" or np.any(nodes < 0)
            or len(np.unique(nodes)) != count or types.shape != nodes.shape or types.dtype.kind not in "US"
            or np.any(types == "") or uv.shape != (count, 2) or uv.dtype.kind not in "iu"):
        raise ValueError("Invalid output-order structure or feature count")
    saved = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
        raise ValueError("Unsupported frozen estimator checkpoint schema")
    for field in fields:
        if saved.get(field) != identity[field]:
            raise ValueError(f"Frozen estimator checkpoint {field} mismatch")
    if saved.get("output_order_sha256") != manifest["output_order"]["sha256"]:
        raise ValueError("Frozen estimator checkpoint output order mismatch")
    recipe = identity["recipe"]
    if (recipe.get("activation") not in ("tanh", "relu", "gelu", "silu")
            or recipe.get("loss") not in ("raw_vector_mse", "normalized_cosine")
            or not isinstance(recipe.get("hidden_sizes"), list) or not 1 <= len(recipe["hidden_sizes"]) <= 4
            or any(type(width) is not int or not 0 < width <= 512 for width in recipe["hidden_sizes"])
            or not isinstance(recipe.get("dropout"), (int, float)) or not 0 <= recipe["dropout"] <= .5):
        raise ValueError("Invalid search MLP architecture identity")
    mean, scale = saved["mean"].numpy().copy(), saved["scale"].numpy().copy()
    fallback = np.asarray(saved["zero_norm_fallback"].numpy(), dtype=np.float64)
    if (mean.shape != (count,) or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0) or fallback.shape != (3,)
            or not np.isfinite(fallback).all() or not np.isclose(np.linalg.norm(fallback), 1)):
        raise ValueError("Invalid checkpoint scaling or zero-vector fallback")
    model = _network(count, recipe)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite frozen estimator weights")
    model.to(device).eval().requires_grad_(False)
    for array in (mean, scale, fallback, *output_order.values()):
        array.setflags(write=False)
    return FrozenEstimator(model, mean, scale, fallback, str(device), batch_size, output_order,
                           {**identity, "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                            "output_order_sha256": manifest["output_order"]["sha256"]})
