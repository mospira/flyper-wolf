"""Focused frozen-checkpoint and causal input contracts; no model downloads."""

import hashlib
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from flyper_wolf.controller import ProportionalController
from flyper_wolf.estimator import TARGET_ENCODING, _network, direction_bearings, load_frozen_estimator
from flyper_wolf.preprocessing import VisualInput
from flyper_wolf.scene import SceneConfig
from flyper_wolf.scoring import aim_geometry, score_interval
from flyper_wolf.simulation import Simulation, WorldFrame


class InferenceCoreTests(unittest.TestCase):
    def test_checkpoint_roundtrip_is_exact_frozen_and_hash_checked(self):
        import torch

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {"feature_mode": "current", "hidden_sizes": [4], "activation": "relu",
                      "dropout": .1, "loss": "normalized_cosine"}
            identity = {"kind": "direction-search-mlp", "target_encoding": TARGET_ENCODING,
                        "recipe": recipe, "input_features": 3, "history_lags": [0],
                        "eligibility_lags": [0, 10, 20], "seed": 74, "selected_epoch": 342}
            model = _network(3, recipe).eval()
            mean = torch.tensor([1., 2., 3.], dtype=torch.float64)
            scale = torch.tensor([2., 3., 4.], dtype=torch.float64)
            np.savez(root/"output-order.npz", node_indices=np.array([0, 2, 4]),
                     cell_types=np.array(["T4a", "T4a", "T5a"]), axial_uv=np.array([[0, 0], [1, 0], [0, 1]]))
            order_hash = hashlib.sha256((root/"output-order.npz").read_bytes()).hexdigest()
            saved = {"schema_version": 1, **identity, "mean": mean, "scale": scale,
                     "zero_norm_fallback": torch.tensor([0., 1., 0.]), "output_order_sha256": order_hash,
                     "model_state_dict": model.state_dict()}
            torch.save(saved, root/"best.pt")
            manifest = {"schema_version": 1, "identity": identity,
                        "checkpoint": {"path": "best.pt", "sha256": hashlib.sha256((root/"best.pt").read_bytes()).hexdigest()},
                        "output_order": {"path": "output-order.npz", "sha256": order_hash}}
            (root/"manifest.yaml").write_text(yaml.safe_dump(manifest))
            predictor = load_frozen_estimator(root/"manifest.yaml")
            features = np.array([[4., 8., -3.]], dtype=np.float32)
            with torch.no_grad():
                expected = model(torch.tensor((features.astype(np.float64)-mean.numpy())/scale.numpy(), dtype=torch.float32)).numpy()
            np.testing.assert_array_equal(predictor.predict(features), expected)
            self.assertTrue(all(not value.requires_grad for value in predictor.model.parameters()))
            self.assertFalse(predictor.mean.flags.writeable)
            with self.assertRaisesRegex(ValueError, "circuit output order"):
                predictor.predict(np.zeros((1, 4)))
            class Fly:
                output_indices = np.array([0, 2, 4])
                cell_types = np.array(["T4a", "unused", "T4a", "unused", "T5a"])
                cell_uv = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])
            predictor.validate_output_order(Fly())
            Fly.output_indices = np.array([2, 0, 4])
            with self.assertRaisesRegex(ValueError, "output order differs"):
                predictor.validate_output_order(Fly())
            (root/"best.pt").write_bytes((root/"best.pt").read_bytes()+b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_frozen_estimator(root/"manifest.yaml")

    def test_geometry_and_controller_use_distinct_interfaces(self):
        scene = SceneConfig()
        forward = WorldFrame(0, (0., 2., 4.), 0., 0.)
        self.assertEqual(aim_geometry(forward, scene), (0., True))
        controller = ProportionalController()
        np.testing.assert_array_equal(controller.action([1., -.5]), [.4, -.2])
        np.testing.assert_array_equal(controller.action([90., -90.]), [1., -1.])
        with self.assertRaises((TypeError, ValueError)):
            controller.action(forward)
        sim = Simulation(scene, seed=7, trajectory_version="smooth-reactive-plane-v1")
        before = sim.snapshot()
        score = score_interval(sim, before, [1., -1.])
        self.assertEqual(sim.snapshot(), before)
        self.assertTrue(0 <= score["on_target_seconds"] <= .01)

    def test_preprocessing_and_fallback_are_explicit(self):
        adapter = VisualInput()
        receptors = adapter(np.full((384, 384, 3), 128, dtype=np.uint8))
        self.assertEqual(receptors.shape, (1, 1, 1, 721))
        np.testing.assert_allclose(receptors, np.float32(128/255), rtol=0, atol=1e-7)
        with self.assertRaises(ValueError):
            adapter(np.zeros((384, 384, 3), dtype=np.float32))
        bearings, zeros = direction_bearings(np.array([[0., 0., 0.], [1., 0., 0.]]), [0., 1., 0.])
        np.testing.assert_array_equal(bearings, [[0., 0.], [90., 0.]])
        self.assertEqual(zeros, 1)

    def test_episode_startup_advances_circuit_without_estimator_or_actions(self):
        """Spies check loop scheduling; they are never available as runner modes."""
        import torch
        from flyper_wolf import runtime

        events = []

        class Fly:
            identity = {"test_spy": True}
            def __init__(self, *args, **kwargs):
                self.steps = 0
            def parameter_hash(self):
                return "fixed-fly"
            def reset(self):
                events.append(("reset", self.steps))
                self.steps = 0
            def step(self, receptors):
                self.steps += 1
                self.assert_receptors(receptors)
                events.append(("step", self.steps))
                return torch.tensor([[float(self.steps), 0., 0.]])
            @staticmethod
            def assert_receptors(receptors):
                if receptors.shape != (1, 1, 1, 721):
                    raise AssertionError("Unexpected circuit input")
            def outputs(self, activity):
                return activity

        class Renderer:
            def __init__(self, scene):
                self.scene = scene
            def capture(self, frame):
                events.append(("capture", frame.tick))
                return np.zeros((384, 384, 3), dtype=np.uint8)
            def close(self):
                events.append(("close", None))

        class Predictor:
            identity = {"test_spy": True}
            fallback = np.array([0., 1., 0.])
            def parameter_hash(self):
                return "fixed-estimator"
            def validate_output_order(self, fly):
                pass
            def predict(self, activity):
                if activity.shape != (1, 3):
                    raise AssertionError("Estimator received more than ordered activity")
                events.append(("predict", int(activity[0, 0])))
                return np.array([[0., 1., 0.]])

        original_truth = runtime.camera_local_bearing
        def truth(frame, scene):
            events.append(("truth", frame.tick))
            return original_truth(frame, scene)

        config = {"episode": {"duration_seconds": .24, "target_seed": 7, "camera_seed": 8},
                  "runtime": {"video": False}, "artifacts": {"estimator_manifest": "unused",
                  "flyvis_model_dir": "unused"}}
        with TemporaryDirectory() as temporary, patch.dict("sys.modules", {
                "flyper_wolf.fly": SimpleNamespace(FrozenFly=Fly),
                "flyper_wolf.rendering": SimpleNamespace(SceneRenderer=Renderer)}), patch.object(
                runtime, "load_frozen_estimator", return_value=Predictor()), patch.object(
                runtime, "_configure_runtime", return_value={"device": "cpu"}), patch.object(
                runtime, "camera_local_bearing", side_effect=truth):
            result = runtime.run_episode(config, Path(temporary))
            with (Path(temporary)/"trace.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(result["metrics"]["circuit_steps"], 24)
        self.assertEqual(result["metrics"]["prediction_updates"], 4)
        self.assertEqual([value for name, value in events if name == "predict"], [21, 22, 23, 24])
        for tick, row in enumerate(rows):
            self.assertEqual(row["prediction_updated"], str(tick >= 20))
            if tick < 20:
                self.assertEqual(row["predicted_horizontal_deg"], "")
                self.assertEqual(float(row["action_yaw"]), 0.)
                self.assertEqual(float(row["action_pitch"]), 0.)
        for tick in range(20, 24):
            self.assertLess(events.index(("predict", tick+1)), events.index(("truth", tick)))
        self.assertEqual(events[-1], ("close", None))


if __name__ == "__main__":
    unittest.main()
