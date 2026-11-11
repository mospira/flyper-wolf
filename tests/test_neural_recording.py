"""Recording spies exercise causal noninterference without model execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flyper_wolf import runtime
from flyper_wolf.cli import _config, parser
from flyper_wolf.configuration import load_config
from flyper_wolf.neural_recording import NeuralRecorder, load_recording


def graph_fly():
    return SimpleNamespace(
        identity={"checkpoint_sha256": "test-checkpoint"},
        cell_types=np.array(["R1", "T4a", "T4a", "T5a"]),
        cell_uv=np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.int32),
        _network=SimpleNamespace(connectome=SimpleNamespace(edges=SimpleNamespace(
            source_index=np.array([0, 1, 2, 3]), target_index=np.array([1, 2, 3, 0])))))


def test_recording_retains_exact_values_graph_order_and_incomplete_prefix(tmp_path):
    fly = graph_fly()
    baseline = np.arange(4, dtype=np.float32)[None]
    recorder = NeuralRecorder(tmp_path, fly, baseline, ticks=3, observation_dt=.01,
                              estimator_identity={}, episode={}, frozen_hash="fly", estimator_hash="estimator")
    assert isinstance(recorder.activity, np.memmap)
    receptors = np.linspace(0, 1, 721, dtype=np.float32).reshape(1, 1, 1, 721)
    activity = baseline + np.float32(.1234567)
    recorder.append(activity, receptors)
    # Changing the source afterward must never alter archival storage.
    expected = activity.copy()
    activity[:] = -99
    recorder.close(complete=False, error="interrupted")
    manifest = json.loads((tmp_path / "neural/manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["complete"] is False
    assert manifest["frames_recorded"] == manifest["valid_prefix_frames"] == 1
    np.testing.assert_array_equal(np.load(tmp_path / "neural/activity.npy")[0], expected[0])
    np.testing.assert_array_equal(np.load(tmp_path / "neural/baseline.npy"), baseline[0])
    np.testing.assert_array_equal(np.load(tmp_path / "neural/inputs.npy")[0], receptors.reshape(721))
    np.testing.assert_array_equal(np.load(tmp_path / "neural/edges.npy"), [[0, 1], [1, 2], [2, 3], [3, 0]])
    with np.load(tmp_path / "neural/cells.npz") as cells:
        np.testing.assert_array_equal(cells["cell_types"], fly.cell_types)
        np.testing.assert_array_equal(cells["axial_uv"], fly.cell_uv)
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == hashlib.sha256((tmp_path / "neural" / name).read_bytes()).hexdigest()
    recorder.close(complete=True)  # Idempotent: failure cannot become success later.
    assert json.loads((tmp_path / "neural/manifest.json").read_text())["complete"] is False


def _spy_episode(monkeypatch, output, *, recording, failure_tick=None, mutate_parameters=False):
    import torch

    events, predictions = [], []

    class Fly:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(graph_fly().__dict__)
            self.steps = 0

        def parameter_hash(self):
            events.append(("fly-hash", self.steps))
            return "mutated" if mutate_parameters and self.steps else "fixed-fly"

        def reset(self):
            events.append(("reset", self.steps))
            return torch.tensor([[.1, .2, .3, .4]], dtype=torch.float32)

        def step(self, receptors):
            self.steps += 1
            if self.steps == failure_tick:
                raise RuntimeError("synthetic interruption")
            return torch.tensor([[self.steps, self.steps + .1, self.steps + .2, self.steps + .3]])

        def outputs(self, activity):
            return activity[:, 1:]

    class Predictor:
        identity = {"test_spy": True}
        fallback = np.array([0., 1., 0.])

        def parameter_hash(self):
            return "fixed-estimator"

        def validate_output_order(self, fly):
            pass

        def predict(self, activity):
            assert activity.shape == (1, 3)
            predictions.append(activity.copy())
            return np.array([[.001 * activity[0, 0], 1., .002]])

    class Renderer:
        def __init__(self, scene):
            pass

        def capture(self, frame):
            return np.full((384, 384, 3), frame.tick, dtype=np.uint8)

        def close(self):
            events.append(("renderer-close", None))

    monkeypatch.setitem(sys.modules, "flyper_wolf.fly", SimpleNamespace(FrozenFly=Fly))
    monkeypatch.setitem(sys.modules, "flyper_wolf.rendering", SimpleNamespace(SceneRenderer=Renderer))
    monkeypatch.setattr(runtime, "load_frozen_estimator", lambda *args, **kwargs: Predictor())
    monkeypatch.setattr(runtime, "_configure_runtime", lambda settings: {"device": "cpu"})
    config = {"episode": {"duration_seconds": .24, "target_seed": 7, "camera_seed": 8},
              "runtime": {"video": False, "record_neural": recording},
              "artifacts": {"estimator_manifest": "unused", "flyvis_model_dir": "unused"}}
    result = runtime.run_episode(deepcopy(config), output)
    return result, events, predictions


def test_capture_does_not_change_predictions_actions_score_or_episode_lifecycle(monkeypatch, tmp_path):
    plain, plain_events, plain_predictions = _spy_episode(monkeypatch, tmp_path / "plain", recording=False)
    saved, saved_events, saved_predictions = _spy_episode(monkeypatch, tmp_path / "saved", recording=True)
    assert (tmp_path / "plain/trace.csv").read_bytes() == (tmp_path / "saved/trace.csv").read_bytes()
    assert plain["metrics"] == saved["metrics"]
    np.testing.assert_array_equal(plain_predictions, saved_predictions)
    assert plain_events == saved_events == [("fly-hash", 0), ("reset", 0), ("fly-hash", 24), ("renderer-close", None)]
    assert not (tmp_path / "plain/neural").exists()
    manifest = json.loads((tmp_path / "saved/neural/manifest.json").read_text())
    assert manifest["complete"] is True and manifest["frames_recorded"] == 24
    assert manifest["frozen_parameters_unchanged"] is True
    activity = np.load(tmp_path / "saved/neural/activity.npy")
    assert activity.shape == (24, 4)
    np.testing.assert_array_equal(activity[:, 0], np.arange(1, 25))
    assert np.load(tmp_path / "saved/neural/inputs.npy").shape == (24, 721)
    np.testing.assert_array_equal(np.load(tmp_path / "saved/neural/baseline.npy"),
                                  np.array([.1, .2, .3, .4], dtype=np.float32))


@pytest.mark.parametrize("failure_tick,mutate_parameters,prefix,match", [
    (23, False, 22, "synthetic interruption"), (None, True, 24, "parameters changed")])
def test_failed_episode_never_marks_capture_complete(monkeypatch, tmp_path, failure_tick, mutate_parameters, prefix, match):
    with pytest.raises(RuntimeError, match=match):
        _spy_episode(monkeypatch, tmp_path, recording=True, failure_tick=failure_tick, mutate_parameters=mutate_parameters)
    manifest = json.loads((tmp_path / "neural/manifest.json").read_text())
    assert manifest["complete"] is False
    assert manifest["frames_recorded"] == prefix
    assert manifest["error"]


def test_recording_is_explicit_and_validated_in_configuration_and_cli(tmp_path):
    path = tmp_path / "episode.yaml"
    path.write_text("{}")
    assert load_config(path)["runtime"]["record_neural"] is False
    args = parser().parse_args(["run", "--config", str(path), "--output", "unused", "--record-neural"])
    assert _config(args)["runtime"]["record_neural"] is True
    path.write_text("runtime:\n  record_neural: 1\n")
    with pytest.raises(ValueError, match="record_neural"):
        load_config(path)

def recorded(run):
    fly = SimpleNamespace(identity={"checkpoint_sha256": "a"*64},
        cell_types=np.array(["T4a", "T4a", "T5a", "Mi1"]),
        cell_uv=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        _network=SimpleNamespace(connectome=SimpleNamespace(edges=SimpleNamespace(
            source_index=np.array([0, 1]), target_index=np.array([1, 2])))))
    rec = NeuralRecorder(run, fly, np.zeros(4, np.float32), ticks=6, observation_dt=.01,
                         estimator_identity={}, episode={}, frozen_hash="fly", estimator_hash="estimator")
    for value in [0, 1, 7, 8, 3, 4]:
        rec.append(np.full(4, value, np.float32), np.full(721, .5, np.float32))
    rec.close(complete=True)


def test_incomplete_or_modified_recording_is_not_visualized(tmp_path):
    recorded(tmp_path)
    manifest_path = tmp_path / "neural/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        load_recording(tmp_path)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest))
    with (tmp_path / "neural/activity.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_recording(tmp_path)




def test_offline_video_preserves_duration_and_checks_source_video(tmp_path):
    av = pytest.importorskip("av")
    from flyper_wolf.neural_video import export_video, frame_ticks
    recorded(tmp_path)
    assert list(frame_ticks(6, .01, 50)) == [0, 2, 4]
    with pytest.raises(ValueError, match="timing"):
        frame_ticks(6, .01, 20)
    output = tmp_path / "activity.mp4"
    report = export_video(tmp_path, output, fps=50)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert [float(frame.time) for frame in frames] == [0, .02, .04]
        assert (frames[0].width, frames[0].height) == (1200, 800)
    assert report["duration_seconds"] == .06
    assert report["video_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_video(tmp_path, output, fps=50)
    (tmp_path / "episode.mp4").write_bytes(b"unrelated")
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "complete", "output_sha256": {}}))
    with pytest.raises(ValueError, match="verified completed run"):
        export_video(tmp_path, tmp_path / "another.mp4", fps=50)
