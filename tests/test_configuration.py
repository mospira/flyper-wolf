from pathlib import Path

import pytest

from flyper_wolf.configuration import load_config


def test_paths_resolve_at_config_and_cli_overrides_win(tmp_path):
    config = tmp_path / "settings" / "episode.yaml"
    config.parent.mkdir()
    config.write_text("model_bundle: ../models/local\nepisode:\n  duration_seconds: 1\n")
    resolved = load_config(config, {"episode.duration_seconds": 2, "runtime.video": False})
    assert Path(resolved["model_bundle"]) == tmp_path / "models" / "local"
    assert resolved["episode"]["duration_seconds"] == 2
    assert resolved["runtime"]["video"] is False


@pytest.mark.parametrize("text", ["episode:\n  duraton_seconds: 10\n", "runtime:\n  integration_dt: 0.003\n",
                                  "episode:\n  duration_seconds: 0.1\n", "episode:\n  initial_yaw: 3\n"])
def test_rejects_mistakes_that_change_timing_or_pose(tmp_path, text):
    path = tmp_path / "episode.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
