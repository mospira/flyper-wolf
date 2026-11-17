"""Cross-language vectors and lossless migration of consumed watch episodes."""

import json

import numpy as np
import pytest

from flyper_wolf.cli import _config, parser
from flyper_wolf.configuration import load_config
from flyper_wolf.scene import SceneConfig
from flyper_wolf.seeding import (
    SEED_SCHEME, derive_seeds, episode_seed_identity, master_seed_from_legacy_pair, normalize_master_seed,
)
from flyper_wolf.simulation import Simulation


@pytest.mark.parametrize("master,target,camera", [
    ("0", 2065550767, 3793791033),
    ("1", 2298633409, 2433363436),
    ("42", 803958421, 3184996902),
    ("9007199254740993", 3543363639, 89972927),
    ("18446744073709551615", 459615264, 3839455607),
    ("9122296628466624745", 302000001, 303000001),
    ("7604549466244035895", 304000001, 305000001),
])
def test_known_vectors_and_inverse(master, target, camera):
    assert derive_seeds(master) == derive_seeds(int(master)) == (target, camera)
    assert master_seed_from_legacy_pair(target, camera) == master
    assert normalize_master_seed(json.loads(json.dumps(master))) == master


def test_inverse_covers_arbitrary_legacy_pairs():
    rng = np.random.default_rng(74)
    pairs = [(0, 0), (2**32 - 1, 0), (0, 2**32 - 1), (2**32 - 1, 2**32 - 1)]
    pairs += [tuple(map(int, row)) for row in rng.integers(0, 2**32, size=(256, 2), dtype=np.uint64)]
    for pair in pairs:
        assert derive_seeds(master_seed_from_legacy_pair(*pair)) == pair


def test_archival_identity_resolution_preserves_source_and_rejects_ambiguity():
    original = {"target_seed": 302000001, "camera_seed": 303000001}
    assert episode_seed_identity(original) == {
        "master_seed": "9122296628466624745", "seed_scheme": SEED_SCHEME}
    assert original == {"target_seed": 302000001, "camera_seed": 303000001}
    for identity in [{}, {"target_seed": 1}, {"master_seed": "0", **original},
                     {"master_seed": "0", "seed_scheme": "future"}]:
        with pytest.raises(ValueError):
            episode_seed_identity(identity)


@pytest.mark.parametrize("value", [True, False, 1.0, None, -1, 2**64, "", "01", "00", "+1", "-1",
                                   "1.0", "1e3", " 1", "1 ", "18446744073709551616", "١"])
def test_rejects_coercion_and_noncanonical_seed_values(value):
    with pytest.raises(ValueError, match="master_seed"):
        normalize_master_seed(value)


def test_migration_preserves_camera_draws_and_target_motion():
    target_seed, camera_seed = derive_seeds("9122296628466624745")
    old_rng, new_rng = np.random.default_rng(303000001), np.random.default_rng(camera_seed)
    assert [old_rng.uniform(-25, 25), old_rng.uniform(-16, 16)] == [new_rng.uniform(-25, 25), new_rng.uniform(-16, 16)]
    old = Simulation(SceneConfig(), seed=302000001, target_speed=1.1)
    new = Simulation(SceneConfig(), seed=target_seed, target_speed=1.1)
    for action in [(0, 0), (.2, -.1), (-1, 1), (0, 0)]:
        assert old.snapshot() == new.snapshot()
        old.step(action)
        new.step(action)


def test_config_and_cli_persist_one_exact_master_seed(tmp_path):
    path = tmp_path / "episode.yaml"
    path.write_text("{}")
    args = parser().parse_args(["run", "--config", str(path), "--output", "unused",
                               "--master-seed", "18446744073709551615"])
    episode = _config(args)["episode"]
    assert episode["master_seed"] == "18446744073709551615"
    assert episode["seed_scheme"] == SEED_SCHEME
    assert not {"target_seed", "camera_seed"} & episode.keys()
    for text in ["episode:\n  target_seed: 7\n", "episode:\n  camera_seed: 8\n",
                 "runtime:\n  seed: 74\n", "episode:\n  seed_scheme: unknown\n"]:
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)
    for flag in ["--target-seed", "--camera-seed"]:
        with pytest.raises(SystemExit):
            parser().parse_args(["run", "--output", "unused", flag, "7"])
    path.write_text("episode:\n  master_seed: 9007199254740993\n")
    assert load_config(path)["episode"]["master_seed"] == "9007199254740993"
