"""Exact, cross-language episode identity without changing legacy trajectories.

The SplitMix64 permutation maps all uint64 master seeds bijectively onto the
two internal uint32 streams. Persist the decimal string, never a JSON number:
JavaScript numbers cannot represent every uint64 exactly.
"""

import re
from collections.abc import Mapping


SEED_SCHEME = "splitmix64-v1"
_MASK = (1 << 64) - 1
_INCREMENT = 0x9E3779B97F4A7C15
_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MULTIPLIER_2 = 0x94D049BB133111EB


def normalize_master_seed(value: int | str) -> str:
    """Return a canonical decimal uint64 string; reject lossy/coerced values."""
    if type(value) is int:
        number = value
    elif type(value) is str and re.fullmatch(r"0|[1-9][0-9]{0,19}", value):
        number = int(value)
    else:
        raise ValueError("master_seed must be a uint64 integer or canonical decimal string")
    if not 0 <= number <= _MASK:
        raise ValueError("master_seed must be in [0, 18446744073709551615]")
    return str(number)


def derive_seeds(master_seed: int | str) -> tuple[int, int]:
    """Return internal (target, camera) uint32 seeds under splitmix64-v1."""
    value = (int(normalize_master_seed(master_seed)) + _INCREMENT) & _MASK
    value = ((value ^ (value >> 30)) * _MULTIPLIER_1) & _MASK
    value = ((value ^ (value >> 27)) * _MULTIPLIER_2) & _MASK
    value ^= value >> 31
    return value & 0xFFFFFFFF, value >> 32


def _undo_right_xor(value: int, shift: int) -> int:
    result = value
    for bits in range(shift, 64, shift):
        result ^= value >> bits
    return result


def master_seed_from_legacy_pair(target_seed: int, camera_seed: int) -> str:
    """Losslessly migrate an archived pair; this is not an episode control API."""
    if any(type(seed) is not int or not 0 <= seed < 2**32
           for seed in (target_seed, camera_seed)):
        raise ValueError("Legacy target/camera seeds must be uint32 integers")
    value = (camera_seed << 32) | target_seed
    value = _undo_right_xor(value, 31)
    value = (value * pow(_MULTIPLIER_2, -1, 1 << 64)) & _MASK
    value = _undo_right_xor(value, 27)
    value = (value * pow(_MULTIPLIER_1, -1, 1 << 64)) & _MASK
    value = _undo_right_xor(value, 30)
    return str((value - _INCREMENT) & _MASK)


def episode_seed_identity(episode: Mapping) -> dict[str, str]:
    """Read current or archived episode identity without modifying its source.

    Legacy pairs are accepted for archival provenance migration. Callers define
    runtime configuration policy separately and must verify archival checksums
    against the original bytes first.
    """
    if not isinstance(episode, Mapping):
        raise ValueError("Episode seed identity must be a mapping")
    if episode.get("seed_scheme", SEED_SCHEME) != SEED_SCHEME:
        raise ValueError(f"seed_scheme must be {SEED_SCHEME}")
    paired = ("target_seed" in episode, "camera_seed" in episode)
    if any(paired) and not all(paired):
        raise ValueError("Archived episode requires both legacy seeds")
    legacy = (master_seed_from_legacy_pair(episode["target_seed"], episode["camera_seed"])
              if all(paired) else None)
    master = normalize_master_seed(episode["master_seed"]) if "master_seed" in episode else legacy
    if master is None:
        raise ValueError("Episode has no master_seed or complete legacy seed identity")
    if legacy is not None and legacy != master:
        raise ValueError("Master seed and archived legacy seeds identify different episodes")
    return {"master_seed": master, "seed_scheme": SEED_SCHEME}
