import hashlib

import pytest
from PIL import Image

from flyper_wolf.media_identity import _without_seed_comment, tag_gif


def test_annotation_preserves_animation_and_is_idempotent(tmp_path):
    path = tmp_path / "episode.gif"
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[20, 40], loop=0,
                   comment=b"Original comment")
    before = path.read_bytes()
    result = tag_gif(path, "18446744073709551615")
    tagged = path.read_bytes()
    assert _without_seed_comment(tagged) == before
    assert result["animation_payload_sha256"] == hashlib.sha256(before).hexdigest()
    with Image.open(path) as image:
        assert b"master_seed=18446744073709551615" in image.info["comment"]
        assert image.n_frames == 2 and image.info["loop"] == 0
        assert image.info["duration"] == 20
        image.seek(1)
        assert image.info["duration"] == 40
    assert tag_gif(path, "18446744073709551615") == result
    assert path.read_bytes() == tagged
    tag_gif(path, "42")
    assert _without_seed_comment(path.read_bytes()) == before


def test_invalid_gif_is_not_overwritten(tmp_path):
    path = tmp_path / "episode.gif"
    path.write_bytes(b"GIF89a truncated")
    with pytest.raises(ValueError):
        tag_gif(path, "42")
    assert path.read_bytes() == b"GIF89a truncated"
