"""Attach episode identity to GIF metadata, preserving the encoded animation.

No image is decoded or recompressed. All image descriptors, palettes, compressed
pixels, timing, loop settings and unrelated extensions retain their exact bytes.
"""

import hashlib
from pathlib import Path

from .media_labels import seed_comment
from .seeding import SEED_SCHEME, normalize_master_seed


_PREFIX = b"FLYPER_EPISODE "


def _subblocks(data, offset):
    start, payload = offset, bytearray()
    while True:
        if offset >= len(data):
            raise ValueError("Truncated GIF extension or image data")
        size = data[offset]
        offset += 1
        if not size:
            return offset, bytes(payload)
        if offset + size > len(data):
            raise ValueError("Truncated GIF subblock")
        payload.extend(data[offset:offset+size])
        offset += size


def _without_seed_comment(data):
    if len(data) < 14 or data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("Expected a complete GIF")
    offset = 13 + (3 * (1 << ((data[10] & 7) + 1)) if data[10] & 128 else 0)
    result = bytearray(data[:offset])
    while offset < len(data):
        start, marker = offset, data[offset]
        offset += 1
        if marker == 0x3B:
            if offset != len(data):
                raise ValueError("Unexpected bytes after GIF trailer")
            return bytes(result) + b";"
        if marker == 0x21:
            if offset >= len(data):
                raise ValueError("Truncated GIF extension")
            label = data[offset]
            offset, payload = _subblocks(data, offset+1)
            if label == 0xFE and payload.startswith(_PREFIX):
                continue
        elif marker == 0x2C:
            if offset + 9 > len(data):
                raise ValueError("Truncated GIF image descriptor")
            packed = data[offset+8]
            offset += 9 + (3 * (1 << ((packed & 7) + 1)) if packed & 128 else 0)
            offset, _ = _subblocks(data, offset+1)  # LZW minimum code size.
        else:
            raise ValueError(f"Unexpected GIF block: {marker:#x}")
        result.extend(data[start:offset])
    raise ValueError("Missing GIF trailer")


def tag_gif(path, master_seed):
    """Write or replace only our comment; return verifiable migration evidence."""
    path = Path(path)
    identity = {"master_seed": normalize_master_seed(master_seed), "seed_scheme": SEED_SCHEME}
    before = path.read_bytes()
    animation = _without_seed_comment(before)
    comment = seed_comment(identity)
    blocks = b"".join(bytes([len(comment[i:i+255])]) + comment[i:i+255]
                      for i in range(0, len(comment), 255))
    header_size = 13 + (3 * (1 << ((animation[10] & 7) + 1)) if animation[10] & 128 else 0)
    after = animation[:header_size] + b"!\xfe" + blocks + b"\0" + animation[header_size:]
    if _without_seed_comment(after) != animation:
        raise RuntimeError("GIF annotation altered animation data")
    if after != before:
        path.write_bytes(after)
    return {**identity, "gif_sha256": hashlib.sha256(after).hexdigest(),
            "annotation": "GIF comment only; encoded pixels, palettes, timing and loop bytes unchanged",
            "animation_payload_sha256": hashlib.sha256(animation).hexdigest()}


def gif_animation_payload_sha256(path):
    """Hash the exact GIF bytes excluding only our episode identity comment."""
    return hashlib.sha256(_without_seed_comment(Path(path).read_bytes())).hexdigest()
