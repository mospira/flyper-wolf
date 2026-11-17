"""Encode the 50 Hz neural overview using one shared, undithered palette."""

import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageSequence

from flyper_wolf.media_labels import seed_comment
from flyper_wolf.neural_recording import _hash


def encode(run, source, output):
    run, source, output = Path(run), Path(source), Path(output)
    receipt_path = source.with_suffix(".json")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite GIF or receipt: {output}")
    render = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (render["video_sha256"] != _hash(source)
            or render["source_manifest_sha256"] != _hash(run / "neural/manifest.json")
            or render["episode_video_sha256"] != _hash(run / "episode.mp4")):
        raise ValueError("Video, render receipt and source recording must agree")
    if (render["fps"], render["frames"], render["duration_seconds"]) != (50, 500, 10):
        raise ValueError("Overview requires a 10-second, 500-frame, 50 fps source")
    identity = {key: render[key] for key in ("master_seed", "seed_scheme")}
    indices = list(range(0, 500, 25))
    samples = []
    timestamps = []
    with av.open(str(source)) as container:
        if any(container.metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Source-video seed identity disagrees with render receipt")
        for index, frame in enumerate(container.decode(video=0)):
            timestamps.append(float(frame.time))
            if index in indices:
                samples.append(frame.to_image().resize((960, 640), Image.Resampling.LANCZOS))
    if len(timestamps) != 500 or any(abs(t - i/50) > 1e-8 for i, t in enumerate(timestamps)):
        raise ValueError("Source-video frames do not have the expected exact cadence")
    atlas = Image.new("RGB", (960*4, 640*5))
    for index, sample in enumerate(samples):
        atlas.paste(sample, (index % 4 * 960, index // 4 * 640))
    palette = atlas.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = []
    with av.open(str(source)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_image().resize((960, 640), Image.Resampling.LANCZOS)
            frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=20,
                   loop=0, disposal=1, optimize=True, comment=seed_comment(identity))
    hashes, durations = [], []
    with Image.open(output) as gif:
        assert gif.size == (960, 640) and gif.info["loop"] == 0
        assert gif.info["comment"] == seed_comment(identity)
        for frame in ImageSequence.Iterator(gif):
            hashes.append(hashlib.sha256(frame.convert("RGB").tobytes()).hexdigest())
            durations.append(frame.info["duration"])
    if len(hashes) != 500 or len(set(hashes)) != 500 or set(durations) != {20}:
        raise ValueError("Encoded GIF did not preserve 500 unique frames at 20 ms each")
    report = {
        "schema_version": 1, **identity,
        "source_video": source.as_posix(), "source_video_sha256": _hash(source),
        "source_render_receipt": receipt_path.as_posix(), "source_render_receipt_sha256": _hash(receipt_path),
        "source_neural_manifest": (run / "neural/manifest.json").as_posix(),
        "source_neural_manifest_sha256": _hash(run / "neural/manifest.json"),
        "source_episode_metadata": (run / "metadata.json").as_posix(),
        "source_episode_metadata_sha256": _hash(run / "metadata.json"),
        "source_episode_video": (run / "episode.mp4").as_posix(),
        "source_episode_video_sha256": _hash(run / "episode.mp4"),
        "renderer_sha256": render["renderer_sha256"], "encoder_sha256": _hash(__file__),
        "gif_sha256": _hash(output), "gif_size_bytes": output.stat().st_size,
        "source_duration_seconds": 10.0, "source_video_fps": 50, "source_video_frames": 500,
        "playback_speed": 1, "frames": 500, "fps": 50, "frame_duration_ms": 20,
        "total_duration_ms": sum(durations), "loop": 0, "width": 960, "height": 640,
        "display_label": render["display_label"], "label_source": "Already rendered in the source neural video",
        "source_observation_hz": 1 / render["source_dt"], "source_ticks": render["source_ticks"],
        "source_dt": render["source_dt"], "display_stride": render["display_stride"],
        "sampling": "Every second recorded 100 Hz activity/input sample; synchronized source episode at 50 fps; no interpolation or duplicated frames",
        "view": render["view"],
        "view_provenance": "Fixed presentation view of actual visual-field coordinates and schematic cell-type depth; activity, receptor input and episode images remain time-varying recorded data",
        "encoding": {
            "palette": "Shared 128-color median-cut palette from 20 uniformly sampled source-video frames",
            "palette_sample_frame_indices": indices, "palette_sample_atlas": "Four columns by five rows, in chronological order",
            "dither": False, "disposal": 1, "optimize": True,
            "resize": "Lanczos from 1200x800 to 960x640",
        },
        "validation": {"decoded_frames": len(hashes), "unique_decoded_frames": len(set(hashes)),
                       "all_frame_durations_ms": 20, "total_duration_ms": sum(durations),
                       "source_frame_timestamps_match_50fps": True},
        "purpose": "Display-only derivative; episode metrics come from the original scoring trace",
        "seed_identity_annotation": {**identity, "annotation": "Exact episode identity embedded in GIF comment during encoding"},
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = encode(args.run, args.source, args.output)
    print(json.dumps({key: report[key] for key in ("gif_sha256", "gif_size_bytes", "frames", "view")}, indent=2))
