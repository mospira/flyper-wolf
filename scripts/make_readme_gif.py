"""Encode a display-only GIF from a completed episode video."""

import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


def encode(source, output, *, fps=10, width=384, speed=1, label=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if fps < 1 or fps > 50 or width < 1 or speed < 1:
        raise ValueError("Require fps in 1..50, positive width and speed >= 1")
    frames, next_sample, duration = [], 0.0, 0.0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        source_rate = float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index/source_rate
            duration = timestamp + 1/source_rate
            if timestamp + 1e-9 < next_sample:
                continue
            rgb = frame.to_image()
            height = round(rgb.height * width / rgb.width)
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
            if label:
                labeled = Image.new("RGB", (width, height+28), "#0c1017")
                labeled.paste(rgb, (0,28))
                ImageDraw.Draw(labeled).text((8,6), label, font=ImageFont.load_default(size=15), fill="white")
                rgb = labeled
            frames.append(rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
            next_sample += speed/fps
    if not frames:
        raise ValueError("Input video has no decodable frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000/fps), loop=0, optimize=True, disposal=2)
    receipt = {"source_video_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "gif_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "source_duration_seconds": duration, "playback_speed": speed,
               "frames": len(frames), "fps": fps, "width": width, "display_label": label,
               "purpose": "Display-only derivative; episode metrics come from the original scoring trace"}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--label", help="Optional model/controller caption above the video")
    args = parser.parse_args()
    print(json.dumps(encode(args.source, args.output, fps=args.fps, width=args.width, speed=args.speed, label=args.label), indent=2))
