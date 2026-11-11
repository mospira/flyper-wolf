"""Make fixed-scale spatial activity maps from a verified neural recording."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flyper_wolf.neural_recording import load_recording
from flyper_wolf.preprocessing import VisualInput
from flyper_wolf.media_labels import recording_label


LAYERS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def make_gif(run, output, fps=10):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    recording = load_recording(run)
    label = recording_label(run, recording)
    activity, baseline = recording["activity"], recording["baseline"]
    inputs, types, uv = recording["inputs"], recording["cell_types"], recording["axial_uv"]
    dt = recording["manifest"]["observation_dt"]
    stride = round(1 / (fps*dt))
    if fps not in (1, 2, 4, 5, 10, 20, 25, 50) or stride < 1 or not np.isclose(stride*fps*dt, 1):
        raise ValueError("Choose a GIF rate dividing both 100 Hz and the GIF timing clock")
    masks = {name: np.flatnonzero(types == name) for name in LAYERS}
    if any(not len(mask) for mask in masks.values()):
        raise ValueError("Recording must include all T4/T5 subtypes")
    deltas = {name: np.asarray(activity[:, mask]) - baseline[mask] for name, mask in masks.items()}
    # Shared scale allows comparisons between displayed types; fixed over time.
    limit = max(float(np.percentile(np.abs(np.concatenate(list(deltas.values()), axis=1)), 99.5)), 1e-6)
    size, header, footer = 222, 54, 56
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    polygons = {}
    for name in ("Input", *LAYERS):
        coords = VisualInput().axial_coordinates if name == "Input" else uv[masks[name]]
        u, v = coords.T
        centers = np.column_stack((1.5*v*3.35+size/2, np.sqrt(3)*(u+v/2)*3.35+size/2+8))
        corners = 3.35*.94*np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)))
        polygons[name] = [[tuple(point) for point in center+corners] for center in centers]
    frames = []
    for tick in range(0, len(activity), stride):
        canvas = Image.new("RGB", (size*3, header+size*3+footer), "#101923")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"FLYPER WOLF / {label}", font=font, fill="white")
        draw.text((12, 31), f"input at {tick*dt:.2f}s -> neural state at {(tick+1)*dt:.2f}s | frozen weights", font=small, fill="#bbcbd8")
        for index, name in enumerate(("Input", *LAYERS)):
            tile = Image.new("RGB", (size, size), "#101923")
            td = ImageDraw.Draw(tile)
            td.text((10, 4), "Receptor brightness" if name == "Input" else name, font=font, fill="white")
            if name == "Input":
                colors = np.repeat(np.rint(inputs[tick, :, None]*255).astype(np.uint8), 3, axis=1)
            else:
                value = np.clip(deltas[name][tick]/limit, -1, 1)
                positive, negative = np.maximum(value, 0), np.maximum(-value, 0)
                colors = np.rint(np.stack((1-negative, 1-np.abs(value), 1-positive), axis=1)*255).astype(np.uint8)
            for polygon, color in zip(polygons[name], colors):
                td.polygon(polygon, fill=tuple(color))
            canvas.paste(tile, ((index%3)*size, header+(index//3)*size))
        y = header+size*3
        draw.text((12, y+8), f"BLUE: below gray baseline | WHITE: unchanged | RED: above (+/- {limit:.3f})", font=small, fill="white")
        draw.text((12, y+30), "Spatial cell maps; continuous model activity, not spikes. Fixed shared color scale.", font=small, fill="#bbcbd8")
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000/fps), loop=0, disposal=2, optimize=True)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = {"schema_version": 1, "display_label": label, "source_neural_manifest_sha256": digest(Path(run)/"neural/manifest.json"),
               "gif_sha256": digest(output), "source_ticks": len(activity), "frames": len(frames),
               "fps": fps, "source_dt": dt, "display_stride": stride, "layers": list(LAYERS),
               "baseline": "Actual pre-episode neutral-gray-adapted circuit state",
               "color_limit": limit, "scale": "Shared symmetric 99.5th percentile across shown types and all source ticks",
               "input_timestamp": "tick * dt", "activity_timestamp": "(tick + 1) * dt",
               "model": recording["manifest"].get("model"),
               "purpose": "Recorded activity visualization; display-only temporal downsampling"}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(frames), "color_limit": limit}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    make_gif(args.run, args.output, args.fps)
