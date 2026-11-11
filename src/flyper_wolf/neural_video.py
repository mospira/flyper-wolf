"""Render recorded circuit activity to MP4 offline, without a browser or models."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .neural_recording import _hash, load_recording
from .preprocessing import VisualInput
from .media_labels import recording_label


def frame_ticks(count, dt, fps):
    if type(fps) is not int or not 1 <= fps <= 100:
        raise ValueError("Video FPS must be an integer between 1 and 100")
    stride = round(1 / (fps * dt))
    if stride < 1 or not math.isclose(stride * fps * dt, 1) or count % stride:
        raise ValueError("Video timing must divide the recording clock and episode duration exactly")
    return range(0, count, stride)


class ActivityScene:
    """Fixed-scale point cloud: true visual-field coordinates, schematic depth."""

    def __init__(self, recording, label="Recorded estimator + controller"):
        self.r = recording
        self.label = label
        types = recording["cell_types"]
        names = list(dict.fromkeys(types.tolist()))
        self.scales = np.empty(len(types), dtype=np.float32)
        self.scale_by_type = {}
        depth = np.empty(len(types))
        for index, name in enumerate(names):
            mask = np.flatnonzero(types == name)
            delta = np.asarray(recording["activity"][:, mask]) - recording["baseline"][mask]
            scale = max(float(np.percentile(np.abs(delta), 99.5)), 1e-6)
            self.scales[mask] = scale
            self.scale_by_type[name] = scale
            depth[mask] = (index - (len(names)-1)/2) * .85
        u, v = recording["axial_uv"].T
        self.xyz = np.column_stack((1.5*v, -np.sqrt(3)*(u+v/2), depth))
        self.font = ImageFont.load_default(size=18)
        self.small = ImageFont.load_default(size=14)
        self.title = ImageFont.load_default(size=26)
        self.receptors = VisualInput().axial_coordinates

    def render(self, tick, episode_image=None):
        dt = self.r["manifest"]["observation_dt"]
        time = tick * dt
        yaw, pitch = .67 + .07*time, -.28
        x, y, z = self.xyz.T
        rx = np.cos(yaw)*x + np.sin(yaw)*z
        rz = -np.sin(yaw)*x + np.cos(yaw)*z
        ry = np.cos(pitch)*y - np.sin(pitch)*rz
        rz = np.sin(pitch)*y + np.cos(pitch)*rz
        px = np.rint(380 + 760*rx/(130-rz)).astype(int)
        py = np.rint(325 - 760*ry/(130-rz)).astype(int)
        delta = (self.r["activity"][tick] - self.r["baseline"]) / self.scales
        strength = np.clip(np.abs(delta), 0, 1)
        tint = np.where(delta[:, None] < 0, [38., 148., 255.], [255., 110., 48.])
        colors = tint * (.025 + .8*strength[:, None])
        cloud = np.zeros((620, 760, 3), dtype=np.float32)
        # Additive splats show all model units; no fabricated anatomy or edges.
        for dx, dy, weight in ((0,0,1),(-1,0,.35),(1,0,.35),(0,-1,.35),(0,1,.35)):
            xx, yy = px+dx, py+dy
            keep = (xx>=0)&(xx<760)&(yy>=0)&(yy<620)
            np.add.at(cloud, (yy[keep], xx[keep]), colors[keep]*weight)
        cloud += np.array([9, 17, 29], dtype=np.float32)
        canvas = Image.new("RGB", (1200, 800), "#09111d")
        canvas.paste(Image.fromarray(np.clip(cloud, 0, 255).astype(np.uint8)), (0, 125))
        draw = ImageDraw.Draw(canvas)
        draw.text((26,20), f"FLYPER WOLF / {self.label} / RECORDED CIRCUIT ACTIVITY", font=self.small, fill="#8ae5d6")
        draw.text((26,48), "Inside the frozen visual circuit", font=self.title, fill="white")
        draw.text((26,88), f"{len(delta):,} model units | {len(self.scale_by_type)} cell types | input {time:.2f}s -> state {time+dt:.2f}s", font=self.font, fill="#bdccdd")
        draw.line((780,135,780,740), fill="#2b3c50")
        draw.text((805,138), "SYNCHRONIZED EPISODE", font=self.small, fill="#bdccdd")
        if episode_image is not None:
            panel = Image.fromarray(episode_image)
            panel.thumbnail((352, 330))
            canvas.paste(panel, (805+(352-panel.width)//2,168))
        else:
            draw.text((805,220), "No episode video in this recording", font=self.small, fill="#bdccdd")
        draw.text((805,520), "RECEPTOR INPUT", font=self.small, fill="#bdccdd")
        for (u, v), value in zip(self.receptors, self.r["inputs"][tick]):
            xx, yy = 980+v*5, 642+(u+v/2)*5.7
            shade = round(float(value)*255)
            draw.ellipse((xx-2,yy-2,xx+2,yy+2), fill=(shade,shade,shade))
        draw.text((26,151), "Below gray baseline", font=self.small, fill="#43a9ff")
        draw.text((230,151), "Above gray baseline", font=self.small, fill="#ff9862")
        draw.text((26,177), "Fixed scale per cell type; color intensity is not comparable across types.", font=self.small, fill="#9bafc7")
        draw.text((26,747), "Visual-field positions from the model; depth is schematic, not brain anatomy or processing order.", font=self.small, fill="#bdccdd")
        draw.text((26,773), "Continuous model activity, not spikes. Camera rotation is for presentation only; model weights remain frozen.", font=self.small, fill="#bdccdd")
        return np.asarray(canvas)


def export_video(run, output, fps=20):
    import av

    run, output = Path(run), Path(output)
    receipt = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Neural video output must end in .mp4")
    if output.exists() or receipt.exists():
        raise FileExistsError(f"Refusing to overwrite video or receipt: {output}")
    recording = load_recording(run)
    count, dt = len(recording["activity"]), recording["manifest"]["observation_dt"]
    ticks = frame_ticks(count, dt, fps)
    source = run / "episode.mp4"
    if source.exists():
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        hashes = metadata.get("output_sha256", {})
        if (metadata.get("status") != "complete" or hashes.get("episode.mp4") != _hash(source)
                or hashes.get("neural/manifest.json") != _hash(run / "neural/manifest.json")):
            raise ValueError("Episode video and neural recording must share a verified completed run")
    scene = ActivityScene(recording, recording_label(run, recording))
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder = av.open(str(source)) if source.exists() else None
    try:
        frames = iter(decoder.decode(video=0)) if decoder else iter(())
        upcoming = next(frames, None)
        current_image = None
        with av.open(str(output), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = 1200, 800, "yuv420p"
            stream.options = {"crf": "20", "preset": "medium"}
            for tick in ticks:
                while upcoming is not None and float(upcoming.time) <= tick*dt + 1e-8:
                    current_image = upcoming.to_ndarray(format="rgb24")
                    upcoming = next(frames, None)
                frame = av.VideoFrame.from_ndarray(scene.render(tick, current_image), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    finally:
        if decoder:
            decoder.close()
    report = {"schema_version": 1, "display_label": scene.label, "video_sha256": _hash(output),
              "source_manifest_sha256": _hash(run / "neural/manifest.json"),
              "renderer_sha256": _hash(__file__), "source_ticks": count,
              "frames": len(ticks), "fps": fps, "duration_seconds": count*dt,
              "source_dt": dt, "display_stride": ticks.step,
              "units": len(scene.scales), "scales_by_type": scene.scale_by_type,
              "scale": "Symmetric per-type 99.5th percentile absolute deviation over the full source recording; clipped for display",
              "layout": "Actual axial visual-field coordinates; schematic type depth, not anatomy or processing order",
              "input_time": "tick * dt; episode video uses the latest frame at or before this time",
              "activity_time": "(tick + 1) * dt", "signal": "Continuous activity minus actual gray-adapted reset state, not spikes",
              "episode_video_sha256": _hash(source) if source.exists() else None,
              "model": recording["manifest"]["model"]}
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
