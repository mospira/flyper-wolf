"""One Panda3D render path for capture, training observations, and replay."""

import math

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject, Geom, GeomNode, GeomTriangles, GeomVertexData,
    GeomVertexFormat, GeomVertexWriter, Texture, loadPrcFileData,
)

from flyper_wolf.scene import SceneConfig
from flyper_wolf.simulation import WorldFrame


def mesh(name, vertices, triangles):
    data = GeomVertexData(name, GeomVertexFormat.getV3(), Geom.UHStatic)
    writer = GeomVertexWriter(data, "vertex")
    for vertex in vertices:
        writer.addData3(*vertex)
    primitive = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        primitive.addVertices(*triangle)
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class SceneRenderer(ShowBase):
    def __init__(self, config: SceneConfig, windowed: bool = False):
        w, h = config.image_size
        loadPrcFileData("flyper-wolf", f"""
            win-size {w} {h}
            window-title Flyper Wolf
            sync-video false
            audio-library-name null
            framebuffer-srgb false
            framebuffer-multisample false
            textures-power-2 none
            notify-level warning
        """)
        super().__init__(windowType="onscreen" if windowed else "offscreen")
        if self.win is None:
            raise RuntimeError("Panda3D could not create an OpenGL render surface")
        self.scene_config = config
        self.disableMouse()
        self.setBackgroundColor(0, 0, 0, 1)
        self.camLens.setFov(config.horizontal_fov, math.degrees(
            2 * math.atan(math.tan(math.radians(config.horizontal_fov) / 2) * h / w)))
        self.camLens.setNearFar(0.05, 1000)
        self.camera.setPos(*config.camera_position)
        self.render.setLightOff()
        self.render.setShaderOff()
        width, depth, height = config.room_dimensions
        x, y = width / 2, depth / 2
        faces = [
            [(-x, -y, 0), (-x, y, 0), (-x, y, height), (-x, -y, height)],
            [(x, -y, 0), (x, y, 0), (x, y, height), (x, -y, height)],
            [(-x, -y, 0), (x, -y, 0), (x, -y, height), (-x, -y, height)],
            [(-x, y, 0), (x, y, 0), (x, y, height), (-x, y, height)],
            [(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)],
            [(-x, -y, height), (x, -y, height), (x, y, height), (-x, y, height)],
        ]
        for i, (face, gray) in enumerate(zip(faces, config.wall_gray)):
            wall = self.render.attachNewNode(mesh(f"wall-{i}", face, [(0, 1, 2), (0, 2, 3)]))
            wall.setTwoSided(True)
            wall.setColor(gray, gray, gray, 1)
        vertices, triangles = [], []
        rings, sectors = 24, 48
        for i in range(rings + 1):
            phi = np.pi * i / rings
            for j in range(sectors):
                theta = 2 * np.pi * j / sectors
                vertices.append((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)))
        for i in range(rings):
            for j in range(sectors):
                a, b = i * sectors + j, i * sectors + (j + 1) % sectors
                triangles.extend([(a, b, a + sectors), (b, b + sectors, a + sectors)])
        self.target = self.render.attachNewNode(mesh("target", vertices, triangles))
        self.target.setTwoSided(True)
        self.target.setScale(config.target_radius)
        self.target.setColor(config.target_gray, config.target_gray, config.target_gray, 1)
        self.capture_texture = Texture()
        self.win.addRenderTexture(self.capture_texture, self.win.RTMCopyRam, self.win.RTPColor)
        if not windowed:
            clock = ClockObject.getGlobalClock()
            clock.setMode(ClockObject.MNonRealTime)
            clock.setDt(config.observation_dt)

    def apply(self, frame: WorldFrame):
        self.target.setPos(*frame.target_position)
        # Panda's positive heading turns left; our positive yaw turns right.
        self.camera.setHpr(-frame.yaw, frame.pitch, 0)

    def capture(self, frame: WorldFrame) -> np.ndarray:
        self.apply(frame)
        self.graphicsEngine.renderFrame()
        data = self.capture_texture.getRamImageAs("RGB")
        w, h = self.scene_config.image_size
        # OpenGL rows start at bottom; all downstream arrays start at top.
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[::-1].copy()

    def close(self):
        self.destroy()
