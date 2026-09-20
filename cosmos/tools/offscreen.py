"""
Headless renderer -- draw frames to PNG with no window.

Useful for three things:
  * verifying the shaders compile and produce sane output on a machine with
    no display (CI, a server, a container),
  * generating the screenshots for your report,
  * rendering a sequence of frames to stitch into a video.

Usage:
    python -m cosmos.tools.offscreen --scene solar --out shots/
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import moderngl

from ..engine.camera import Camera
from ..engine.renderer import Renderer
from ..engine.trails import Trails
from ..physics import constants as K
from ..physics.bodies import KIND_ASTEROID, KIND_BLACKHOLE
from ..physics.simulation import Simulation
from ..scenes import load


def save_png(fbo_bytes, w, h, path):
    from PIL import Image
    img = Image.frombytes("RGB", (w, h), fbo_bytes)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="solar")
    ap.add_argument("--out", default="shots")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--spin-days", type=float, default=0.0,
                    help="simulated days to advance between frames")
    ap.add_argument("--visual-scale", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    store, meta = load(args.scene)
    sim = Simulation(store, dt=meta.get("suggested_dt", 3600.0),
                     integrator="verlet",
                     relativistic=meta.get("relativistic", False))

    ctx = moderngl.create_standalone_context(backend="egl")
    print("GL:", ctx.info["GL_RENDERER"], "|", ctx.info["GL_VERSION"])

    rend = Renderer(ctx, args.width, args.height, max_bodies=store.n)

    # A standalone context has no default framebuffer, so make one.
    out_tex = ctx.texture((args.width, args.height), 4)
    out_fbo = ctx.framebuffer([out_tex])
    rend.set_output(out_fbo)

    cam = Camera(fov_deg=55.0)
    focus = store.index(meta["focus"]) if meta.get("focus") in store.names else 0
    d = meta.get("start_distance", 5 * K.AU)
    cam.position = store.pos[focus] + np.array([0.35 * d, 0.42 * d, 0.85 * d])
    cam.look_at(store.pos[focus])

    vs = args.visual_scale
    if vs is None:
        vs = 900.0 if args.scene == "solar" else 1.0
    star_scale = 22.0 if args.scene == "solar" else 1.0

    tracked = [i for i in range(store.n)
               if store.trail_flag[i] and store.kind[i] != KIND_ASTEROID][:14]
    trails = Trails(tracked, capacity=900)

    for f in range(args.frames):
        if args.spin_days > 0 and f > 0:
            sim.advance(args.spin_days * K.DAY, max_substeps=100000)
        trails.record(store.pos)
        rend.render(store, cam, trails, visual_scale=vs, star_scale=star_scale)
        ctx.finish()
        data = out_fbo.read(components=3)
        path = os.path.join(args.out, f"{args.scene}_{f:04d}.png")
        save_png(data, args.width, args.height, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
