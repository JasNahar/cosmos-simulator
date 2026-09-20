"""
The renderer.

Frame structure:

    1. Cull and sort bodies on the CPU (float64), producing camera-relative
       instance data in float32.
    2. Render the starfield into an HDR framebuffer.
    3. Render bodies as GPU sphere impostors -- two draw calls, one for
       things big enough to shade and one for distant points.
    4. Render orbit trails.
    5. If the scene has black holes, run the gravitational lensing pass.
    6. Tonemap HDR -> screen.
    7. Blit the HUD.

Everything the GPU sees is camera-relative and float32; everything the CPU
computes is absolute and float64. That boundary is the single most important
design decision in the whole renderer.
"""

from __future__ import annotations

import numpy as np

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None

from ..physics import constants as K
from ..physics.bodies import KIND_BLACKHOLE, KIND_STAR
from . import shaders
from .sky import make_starfield

MAX_LIGHTS = 8
MAX_BH = 4


class Renderer:
    def __init__(self, ctx, width: int, height: int, max_bodies: int,
                 star_count: int = 9000):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.max_bodies = max_bodies

        self.exposure = 1.0
        self.ambient = 0.012
        self.min_pixels = 1.4          # LOD threshold
        self.show_trails = True
        self.show_stars = True

        # ---- programs ----
        self.prog_body = ctx.program(vertex_shader=shaders.BODY_VS,
                                     fragment_shader=shaders.BODY_FS)
        self.prog_stars = ctx.program(vertex_shader=shaders.STARS_VS,
                                      fragment_shader=shaders.STARS_FS)
        self.prog_trail = ctx.program(vertex_shader=shaders.TRAIL_VS,
                                      fragment_shader=shaders.TRAIL_FS)
        self.prog_lens = ctx.program(vertex_shader=shaders.LENS_VS,
                                     fragment_shader=shaders.LENS_FS)
        self.prog_tone = ctx.program(vertex_shader=shaders.BLIT_VS,
                                     fragment_shader=shaders.TONEMAP_FS)
        self.prog_hud = ctx.program(vertex_shader=shaders.BLIT_VS,
                                    fragment_shader=shaders.HUD_FS)

        # ---- static geometry ----
        # One quad, reused by every body via instancing. Four vertices for
        # the entire scene's worth of spheres.
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo_quad = ctx.buffer(quad.tobytes())

        fullscreen = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.vbo_fs = ctx.buffer(fullscreen.tobytes())
        self.vao_lens = ctx.vertex_array(self.prog_lens,
                                         [(self.vbo_fs, "2f", "in_pos")])
        self.vao_tone = ctx.vertex_array(self.prog_tone,
                                         [(self.vbo_fs, "2f", "in_pos")])
        self.vao_hud = ctx.vertex_array(self.prog_hud,
                                        [(self.vbo_fs, "2f", "in_pos")])

        # ---- dynamic instance buffers ----
        # 8 floats per body: centre(3) radius(1) colour(3) emissive(1)
        #
        # Two buffers, not one, because the two body passes need different
        # GL state (opaque vs additive) and OpenGL instancing has no way to
        # start partway through a buffer.
        self.stride = 8
        self.vbo_solid = ctx.buffer(reserve=max_bodies * self.stride * 4,
                                    dynamic=True)
        self.vbo_point = ctx.buffer(reserve=max_bodies * self.stride * 4,
                                    dynamic=True)
        fmt = ("3f 1f 3f 1f /i", "in_center", "in_radius",
               "in_color", "in_emissive")
        self.vao_solid = ctx.vertex_array(
            self.prog_body,
            [(self.vbo_quad, "2f", "in_corner"), (self.vbo_solid, *fmt)])
        self.vao_point = ctx.vertex_array(
            self.prog_body,
            [(self.vbo_quad, "2f", "in_corner"), (self.vbo_point, *fmt)])

        # ---- starfield ----
        dirs, cols, sizes = make_starfield(star_count)
        star_data = np.concatenate(
            [dirs, cols, sizes[:, None]], axis=1).astype("f4")
        self.vbo_stars = ctx.buffer(star_data.tobytes())
        self.vao_stars = ctx.vertex_array(
            self.prog_stars,
            [(self.vbo_stars, "3f 3f 1f", "in_dir", "in_color", "in_size")],
        )
        # Stars sit just inside the far plane. They must be INSIDE it or
        # they get clipped away entirely -- which looks exactly like "my
        # starfield isn't working" and costs an hour to diagnose.
        self.star_distance = None       # set per-frame from camera.far

        # ---- trails ----
        self.vbo_trail = ctx.buffer(reserve=4000 * 4 * 4, dynamic=True)
        self.vao_trail = ctx.vertex_array(
            self.prog_trail,
            [(self.vbo_trail, "3f 1f", "in_pos", "in_age")],
        )

        self._make_targets(width, height)

        # Scratch buffer reused every frame -- allocating 20k x 8 floats per
        # frame would generate a lot of avoidable garbage.
        self._inst = np.zeros((max_bodies, self.stride), dtype="f4")

    @staticmethod
    def _mat(m):
        """Pack a numpy matrix for GLSL.

        GLSL reads uniform matrices COLUMN-major; numpy `.tobytes()` is
        row-major. So the bytes we upload must be the transpose, otherwise
        every rotation in the scene is silently inverted -- which looks
        almost right and is horrible to debug.
        """
        return np.ascontiguousarray(m.T, dtype="f4").tobytes()

    # ------------------------------------------------------------------ fbos
    def _make_targets(self, w, h):
        ctx = self.ctx
        # f2 = half-float. An HDR target: a star's surface is thousands of
        # times brighter than a lit asteroid, and clamping to [0,1] before
        # tonemapping would turn every star into a flat white disc.
        self.tex_scene = ctx.texture((w, h), 4, dtype="f2")
        self.tex_scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_scene.repeat_x = False
        self.tex_scene.repeat_y = False
        self.depth = ctx.depth_texture((w, h))
        self.fbo_scene = ctx.framebuffer([self.tex_scene], self.depth)

        self.tex_lens = ctx.texture((w, h), 4, dtype="f2")
        self.tex_lens.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_lens = ctx.framebuffer([self.tex_lens])

        # Final destination. With a window this is the default framebuffer;
        # in a standalone (headless) context there is no default framebuffer,
        # so the caller supplies one via `set_output`.
        if getattr(self, "output", None) is None:
            self.output = ctx.screen

    def set_output(self, fbo):
        """Choose the final destination framebuffer (headless rendering)."""
        self.output = fbo

    def resize(self, w, h):
        self.width, self.height = w, h
        for o in (self.fbo_scene, self.fbo_lens, self.tex_scene,
                  self.tex_lens, self.depth):
            o.release()
        self._make_targets(w, h)

    # ------------------------------------------------------- instance build
    def _build_instances(self, store, camera, visual_scale, star_scale):
        """CPU-side culling and packing. Returns (n_solid, n_point, bh_list).

        Bodies are ordered in the buffer as [solid ... | points ...] so the
        two draw calls are contiguous slices with no extra copying.
        """
        alive = store.alive
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            return 0, 0, []

        # --- camera-relative, in float64, then narrowed to float32 --------
        rel = store.pos[idx] - camera.position
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        dist = np.maximum(dist, 1.0)

        # --- frustum cull -------------------------------------------------
        # Keep anything whose direction is within the view cone (plus margin
        # for the body's own angular size). One dot product per body.
        forward, _, _ = camera.basis()
        cos_ang = (rel @ forward) / dist
        half = camera.fov * 0.5 * np.sqrt(1.0 + camera.aspect ** 2) + 0.12
        keep = cos_ang > np.cos(min(half, np.pi * 0.98))

        idx, rel, dist = idx[keep], rel[keep], dist[keep]
        if idx.size == 0:
            return 0, 0, []

        kind = store.kind[idx]
        is_star = kind == KIND_STAR
        is_bh = kind == KIND_BLACKHOLE

        # --- visual radius -------------------------------------------------
        # PHYSICS always uses the true radius. Only the drawn size is scaled,
        # because at true scale Earth is 0.04 pixels wide from 1 AU and the
        # scene is an empty black screen with one dot in it.
        radius = store.radius[idx] * visual_scale
        radius = np.where(is_star, store.radius[idx] * star_scale, radius)
        # A black hole is drawn at its SHADOW radius (2.598 r_s), which is
        # what an observer actually sees -- noticeably bigger than the horizon.
        if np.any(is_bh):
            rs = K.schwarzschild_radius(np.maximum(store.mass[idx], 1.0))
            radius = np.where(is_bh, 2.598 * rs, radius)

        # --- LOD split ------------------------------------------------------
        focal = 1.0 / np.tan(camera.fov * 0.5)
        pixels = (self.height * 0.5) * focal * radius / dist
        solid_mask = (pixels >= self.min_pixels) | is_bh
        order = np.argsort(~solid_mask, kind="stable")   # solids first
        idx, rel, radius = idx[order], rel[order], radius[order]
        n_solid = int(solid_mask.sum())
        n_total = min(idx.size, self.max_bodies)

        # --- pack -----------------------------------------------------------
        buf = self._inst[:n_total]
        buf[:, 0:3] = rel[:n_total]
        buf[:, 3] = radius[:n_total]
        col = store.color[idx[:n_total]]
        kind2 = store.kind[idx[:n_total]]
        buf[:, 4:7] = np.where((kind2 == KIND_BLACKHOLE)[:, None], 0.0, col)
        buf[:, 7] = (kind2 == KIND_STAR).astype("f4")

        n_solid = min(n_solid, n_total)
        if n_solid:
            self.vbo_solid.write(np.ascontiguousarray(buf[:n_solid]).tobytes())
        if n_total > n_solid:
            self.vbo_point.write(np.ascontiguousarray(buf[n_solid:]).tobytes())

        # --- black holes for the lensing pass -------------------------------
        bh = []
        bh_idx = np.flatnonzero(store.kind[:store.n] == KIND_BLACKHOLE)
        for b in bh_idx[:MAX_BH]:
            if not store.alive[b]:
                continue
            r = store.pos[b] - camera.position
            bh.append((r, float(K.schwarzschild_radius(max(store.mass[b], 1.0)))))

        return n_solid, max(0, n_total - n_solid), bh

    # ------------------------------------------------------------- lighting
    def _upload_lights(self, store, camera):
        view = camera.view_rotation()
        stars = np.flatnonzero((store.kind == KIND_STAR) & store.alive)

        # GLSL uniform ARRAYS must be written at their full declared length,
        # so we always upload MAX_LIGHTS entries and use uLightCount to say
        # how many are real.
        pos_view = np.zeros((MAX_LIGHTS, 3), dtype="f4")
        col = np.zeros((MAX_LIGHTS, 3), dtype="f4")
        lum_norm = np.zeros(MAX_LIGHTS, dtype="f4")

        n = 0
        if stars.size:
            # Brightest first -- a scene with 500 stars still only needs the
            # handful that contribute meaningfully to any given surface.
            stars = stars[np.argsort(-store.luminosity[stars])][:MAX_LIGHTS]
            n = int(stars.size)
            rel = store.pos[stars] - camera.position
            pos_view[:n] = (view @ rel.T).T.astype("f4")
            col[:n] = store.color[stars].astype("f4")
            # Normalised so a solar-luminosity star at 1 AU gives irradiance 1.
            lum_norm[:n] = (store.luminosity[stars] / K.L_SUN * K.AU ** 2)

        self.prog_body["uLightCount"].value = n
        self.prog_body["uLightPos"].write(pos_view.tobytes())
        self.prog_body["uLightColor"].write(col.tobytes())
        self.prog_body["uLightLum"].write(lum_norm.tobytes())

    # ----------------------------------------------------------------- frame
    def render(self, store, camera, trails=None, visual_scale=1.0,
               star_scale=1.0, hud_texture=None):
        ctx = self.ctx
        camera.aspect = self.width / self.height
        view = camera.view_rotation()
        proj = camera.projection()
        fcoef = camera.log_depth_coef
        focal = 1.0 / np.tan(camera.fov * 0.5)

        n_solid, n_point, bh = self._build_instances(
            store, camera, visual_scale, star_scale)

        self.fbo_scene.use()
        ctx.viewport = (0, 0, self.width, self.height)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        # ---------------------------------------------------- starfield
        if self.show_stars:
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.depth_mask = False
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            p = self.prog_stars
            p["uView"].write(self._mat(view))
            p["uProj"].write(self._mat(proj))
            p["uFcoef"].value = fcoef
            p["uDistance"].value = float(self.star_distance or camera.far * 0.4)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.vao_stars.render(moderngl.POINTS)
            ctx.depth_mask = True

        # ---------------------------------------------------- bodies
        p = self.prog_body
        p["uView"].write(self._mat(view))
        p["uProj"].write(self._mat(proj))
        p["uFcoef"].value = fcoef
        p["uViewportH"].value = float(self.height)
        p["uFocal"].value = float(focal)
        p["uMinPixels"].value = float(self.min_pixels)
        p["uExposure"].value = float(self.exposure)
        p["uAmbient"].value = float(self.ambient)
        self._upload_lights(store, camera)

        if n_solid > 0:
            # Opaque pass: depth test and depth write on, no blending.
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            ctx.depth_func = "<"
            self.vao_solid.render(moderngl.TRIANGLE_STRIP, instances=n_solid)

        if n_point > 0:
            # Distant points: additive, depth-tested but not depth-writing,
            # so thousands of faint dots accumulate instead of z-fighting
            # with each other.
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            ctx.depth_mask = False
            self.vao_point.render(moderngl.TRIANGLE_STRIP, instances=n_point)
            ctx.depth_mask = True

        # ---------------------------------------------------- trails
        if trails is not None and self.show_trails:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.depth_mask = False
            tp = self.prog_trail
            tp["uView"].write(self._mat(view))
            tp["uProj"].write(self._mat(proj))
            tp["uFcoef"].value = fcoef
            for k, bi in enumerate(trails.indices):
                pts, age = trails.polyline(k, camera.position)
                if pts is None:
                    continue
                data = np.concatenate([pts, age[:, None]], axis=1).astype("f4")
                nbytes = data.nbytes
                if nbytes > self.vbo_trail.size:
                    self.vbo_trail.orphan(nbytes)
                self.vbo_trail.write(data.tobytes())
                tp["uColor"].value = tuple(float(c) for c in store.color[bi])
                self.vao_trail.render(moderngl.LINE_STRIP, vertices=len(data))
            ctx.depth_mask = True

        # ---------------------------------------------------- lensing
        source = self.tex_scene
        if bh:
            self.fbo_lens.use()
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            lp = self.prog_lens
            self.tex_scene.use(0)
            lp["uScene"].value = 0
            lp["uTanHalfFov"].value = float(np.tan(camera.fov * 0.5))
            lp["uAspect"].value = float(camera.aspect)
            lp["uBhCount"].value = len(bh)

            ndc = np.zeros((MAX_BH, 2), dtype="f4")
            rs_a = np.zeros(MAX_BH, dtype="f4")
            dist_a = np.ones(MAX_BH, dtype="f4")
            behind = np.ones(MAX_BH, dtype="f4")
            disk = np.zeros((MAX_BH, 3), dtype="f4")
            for i, (rel, rs) in enumerate(bh):
                v = view @ rel
                d = -v[2]
                if d > 1.0:
                    ndc[i] = ((focal / camera.aspect) * v[0] / d,
                              focal * v[1] / d)
                    behind[i] = 0.0
                rs_a[i] = rs
                dist_a[i] = max(float(np.linalg.norm(rel)), 1.0)
                disk[i] = (1.0, 0.72, 0.42)
            lp["uBhNdc"].write(ndc.tobytes())
            lp["uBhRs"].write(rs_a.tobytes())
            lp["uBhDist"].write(dist_a.tobytes())
            lp["uBehind"].write(behind.tobytes())
            lp["uBhDiskColor"].write(disk.tobytes())
            self.vao_lens.render(moderngl.TRIANGLES)
            source = self.tex_lens

        # ---------------------------------------------------- tonemap
        self.output.use()
        ctx.viewport = (0, 0, self.width, self.height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        source.use(0)
        self.prog_tone["uScene"].value = 0
        self.vao_tone.render(moderngl.TRIANGLES)

        # ---------------------------------------------------- hud
        if hud_texture is not None:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            hud_texture.use(0)
            self.prog_hud["uHud"].value = 0
            self.vao_hud.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)
