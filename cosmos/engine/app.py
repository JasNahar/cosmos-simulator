"""
The interactive application: window, input, main loop.

Loop structure per frame:

    dt_real  = wall-clock seconds since the last frame
    dt_sim   = dt_real * time_scale        <- simulated seconds to advance
    sim.advance(dt_sim)                    <- fixed-size physics substeps
    renderer.render(...)

The physics uses a FIXED timestep internally, regardless of framerate. This
matters: a variable timestep destroys the conservation properties of a
symplectic integrator, which is the entire reason for using one. The
framerate decides how many fixed steps to take, never how large they are.
"""

from __future__ import annotations

import time

import numpy as np
import pygame

import moderngl

from ..physics import constants as K
from ..physics.bodies import KIND_ASTEROID, KIND_BLACKHOLE, KIND_STAR
from ..physics.simulation import Simulation
from ..scenes import load
from .camera import Camera
from .hud import Hud, fmt_distance, fmt_time
from .renderer import Renderer
from .trails import Trails

INTEGRATOR_CYCLE = ["verlet", "yoshida4", "euler"]

HELP = [
    "WASD/Space/Shift fly   Mouse look   Tab release mouse   Scroll speed",
    "F/V cycle focus   C follow on/off   ,/. time scale   P pause   R reset view",
    "T trails   Y starfield   [/] planet size   G relativity   I integrator   H help",
]


class App:
    def __init__(self, scene="solar", width=1440, height=810,
                 asteroids=None, vsync=True):
        self.width, self.height = width, height
        self._hud_cool = 0.0
        self._hud_tex = None

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.set_mode((width, height),
                                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption(f"Cosmos -- {scene}")
        self.clock = pygame.time.Clock()

        self.ctx = moderngl.create_context()

        # ---- scene -------------------------------------------------------
        kwargs = {}
        if asteroids is not None:
            kwargs["asteroids" if scene == "solar" else "disk_particles"] = asteroids
        self.store, self.meta = load(scene, **kwargs)
        self.scene_name = scene
        self.sim = Simulation(
            self.store,
            dt=self.meta.get("suggested_dt", 3600.0),
            integrator="verlet",
            relativistic=self.meta.get("relativistic", False),
        )

        self.renderer = Renderer(self.ctx, width, height, max_bodies=self.store.n)
        self.hud = Hud(self.ctx, width, height)

        # ---- view --------------------------------------------------------
        self.visual_scale = 900.0 if scene == "solar" else 1.0
        self.star_scale = 22.0 if scene == "solar" else 1.0
        self.camera = Camera(fov_deg=58.0, near=1.0, far=3.0e14)

        # Bodies worth following: everything that isn't an anonymous rock.
        self.focus_list = [i for i in range(self.store.n)
                           if self.store.kind[i] != KIND_ASTEROID]
        fname = self.meta.get("focus")
        self.focus_i = (self.focus_list.index(self.store.index(fname))
                        if fname in self.store.names else 0)
        self.follow = True
        self.reset_view()

        tracked = [i for i in self.focus_list if self.store.trail_flag[i]][:16]
        self.trails = Trails(tracked, capacity=1800)

        # ---- controls ----------------------------------------------------
        self.time_scale = self.meta.get("suggested_dt", 3600.0) * 40.0
        self.paused = False
        self.speed = self.meta.get("start_distance", K.AU) * 0.35   # m/s of flight
        self.grabbed = True
        self.show_help = True
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        pygame.mouse.get_rel()

        self.running = True
        self._fps = 60.0
        self._phys_ms = 0.0
        self._draw_ms = 0.0

    # -------------------------------------------------------------- helpers
    @property
    def focus_body(self) -> int:
        return self.focus_list[self.focus_i]

    def reset_view(self):
        b = self.focus_body
        r = max(self.store.radius[b] * self.visual_scale, 1.0)
        d = max(self.meta.get("start_distance", 5 * K.AU), r * 6.0)
        self.camera.position = self.store.pos[b] + np.array([0.3 * d, 0.35 * d, 0.9 * d])
        self.camera.look_at(self.store.pos[b])
        self._follow_offset = self.camera.position - self.store.pos[b]

    # ---------------------------------------------------------------- input
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.width, self.height = e.w, e.h
                self.renderer.resize(e.w, e.h)
                self.hud.resize(e.w, e.h)
            elif e.type == pygame.MOUSEWHEEL:
                self.speed *= 1.25 ** e.y
            elif e.type == pygame.KEYDOWN:
                self.on_key(e.key)

    def on_key(self, key):
        s = self.store
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_TAB:
            self.grabbed = not self.grabbed
            pygame.event.set_grab(self.grabbed)
            pygame.mouse.set_visible(not self.grabbed)
            pygame.mouse.get_rel()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_t:
            self.renderer.show_trails = not self.renderer.show_trails
        elif key == pygame.K_y:
            self.renderer.show_stars = not self.renderer.show_stars
        elif key == pygame.K_c:
            self.follow = not self.follow
            self._follow_offset = self.camera.position - s.pos[self.focus_body]
        elif key == pygame.K_r:
            self.reset_view()
        elif key in (pygame.K_f, pygame.K_v):
            step = 1 if key == pygame.K_f else -1
            self.focus_i = (self.focus_i + step) % len(self.focus_list)
            self.reset_view()
        elif key == pygame.K_COMMA:
            self.time_scale /= 2.0
        elif key == pygame.K_PERIOD:
            self.time_scale *= 2.0
        elif key == pygame.K_LEFTBRACKET:
            self.visual_scale = max(1.0, self.visual_scale / 1.6)
        elif key == pygame.K_RIGHTBRACKET:
            self.visual_scale *= 1.6
        elif key == pygame.K_g:
            self.sim.relativistic = not self.sim.relativistic
        elif key == pygame.K_i:
            nxt = INTEGRATOR_CYCLE[
                (INTEGRATOR_CYCLE.index(self.sim.integrator_name) + 1)
                % len(INTEGRATOR_CYCLE)]
            self.sim.set_integrator(nxt)
            self.sim.energy0 = self.store.total_energy()
            self.sim.time = 0.0

    def handle_motion(self, dt_real):
        if self.grabbed:
            mdx, mdy = pygame.mouse.get_rel()
            self.camera.rotate(-mdx * 0.0022, -mdy * 0.0022)

        keys = pygame.key.get_pressed()
        fwd = (keys[pygame.K_w] - keys[pygame.K_s])
        rgt = (keys[pygame.K_d] - keys[pygame.K_a])
        up = (keys[pygame.K_SPACE] - keys[pygame.K_LSHIFT])
        if fwd or rgt or up:
            v = np.array([fwd, rgt, up], dtype=float)
            v /= np.linalg.norm(v)          # no free diagonal speed boost
            boost = 6.0 if keys[pygame.K_LCTRL] else 1.0
            amt = self.speed * boost * dt_real
            self.camera.move_local(v[0] * amt, v[1] * amt, v[2] * amt)
            if self.follow:
                self._follow_offset = (self.camera.position
                                       - self.store.pos[self.focus_body])

    # ----------------------------------------------------------------- loop
    def run(self):
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt_real = min(now - last, 0.1)      # clamp: never chase a stall
            last = now

            self.handle_events()
            self.handle_motion(dt_real)

            t0 = time.perf_counter()
            if not self.paused:
                self.sim.advance(dt_real * self.time_scale, max_substeps=400)
                self.trails.record(self.store.pos)
            self._phys_ms = (time.perf_counter() - t0) * 1000.0

            # Following: keep a constant offset from the tracked body.
            # Without this, watching Earth means watching it leave at 30 km/s.
            if self.follow:
                self.camera.position = (self.store.pos[self.focus_body]
                                        + self._follow_offset)

            t0 = time.perf_counter()
            self._hud_cool -= dt_real
            if self._hud_cool <= 0.0 or self._hud_tex is None:
                self._hud_cool = 0.1        # 10 Hz is faster than you can read
                self._hud_tex = self.hud.draw(
                    self.hud_lines(), HELP if self.show_help else None)
            tex = self._hud_tex
            self.renderer.render(self.store, self.camera, self.trails,
                                 visual_scale=self.visual_scale,
                                 star_scale=self.star_scale,
                                 hud_texture=tex)
            pygame.display.flip()
            self._draw_ms = (time.perf_counter() - t0) * 1000.0

            self.clock.tick(120)
            self._fps = self.clock.get_fps()

        pygame.quit()

    # ------------------------------------------------------------------ hud
    def hud_lines(self):
        s = self.store
        b = self.focus_body
        d = float(np.linalg.norm(self.camera.position - s.pos[b]))
        alive = int(s.alive.sum())
        drift = self.sim.energy_drift

        lines = [
            f"{self.meta.get('name', self.scene_name)}",
            f"t = {fmt_time(self.sim.time)}    dt = {fmt_time(self.sim.dt)}"
            f"    x{self.time_scale:,.0f}{'  [PAUSED]' if self.paused else ''}",
            f"integrator {self.sim.integrator_name}"
            f"   relativity {'ON' if self.sim.relativistic else 'off'}"
            f"   |dE/E| = {drift:.3e}",
            f"bodies {alive:,} / {s.n:,}"
            f"    {self._fps:5.1f} fps"
            f"   physics {self._phys_ms:5.2f} ms   draw {self._draw_ms:5.2f} ms",
            "",
            f"focus  {s.names[b]}"
            f"   {'[following]' if self.follow else '[free]'}",
            f"range  {fmt_distance(d)}",
            f"speed  {fmt_distance(self.speed)}/s"
            f"    planet scale x{self.visual_scale:,.0f}",
        ]
        if s.kind[b] == KIND_BLACKHOLE:
            rs = K.schwarzschild_radius(s.mass[b])
            lines.append(f"r_s    {fmt_distance(rs)}   ({d / rs:,.1f} r_s away)")
        if self.sim.events:
            lines.append("")
            lines.extend(self.sim.events[-3:])
        return lines


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cosmos -- N-body space simulator")
    ap.add_argument("--scene", default="solar",
                    help="solar | sgra | binary")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=810)
    ap.add_argument("--bodies", type=int, default=None,
                    help="number of asteroids / disk particles")
    args = ap.parse_args()
    App(scene=args.scene, width=args.width, height=args.height,
        asteroids=args.bodies).run()


if __name__ == "__main__":
    main()
