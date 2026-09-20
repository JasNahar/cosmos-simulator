"""
The Simulation object -- ties bodies, gravity and the integrator together.

It owns the clock. The renderer asks it for positions; it never asks the
renderer for anything. Keeping that one-directional is what will let you
later run the physics on a background thread, or headless for a batch run,
without touching the graphics code at all.
"""

from __future__ import annotations

import numpy as np

from . import constants as K
from . import gravity
from .bodies import BodyStore, KIND_BLACKHOLE
from .integrators import INTEGRATORS


class Simulation:
    def __init__(self,
                 store: BodyStore,
                 dt: float = 3600.0,
                 integrator: str = "verlet",
                 relativistic: bool = False,
                 collisions: bool = True):
        self.store = store
        self.dt = dt                       # seconds of simulated time per step
        self.integrator_name = integrator
        self.integrator = INTEGRATORS[integrator]
        self.relativistic = relativistic
        self.collisions = collisions

        self.time = 0.0                    # simulated seconds since start
        self.steps = 0
        self._accum = 0.0
        self._drift_cache = 0.0
        self._drift_step = 0
        self._drift_every = 40      # recompute every N steps, not every frame
        self.events: list[str] = []        # human-readable log of what happened

        # Index of the most massive body -- used for the GR correction and
        # for reporting orbital elements.
        self._dominant = int(np.argmax(store.mass))

        # Use chunked gravity above this many bodies to keep memory sane.
        self._chunk_threshold = 3000

        self._refresh_source_split()
        self.store.acc = self._forces(self.store.pos)
        self.energy0 = self.store.total_energy()

    # -- forces -------------------------------------------------------------

    def _refresh_source_split(self) -> None:
        """Decide which bodies actually exert gravity.

        Bodies given zero mass (asteroids, disk particles, probes) are test
        particles: they feel gravity but don't create it. If most of the
        scene is test particles we can use the far cheaper N*M path.
        """
        s = self.store
        self._src = np.flatnonzero((s.mass > 0.0) & s.alive)
        self._use_sources = self._src.size * 4 < s.n

    def _forces(self, pos: np.ndarray) -> np.ndarray:
        s = self.store
        if self._use_sources:
            src = self._src
            a = gravity.accelerations_from_sources(
                pos, pos[src], s.mass[src], s.softening[src]
            )
        elif s.n > self._chunk_threshold:
            a = gravity.accelerations_chunked(pos, s.mass, s.softening, s.alive)
        else:
            a = gravity.accelerations(pos, s.mass, s.softening, s.alive)
        if self.relativistic:
            a = a + gravity.relativistic_correction(
                pos, s.vel, s.mass * s.alive, self._dominant
            )
        return a

    # -- stepping -----------------------------------------------------------

    def step(self, dt: float | None = None) -> None:
        """Advance one timestep."""
        s = self.store
        h = self.dt if dt is None else dt
        s.pos, s.vel, s.acc = self.integrator(s.pos, s.vel, s.acc, h, self._forces)
        self.time += h
        self.steps += 1
        if self.collisions:
            self._handle_collisions()

    def advance(self, seconds: float, max_substeps: int = 64) -> int:
        """Advance by a wall-clock-driven amount of simulated time.

        The accumulator matters. The naive `n = int(seconds / self.dt)`
        throws away the remainder every frame -- and if a frame asks for less
        than one full timestep, it advances by nothing at all, at any
        framerate, with no error message.

        Banking the leftover time fixes that: the simulated rate is exactly
        what was asked for, while every individual step stays the same fixed
        size, which is what the symplectic integrator requires.
        """
        self._accum += seconds
        n = int(self._accum / self.dt)
        if n > max_substeps:
            # Machine can't keep up: take what we can and drop the backlog,
            # or each frame falls further behind until the program locks.
            n = max_substeps
            self._accum = 0.0
        else:
            self._accum -= n * self.dt
        for _ in range(n):
            self.step()
        return n

    # -- collisions ---------------------------------------------------------

    def _handle_collisions(self) -> None:
        """Perfectly inelastic merging, plus absorption by black holes.

        When two bodies overlap we merge them into one, conserving mass and
        momentum. Real collisions shatter things, but merging is the standard
        cheap model and it keeps the simulation stable -- without it, a close
        pass can fling a body away at absurd speed.
        """
        s = self.store
        idx = np.flatnonzero(s.alive)
        if idx.size < 2:
            return

        # Only test against the massive bodies. Two asteroids meeting in a
        # belt is astronomically rare over the timescales we simulate; an
        # asteroid meeting Jupiter is not. This keeps the check O(N*M).
        src = np.flatnonzero((s.mass > 0.0) & s.alive)
        if src.size == 0:
            return

        def _reach(ix):
            return np.where(
                s.kind[ix] == KIND_BLACKHOLE,
                np.maximum(K.schwarzschild_radius(np.maximum(s.mass[ix], 1.0)),
                           s.radius[ix]),
                s.radius[ix],
            )

        d = s.pos[src][None, :, :] - s.pos[idx][:, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        touch = (_reach(idx)[:, None] + _reach(src)[None, :]) ** 2
        r2[r2 < 1e-9] = np.inf                      # a body vs itself

        hits = np.argwhere(r2 < touch)
        merged: set[int] = set()
        for a_l, b_l in hits:
            a, b = int(idx[a_l]), int(src[b_l])
            if a == b or a in merged or b in merged:
                continue
            # Keep the heavier one; it absorbs the lighter.
            keep, gone = (a, b) if s.mass[a] >= s.mass[b] else (b, a)
            total = s.mass[keep] + s.mass[gone]
            s.vel[keep] = (s.vel[keep] * s.mass[keep] + s.vel[gone] * s.mass[gone]) / total
            s.pos[keep] = (s.pos[keep] * s.mass[keep] + s.pos[gone] * s.mass[gone]) / total
            if s.kind[keep] != KIND_BLACKHOLE:
                # Volume adds, so radius grows as the cube root of mass.
                s.radius[keep] *= (total / s.mass[keep]) ** (1.0 / 3.0)
            s.mass[keep] = total
            s.alive[gone] = False
            s.mass[gone] = 0.0
            merged.add(gone)
            verb = "swallowed" if s.kind[keep] == KIND_BLACKHOLE else "absorbed"
            self.events.append(f"{s.names[keep]} {verb} {s.names[gone]}")

        if merged:
            s.softening = np.maximum(s.radius, 1.0)
            self._refresh_source_split()
            s.acc = self._forces(s.pos)

    # -- diagnostics --------------------------------------------------------

    @property
    def energy_drift(self) -> float:
        """Cached read of the energy drift.

        A HUD readout must never be expensive. Recomputing this every frame
        is what made the framerate collapse -- a diagnostic that costs more
        than the thing it diagnoses is worse than no diagnostic at all.
        """
        if self.steps - self._drift_step >= self._drift_every:
            self._drift_step = self.steps
            self._drift_cache = self._energy_drift_now()
        return self._drift_cache

    def _energy_drift_now(self) -> float:
        """Relative change in total energy since t=0. The real computation.

        This is the single best health check on the simulation. With a
        symplectic integrator and a sensible dt it should oscillate around
        something like 1e-9 and never trend. If it climbs steadily, your
        timestep is too large for the tightest orbit in the scene.
        """
        e = self.store.total_energy()
        return abs((e - self.energy0) / self.energy0) if self.energy0 else 0.0
    def set_dt(self, dt: float) -> None:
        self.dt = float(dt)

    def set_integrator(self, name: str) -> None:
        self.integrator_name = name
        self.integrator = INTEGRATORS[name]
        self.store.acc = self._forces(self.store.pos)
