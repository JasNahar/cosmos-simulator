"""
The body store.

Design note -- "SoA" vs "AoS":
A naive approach is a list of Body objects, each with .position, .velocity.
That's an "array of structs" and it is *slow* in Python: computing gravity
means a Python-level loop over N^2 pairs.

Instead we use a "struct of arrays": one (N,3) numpy array for all positions,
one for all velocities, one (N,) array of masses. Now the entire N-body force
calculation is a handful of numpy operations running in C. For N=2000 that is
the difference between 0.5 fps and 60 fps.

`Body` below is a convenience/description object used when *building* a scene.
`BodyStore` is what the simulation actually runs on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import constants as K


# Body kinds drive how the renderer treats them.
KIND_STAR = 0        # emits light
KIND_PLANET = 1      # lit by stars
KIND_ASTEROID = 2    # lit, drawn small, no trail by default
KIND_BLACKHOLE = 3   # emits nothing, drawn as horizon + photon ring



class Body:
    """Human-friendly description of one object, used at scene-build time."""
    def __init__(self,name,mass,radius,position,velocity,kind=KIND_PLANET,color=(1.0,1.0,1.0),luminosity=0.0,trail=True):
        self.name=name
        self.mass=mass
        self.radius=radius
        self.position=position
        self.velocity=velocity
        self.kind=kind
        self.color=color
        self.luminosity=luminosity
        self.trail=trail
        self.position = np.asarray(position, dtype=np.float64).reshape(3)   # converts position and velocity into vector in nparray for fater processing
        self.velocity = np.asarray(velocity, dtype=np.float64).reshape(3)


class BodyStore:
    """Struct-of-arrays container for the whole simulation."""

    def __init__(self, bodies):
        n = len(bodies)
        self.n = n
        self.names = [b.name for b in bodies]
        self.mass = np.array([b.mass for b in bodies], dtype=np.float64)
        self.radius = np.array([b.radius for b in bodies], dtype=np.float64)
        self.pos = np.array([b.position for b in bodies], dtype=np.float64)
        self.vel = np.array([b.velocity for b in bodies], dtype=np.float64)
        self.acc = np.zeros((n, 3), dtype=np.float64)
        self.kind = np.array([b.kind for b in bodies], dtype=np.int32)
        self.color = np.array([b.color for b in bodies], dtype=np.float32)
        self.luminosity = np.array([b.luminosity for b in bodies], dtype=np.float64)
        self.trail_flag = np.array([b.trail for b in bodies], dtype=bool)
        self.alive = np.ones(n, dtype=bool)

        # Softening length: gravity uses 1/r^2, which explodes as r -> 0.
        # Two point masses passing close would get an infinite kick and the
        # simulation would blow up. We replace r^2 with (r^2 + eps^2), which
        # is exact at large distances and finite at zero. Physically it says
        # "these are spheres, not points". Setting it to the body radius is a
        # reasonable default.
        self.softening = np.maximum(self.radius, 1.0).copy()

    # -- convenience views --------------------------------------------------
    def total_mass(self):
        return self.mass.sum()
    
    def index(self, name: str) -> int:
        return self.names.index(name)

    def center_of_mass(self) -> np.ndarray:
        m = self.mass[self.alive]
        return (self.pos[self.alive] * m[:, None]).sum(axis=0) / m.sum()

    def com_velocity(self) -> np.ndarray:
        m = self.mass[self.alive]
        return (self.vel[self.alive] * m[:, None]).sum(axis=0) / m.sum()

    def zero_momentum(self) -> None:
        """Remove net drift so the whole system doesn't slide off screen.

        Momentum is conserved, so whatever net velocity you start with, you
        keep forever. Subtracting the centre-of-mass velocity puts us in the
        frame where the system as a whole stands still.
        """
        self.vel -= self.com_velocity()
        self.pos -= self.center_of_mass()

    # -- diagnostics --------------------------------------------------------

    def _massive(self) -> np.ndarray:
        """Indices of bodies that carry mass.

        Test particles have mass exactly 0, so they contribute exactly 0 to
        both kinetic and potential energy. Excluding them is not an
        approximation -- it is the same number, computed without building a
        1510x1510 matrix and multiplying two million pairs by zero.
        """
        return np.flatnonzero((self.mass > 0.0) & self.alive)

    def kinetic_energy(self) -> float:
        i = self._massive()
        v2 = np.einsum("ij,ij->i", self.vel[i], self.vel[i])
        return float(0.5 * np.sum(self.mass[i] * v2))

    def potential_energy(self) -> float:
        """U = -G * sum over unique pairs of m_i m_j / r_ij."""
        i = self._massive()
        if i.size < 2:
            return 0.0
        p, m, soft = self.pos[i], self.mass[i], self.softening[i]
        d = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        rmin2 = (soft[:, None] + soft[None, :]) ** 2
        r = np.sqrt(np.maximum(r2, rmin2))    # same force law as gravity.py
        mm = m[:, None] * m[None, :]
        iu = np.triu_indices(i.size, k=1)
        return float(-K.G * np.sum(mm[iu] / r[iu]))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def angular_momentum(self) -> np.ndarray:
        return np.sum(
            np.cross(self.pos, self.vel) * (self.mass * self.alive)[:, None], axis=0
        )
