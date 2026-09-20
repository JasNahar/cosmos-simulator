"""
Black hole scenes.

Two builders:

  sgr_a()  -- a supermassive black hole modelled on Sagittarius A*, the
              4.3-million-solar-mass object at the centre of our galaxy,
              with an accretion disk and a few "S-stars" on wildly
              eccentric orbits. Turn on relativistic mode and watch their
              orbits precess -- this is a real, measured effect (the GRAVITY
              collaboration detected S2's precession in 2020).

  binary() -- two 30-solar-mass black holes spiralling around each other
              with a circumbinary disk. This is the LIGO-style system.

A note on scale. A black hole is *small*. Sgr A* has four million times the
Sun's mass packed into an event horizon 0.085 AU across -- about a fifth of
Mercury's orbit. A stellar-mass hole's horizon is the size of a city. What
makes them dramatic isn't size, it's the steepness of the gravity well right
next to them.

Right now the renderer draws them as a black disk with a bright photon
ring at 1.5 r_s -- geometrically correct, but not yet *lensed*. Real
gravitational lensing (bending the background starfield around the hole)
is a screen-space raymarching shader, and it's the natural next module.
"""

from __future__ import annotations

import numpy as np

from ..physics import constants as K
from ..physics.bodies import (Body, BodyStore, KIND_ASTEROID, KIND_BLACKHOLE,
                              KIND_STAR)
from ..physics.kepler import circular_orbit, elements_to_state


def _disk(central_mass: float, central_pos, central_vel,
          r_inner: float, r_outer: float, n: int, rng,
          thickness: float = 0.02, hot: bool = True) -> list[Body]:
    """A ring of test particles on near-circular orbits.

    Real accretion disks are held up by pressure and drained by viscosity;
    this is the ballistic version -- pure gravity, no gas physics. It still
    reproduces the right shape and the right differential rotation (inner
    material orbits much faster, which is what shears a disk into spirals).
    """
    mu = K.G * central_mass
    out = []
    # Surface density falling as r^-1 gives a natural-looking disk.
    u = rng.random(n)
    r = r_inner * (r_outer / r_inner) ** u
    phase = rng.uniform(0, 360, n)
    tilt = np.degrees(rng.normal(0.0, thickness, n))
    for j in range(n):
        p, v = circular_orbit(r[j], mu, tilt[j], phase[j])
        # Hotter (bluer, brighter) closer in -- disk temperature goes as
        # roughly r^-3/4, which is the standard thin-disk result.
        t = (r_inner / r[j]) ** 0.75
        color = ((0.6 + 0.4 * t), (0.4 + 0.3 * t), (0.15 + 0.75 * t)) if hot else (0.6, 0.6, 0.6)
        out.append(Body(
            name=f"disk{j}", mass=0.0, radius=r_inner * 0.012,
            position=np.asarray(central_pos) + p,
            velocity=np.asarray(central_vel) + v,
            kind=KIND_ASTEROID, color=color, trail=False,
        ))
    return out


def sgr_a(disk_particles: int = 2500, s_stars: int = 6, seed: int = 3) -> BodyStore:
    rng = np.random.default_rng(seed)
    M = 4.297e6 * K.M_SUN
    rs = K.schwarzschild_radius(M)

    bodies = [Body(
        name="Sgr A*", mass=M, radius=rs,
        position=np.zeros(3), velocity=np.zeros(3),
        kind=KIND_BLACKHOLE, color=(0.0, 0.0, 0.0), trail=False,
    )]

    # Inner edge near the ISCO (3 r_s): inside that radius no stable circular
    # orbit exists at all, so a real accretion disk simply stops there. We
    # start a little further out at 8 r_s to keep the timestep reasonable.
    bodies += _disk(M, np.zeros(3), np.zeros(3),
                    r_inner=8 * rs, r_outer=80 * rs,
                    n=disk_particles, rng=rng, thickness=0.025)

    # S-stars: real ones orbit at 100-1000 AU with e up to 0.97. Placed a
    # little further out here so a single timestep can resolve both them
    # and the inner disk.
    palette = [(0.75, 0.82, 1.0), (0.9, 0.9, 1.0), (1.0, 0.9, 0.75)]
    for j in range(s_stars):
        a = rng.uniform(2000, 7000) * rs
        e = rng.uniform(0.55, 0.92)
        mu = K.G * M
        p, v = elements_to_state(a, e, rng.uniform(0, 180), rng.uniform(0, 360),
                                 rng.uniform(0, 360), rng.uniform(0, 360), mu)
        mass = rng.uniform(8, 20) * K.M_SUN
        bodies.append(Body(
            name=f"S{j+1}", mass=mass, radius=4.0 * K.R_SUN,
            position=p, velocity=v, kind=KIND_STAR,
            color=palette[j % len(palette)],
            luminosity=2000 * K.L_SUN, trail=True,
        ))

    store = BodyStore(bodies)
    # The event horizon is the capture radius: anything crossing it is gone.
    store.softening[0] = rs
    store.zero_momentum()
    return store


def binary(disk_particles: int = 2000, seed: int = 11) -> BodyStore:
    rng = np.random.default_rng(seed)
    m1, m2 = 36.0 * K.M_SUN, 29.0 * K.M_SUN
    sep = 8.0e8                     # metres between them
    total = m1 + m2

    # Place both on a circular mutual orbit about their common barycentre.
    # Each orbits at a distance inversely proportional to its own mass.
    v_rel = np.sqrt(K.G * total / sep)
    r1 = sep * m2 / total
    r2 = sep * m1 / total
    v1 = v_rel * m2 / total
    v2 = v_rel * m1 / total

    bodies = [
        Body("BH-A", m1, K.schwarzschild_radius(m1),
             [-r1, 0, 0], [0, -v1, 0], KIND_BLACKHOLE, (0, 0, 0), trail=True),
        Body("BH-B", m2, K.schwarzschild_radius(m2),
             [r2, 0, 0], [0, v2, 0], KIND_BLACKHOLE, (0, 0, 0), trail=True),
    ]

    # A circumbinary disk -- it has to start well outside the pair, because
    # anything closer than ~2x the separation gets ejected within a few
    # orbits. That clearing is a real and observed phenomenon.
    bodies += _disk(total, np.zeros(3), np.zeros(3),
                    r_inner=2.6 * sep, r_outer=14 * sep,
                    n=disk_particles, rng=rng, thickness=0.05)

    store = BodyStore(bodies)
    store.zero_momentum()
    return store


METADATA = {
    "sgr_a": {
        "name": "Sagittarius A*",
        "suggested_dt": 40.0,
        "focus": "Sgr A*",
        "start_distance": 45 * K.schwarzschild_radius(4.297e6 * K.M_SUN),
        "relativistic": True,
        "notes": "Relativistic mode on: watch the S-star orbits precess.",
    },
    "binary": {
        "name": "Binary Black Holes",
        "suggested_dt": 1.0,
        "focus": "BH-A",
        "start_distance": 2.0e10,
        "relativistic": False,
        "notes": "Watch the pair carve a gap in the circumbinary disk.",
    },
}
