"""
Gravity.

The core of the whole simulator is one line of physics:

    a_i = G * sum_j!=i  m_j * (r_j - r_i) / |r_j - r_i|^3

Every body is pulled by every other body. That's N*(N-1) interactions --
"direct summation" or "particle-particle". It is O(N^2), which sounds bad,
but numpy does the whole thing in compiled C, so a few thousand bodies runs
comfortably in real time. (Beyond ~10k you want a Barnes-Hut octree, which
is O(N log N). That's a later module -- the interface below won't change.)

Two refinements are included:

1. Softening (see bodies.py) so close encounters don't produce infinite forces.
2. An optional first-order relativistic correction, which is what makes
   Mercury's orbit precess and makes orbits near a black hole behave
   correctly instead of like idealised Kepler ellipses.

A NOTE ON SOFTENING, because the obvious choice is wrong.

The textbook trick is Plummer softening: replace r^2 with (r^2 + eps^2).
It is smooth and easy, but it perturbs the force at EVERY distance, not just
close in. With eps set to the Sun's radius, Mercury's orbit feels a relative
force error of (R_sun/a)^2 ~ 1.4e-4 -- which drives a spurious perihelion
precession of about -63,000 arcsec/century, roughly 1500x larger than the
relativistic effect we are trying to measure. It silently destroys the
headline result.

What we use instead is a floor on the separation:

    inv_r3 = max(r^2, r_contact^2) ** -1.5

This is EXACTLY Newtonian everywhere outside contact -- zero error at
Mercury, zero error anywhere that matters -- and merely caps the force when
two bodies overlap, which is the only place the singularity is a problem.
"""

from __future__ import annotations

import numpy as np

from . import constants as K


# Above this many bodies the (N,N,3) temporary stops fitting comfortably in
# cache and the naive broadcast becomes memory-bound -- measurably slower than
# a chunked loop despite doing identical arithmetic.
CHUNK_ABOVE = 1024


def accelerations(pos, mass, softening, alive=None):
    """Newtonian gravitational acceleration on every body. Shape (N,3).

    One Python loop, over i. The inner loop over j is gone -- every numpy
    line below handles ALL the other bodies in a single call, which is what
    makes numpy worth using.
    """
    n = pos.shape[0]
    m = mass if alive is None else mass * alive
    acc = np.zeros_like(pos)

    for i in range(n):
        # vectors FROM body i TO every body.  (N,3)
        d = pos - pos[i]

        # squared distance from i to every body.  (N,)
        r2 = np.einsum("ij,ij->i", d, d)

        # separation floor, so overlapping bodies don't produce infinite force
        r2 = np.maximum(r2, (softening[i] + softening) ** 2)

        # body i must not pull on itself. inf here makes the next line give 0.
        r2[i] = np.inf

        # strength of each pull: G*m_j / r^3.  (N,)
        w = m * r2 ** -1.5

        # multiply each direction by its strength, add them all up
        acc[i] = K.G * np.einsum("i,ij->j", w, d)

    return acc

def accelerations_chunked(pos, mass, softening, alive=None, chunk=512):
    """Same result as `accelerations` but with bounded memory.

    Use when N is large enough that the (N,N,3) temporary won't fit.
    Trades a little speed for a lot of headroom.
    """
    n = pos.shape[0]
    m = mass if alive is None else mass * alive
    acc = np.zeros_like(pos)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = pos[None, :, :] - pos[start:stop, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        rmin2 = (softening[start:stop, None] + softening[None, :]) ** 2
        inv_r3 = np.maximum(r2, rmin2) ** -1.5
        for local, glob in enumerate(range(start, stop)):
            inv_r3[local, glob] = 0.0
        acc[start:stop] = K.G * np.einsum("ij,ijk->ik", m * inv_r3, d)
    return acc


def accelerations_from_sources(pos: np.ndarray,
                               src_pos: np.ndarray,
                               src_mass: np.ndarray,
                               src_soft: np.ndarray) -> np.ndarray:
    """Acceleration on N bodies caused by only M gravitating "sources".

    This is the big optimisation for asteroid belts and accretion disks.
    An asteroid weighs ~1e-12 of the Earth; its pull on anything else is
    genuinely negligible, so we can treat it as a *test particle* -- it
    feels gravity but does not exert it.

    That turns the cost from O(N^2) into O(N*M). With 10 massive bodies and
    5000 asteroids, that is 50,000 interactions instead of 25,000,000.
    The asteroids still move on fully correct trajectories, including
    resonances with Jupiter -- they just don't tug on each other, which is
    true to about one part in a billion anyway.
    """
    d = src_pos[None, :, :] - pos[:, None, :]           # (N,M,3)
    r2 = np.einsum("ijk,ijk->ij", d, d)
    inv_r3 = np.maximum(r2, src_soft[None, :] ** 2) ** -1.5
    inv_r3[r2 < 1e-9] = 0.0                             # a source ignores itself
    return K.G * np.einsum("ij,ijk->ik", src_mass[None, :] * inv_r3, d)


def relativistic_correction(pos: np.ndarray,
                            vel: np.ndarray,
                            mass: np.ndarray,
                            dominant: int) -> np.ndarray:
    """First post-Newtonian correction from one dominant mass.

    Newton says a bound orbit is a closed ellipse that never moves. General
    relativity says the ellipse slowly rotates. To leading order the extra
    acceleration is:

        a_GR = -(G*M/r^2) * 3*h^2 / (c^2 * r^2)  * rhat

    where h = |r x v| is the specific angular momentum. It is a tiny term
    (~1e-8 of Newtonian gravity for Mercury) but it accumulates: it produces
    Mercury's famous 43 arcseconds per century of perihelion precession, the
    first real evidence for GR.

    Near a black hole this term is no longer tiny, and it is what makes
    close orbits precess dramatically and eventually plunge.
    """
    M = mass[dominant]
    d = pos - pos[dominant]                # vector from the massive body
    v = vel - vel[dominant]
    r2 = np.einsum("ij,ij->i", d, d)
    # The dominant body's own separation from itself is zero. Guard BOTH r2
    # and r, or the 0 * inf below produces NaN that then poisons every
    # subsequent timestep.
    r2 = np.maximum(r2, 1.0)
    r = np.sqrt(r2)

    h = np.cross(d, v)
    h2 = np.einsum("ij,ij->i", h, h)

    factor = -(K.G * M) * 3.0 * h2 / (K.C ** 2 * r2 * r2 * r)
    corr = factor[:, None] * d
    corr[dominant] = 0.0
    return corr
