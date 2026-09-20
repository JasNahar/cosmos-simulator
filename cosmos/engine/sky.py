"""
Background starfield and blackbody colour.

Stars are not white. A star's colour is set almost entirely by its surface
temperature, because a star radiates very nearly as a blackbody. Betelgeuse
is 3,500 K and orange; Rigel is 12,000 K and blue-white; the Sun is 5,778 K
and, from outside the atmosphere, essentially white.

`blackbody_rgb` converts temperature to displayable colour. `make_starfield`
builds a background sky with a plausible distribution of both.
"""

from __future__ import annotations

import numpy as np


def blackbody_rgb(temp_k: np.ndarray) -> np.ndarray:
    """Approximate sRGB colour of a blackbody at temperature T (kelvin).

    Uses the Tanner Helland polynomial fit to the Planck curve integrated
    against the CIE colour matching functions -- accurate to a few percent
    over 1000-40000 K, which is far better than the eye can judge on a
    point of light.
    """
    t = np.clip(np.asarray(temp_k, dtype=np.float64), 1000.0, 40000.0) / 100.0

    # np.where evaluates BOTH branches, so every expression has to be safe
    # for every input -- hence the clamps inside. Without them you get
    # "invalid value in power" warnings and NaN colours for cool stars.
    t_hot = np.maximum(t - 60.0, 1e-6)

    # --- red ---
    r = np.where(t <= 66.0, 255.0, 329.698727446 * t_hot ** -0.1332047592)

    # --- green ---
    g = np.where(
        t <= 66.0,
        99.4708025861 * np.log(np.maximum(t, 1e-6)) - 161.1195681661,
        288.1221695283 * t_hot ** -0.0755148492,
    )

    # --- blue ---
    b = np.where(
        t >= 66.0, 255.0,
        np.where(t <= 19.0, 0.0,
                 138.5177312231 * np.log(np.maximum(t - 10.0, 1e-6)) - 305.0447927307),
    )

    rgb = np.stack([r, g, b], axis=-1) / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_starfield(n: int = 9000, seed: int = 42):
    """Generate a background sky.

    Returns (directions, colours, sizes) ready to upload to the GPU.

    Two details that make it look right rather than like television static:

    1. Directions are sampled uniformly on the SPHERE, not uniformly in
       (theta, phi). The naive version clumps stars at the poles, which is
       instantly recognisable as wrong.

    2. Brightness follows a power law -- a handful of bright stars and a
       great many faint ones. A uniform distribution of brightness looks
       artificial because the real sky is overwhelmingly faint stars.
    """
    rng = np.random.default_rng(seed)

    # Uniform on the sphere: z uniform in [-1,1], phi uniform in [0,2pi).
    # This is Archimedes' hat-box theorem -- equal areas of a sphere project
    # to equal areas of the enclosing cylinder.
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack([r * np.cos(phi), z, r * np.sin(phi)], axis=1).astype(np.float32)

    # Temperature distribution roughly matching the real stellar population:
    # dominated by cool dwarfs, with a thin tail of hot bright stars.
    u = rng.random(n)
    temp = np.where(u < 0.76, rng.uniform(2600, 5200, n),
           np.where(u < 0.94, rng.uniform(5200, 7500, n),
                              rng.uniform(7500, 26000, n)))
    colors = blackbody_rgb(temp)

    # Power-law apparent brightness.
    sizes = (1.1 + 5.0 * rng.power(0.32, n)).astype(np.float32)

    # The Milky Way: a modest excess of stars near the galactic plane. Purely
    # cosmetic, but its absence is one of the things that makes procedural
    # skies look flat.
    band = np.exp(-((dirs[:, 1] / 0.16) ** 2))
    sizes = (sizes * (0.72 + 0.55 * band)).astype(np.float32)

    return dirs, colors, sizes
