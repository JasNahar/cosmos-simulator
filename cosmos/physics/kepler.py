"""
Orbital elements -> position and velocity.

Real astronomical data comes as six "Keplerian elements", not as x/y/z:

    a      semi-major axis      -- the size of the ellipse
    e      eccentricity         -- how squashed it is (0 = circle, <1 = ellipse)
    i      inclination          -- tilt relative to the reference plane
    Omega  longitude of ascending node -- where it crosses that plane
    omega  argument of periapsis        -- where the closest point sits
    M      mean anomaly                 -- where the body is along the orbit *now*

To place a planet we have to convert those into a Cartesian state vector.
That requires solving Kepler's equation

    M = E - e*sin(E)

for the eccentric anomaly E. It has no closed-form solution, so we use
Newton-Raphson -- it converges in about 4 iterations for planetary
eccentricities.

This is how the simulator gets a solar system that actually matches the
real one instead of a cartoon of evenly spaced circles.
"""

from __future__ import annotations

import numpy as np

from . import constants as K


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 60) -> float:
    """Solve M = E - e*sin(E) for E by Newton-Raphson."""
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    # A good starting guess matters for high eccentricity.
    E = M + e * np.sin(M) if e < 0.8 else np.pi * np.sign(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def elements_to_state(a: float,
                      e: float,
                      i_deg: float,
                      Omega_deg: float,
                      omega_deg: float,
                      M_deg: float,
                      mu: float):
    """Convert Keplerian elements to (position, velocity) in metres and m/s.

    `mu` is the standard gravitational parameter G*(M_central + m_body).
    Returns vectors in the reference frame where the x-y plane is the
    reference plane (for the solar system: the ecliptic).
    """
    i = np.radians(i_deg)
    Om = np.radians(Omega_deg)
    w = np.radians(omega_deg)
    M = np.radians(M_deg)

    E = solve_kepler(M, e)

    # Position in the orbital plane (periapsis along +x)
    x_p = a * (np.cos(E) - e)
    y_p = a * np.sqrt(1.0 - e * e) * np.sin(E)

    # Velocity in the orbital plane, from differentiating the above
    r = a * (1.0 - e * np.cos(E))
    Edot = np.sqrt(mu / a ** 3) / (1.0 - e * np.cos(E))
    vx_p = -a * np.sin(E) * Edot
    vy_p = a * np.sqrt(1.0 - e * e) * np.cos(E) * Edot

    # Rotate: periapsis (w), then inclination (i), then node (Om).
    cw, sw = np.cos(w), np.sin(w)
    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(Om), np.sin(Om)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                ci],
    ])

    pos = R @ np.array([x_p, y_p, 0.0])
    vel = R @ np.array([vx_p, vy_p, 0.0])
    return pos, vel


def circular_orbit(radius: float, mu: float, inclination_deg: float = 0.0,
                   phase_deg: float = 0.0):
    """Shortcut for a circular orbit -- handy for asteroid belts and disks.

    Circular orbital speed comes straight from balancing gravity against
    the centripetal requirement: v = sqrt(mu / r).
    """
    return elements_to_state(radius, 0.0, inclination_deg, 0.0, 0.0, phase_deg, mu)


def state_to_elements(pos: np.ndarray, vel: np.ndarray, mu: float) -> dict:
    """Inverse conversion -- useful for reading out what an orbit *became*.

    Run this on a body after simulating for a while to see how its orbit
    has been perturbed by everything else.
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    h_vec = np.cross(pos, vel)
    h = np.linalg.norm(h_vec)

    energy = v * v / 2.0 - mu / r
    a = -mu / (2.0 * energy)
    e_vec = np.cross(vel, h_vec) / mu - pos / r
    e = np.linalg.norm(e_vec)
    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

    n_vec = np.cross([0, 0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)
    Omega = np.degrees(np.arctan2(n_vec[1], n_vec[0])) % 360.0 if n > 0 else 0.0
    if n > 0 and e > 1e-12:
        omega = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1)))
        if e_vec[2] < 0:
            omega = 360.0 - omega
    else:
        omega = 0.0

    return {
        "a": a, "e": e, "i": i, "Omega": Omega, "omega": omega,
        "period": 2 * np.pi * np.sqrt(abs(a) ** 3 / mu),
        "apoapsis": a * (1 + e), "periapsis": a * (1 - e),
    }
