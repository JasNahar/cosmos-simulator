"""
The real solar system, built from real J2000 orbital elements.

The numbers below are NASA/JPL's mean elements for the epoch J2000
(1 January 2000, 12:00 TT). Feeding them through kepler.elements_to_state
puts every planet exactly where it was on that date -- correct distances,
correct eccentricities, correct inclinations, correct phase around its orbit.

From there the N-body integrator takes over, and everything after t=0 is
genuine gravitational simulation, not a scripted animation. Jupiter really
does perturb Saturn. The asteroid belt really does develop Kirkwood gaps.
"""

from __future__ import annotations

import numpy as np

from ..physics import constants as K
from ..physics.bodies import (Body, BodyStore, KIND_ASTEROID, KIND_PLANET,
                              KIND_STAR)
from ..physics.kepler import elements_to_state

# name, mass(kg), radius(m), a(AU), e, i, Omega, varpi(long. of perihelion),
# L(mean longitude), colour
#
# Note the last two: catalogues publish the *longitude of perihelion*
# (varpi = Omega + omega) and *mean longitude* (L = varpi + M), not omega and
# M directly. We convert below.
PLANETS = [
    ("Mercury", 3.3011e23, 2.4397e6,
     0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350,
     (0.62, 0.58, 0.55)),
    ("Venus", 4.8675e24, 6.0518e6,
     0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950,
     (0.94, 0.83, 0.60)),
    ("Earth", 5.97217e24, 6.371e6,
     1.00000261, 0.01671123, -0.00001531, 0.0, 102.93768193, 100.46457166,
     (0.24, 0.45, 0.78)),
    ("Mars", 6.4171e23, 3.3895e6,
     1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959, -4.55343205,
     (0.79, 0.38, 0.24)),
    ("Jupiter", 1.89813e27, 6.9911e7,
     5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644051,
     (0.83, 0.72, 0.58)),
    ("Saturn", 5.6832e26, 5.8232e7,
     9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 49.95424423,
     (0.89, 0.81, 0.62)),
    ("Uranus", 8.6811e25, 2.5362e7,
     19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 313.23810451,
     (0.56, 0.79, 0.82)),
    ("Neptune", 1.02409e26, 2.4622e7,
     30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, -55.12002969,
     (0.28, 0.44, 0.82)),
]


def build(asteroids: int = 1500,
          include_moon: bool = True,
          visual_scale: float = 1.0,
          seed: int = 7) -> BodyStore:
    """Assemble the solar system.

    `visual_scale` multiplies the *drawn* radius of the planets. At true
    scale the Sun is 0.5% of the width of Mercury's orbit and Earth is
    invisible from anywhere useful -- which is honest but no fun to fly
    around. The physics always uses true radii; only the visuals scale.
    """
    rng = np.random.default_rng(seed)
    bodies: list[Body] = []

    sun = Body(
        name="Sun",
        mass=K.M_SUN,
        radius=K.R_SUN * visual_scale,
        position=np.zeros(3),
        velocity=np.zeros(3),
        kind=KIND_STAR,
        color=(1.0, 0.94, 0.80),
        luminosity=K.L_SUN,
        trail=False,
    )
    bodies.append(sun)

    for (name, mass, radius, a_au, e, inc, Omega, varpi, L, color) in PLANETS:
        a = a_au * K.AU
        omega = varpi - Omega          # argument of perihelion
        M = L - varpi                  # mean anomaly at epoch
        mu = K.G * (K.M_SUN + mass)
        pos, vel = elements_to_state(a, e, inc, Omega, omega, M, mu)
        bodies.append(Body(
            name=name, mass=mass, radius=radius * visual_scale,
            position=pos, velocity=vel,
            kind=KIND_PLANET, color=color, trail=True,
        ))

    if include_moon:
        earth = next(b for b in bodies if b.name == "Earth")
        mu_e = K.G * (earth.mass + 7.342e22)
        # The Moon's orbit is inclined ~5.1 deg to the ecliptic, a=384,400 km.
        m_pos, m_vel = elements_to_state(3.844e8, 0.0549, 5.145, 0.0, 0.0, 135.0, mu_e)
        bodies.append(Body(
            name="Moon", mass=7.342e22, radius=1.7374e6 * visual_scale,
            position=earth.position + m_pos, velocity=earth.velocity + m_vel,
            kind=KIND_PLANET, color=(0.62, 0.60, 0.58), trail=True,
        ))

    # --- the asteroid belt --------------------------------------------------
    # Mass zero => test particles (see gravity.accelerations_from_sources).
    # Real belt: 2.06-3.27 AU, eccentricities peaking around 0.15,
    # inclinations up to ~20 deg with most under 10.
    if asteroids > 0:
        a_ast = rng.uniform(2.06, 3.27, asteroids) * K.AU
        e_ast = np.abs(rng.normal(0.0, 0.10, asteroids)).clip(0, 0.35)
        i_ast = np.abs(rng.normal(0.0, 7.0, asteroids)).clip(0, 25.0)
        Om_ast = rng.uniform(0, 360, asteroids)
        w_ast = rng.uniform(0, 360, asteroids)
        M_ast = rng.uniform(0, 360, asteroids)
        mu = K.G * K.M_SUN
        # Slightly reddish-grey, varying: C-type (dark) vs S-type (lighter).
        shade = rng.uniform(0.25, 0.75, asteroids)
        for j in range(asteroids):
            p, v = elements_to_state(a_ast[j], e_ast[j], i_ast[j],
                                     Om_ast[j], w_ast[j], M_ast[j], mu)
            s = shade[j]
            bodies.append(Body(
                name=f"ast{j}", mass=0.0, radius=6.0e5 * visual_scale,
                position=p, velocity=v, kind=KIND_ASTEROID,
                color=(s * 0.9, s * 0.82, s * 0.72), trail=False,
            ))

    store = BodyStore(bodies)
    store.zero_momentum()
    return store


METADATA = {
    "name": "Solar System (J2000)",
    "suggested_dt": 3600.0,          # 1 hour -- resolves the Moon comfortably
    "suggested_scale": 1.0,
    "focus": "Sun",
    "start_distance": 6.0 * K.AU,
    "notes": "Real J2000 elements. Try focusing Jupiter and watching the belt.",
}
