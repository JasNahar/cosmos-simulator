"""
Physical constants, in SI units.

Everything in this simulator is stored in SI (metres, kilograms, seconds)
using float64. That sounds insane when the Sun weighs 2e30 kg and Neptune
orbits at 4.5e12 m -- but float64 has ~15-16 significant digits, so we can
represent a metre-scale detail on a solar-system-scale distance. Precision
only becomes a problem on the GPU (float32), and we solve that in the
renderer with camera-relative coordinates.

The rule: SI everywhere in physics/, converted only at the rendering edge.
"""

# --- Fundamental -----------------------------------------------------------
G = 6.67430e-11          # gravitational constant, m^3 kg^-1 s^-2
C = 299_792_458.0        # speed of light, m/s
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann, W m^-2 K^-4

# --- Mass ------------------------------------------------------------------
M_SUN = 1.98847e30       # kg
M_EARTH = 5.972168e24    # kg
M_JUPITER = 1.89813e27   # kg

# --- Length ----------------------------------------------------------------
AU = 1.495978707e11      # astronomical unit, m
R_SUN = 6.957e8          # m
R_EARTH = 6.371e6        # m
LY = 9.4607304725808e15  # light year, m
PARSEC = 3.0856775814913673e16  # m

# --- Time ------------------------------------------------------------------
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
YEAR = 365.25 * DAY      # Julian year, s

# --- Luminosity ------------------------------------------------------------
L_SUN = 3.828e26         # W

# --- Derived helpers -------------------------------------------------------


def schwarzschild_radius(mass_kg: float) -> float:
    """Event horizon radius of a non-rotating black hole: r_s = 2GM/c^2.

    For the Sun this is ~2.95 km. For a 10 solar-mass black hole, ~30 km.
    Anything closer than this cannot send light back out.
    """
    return 2.0 * G * mass_kg / (C * C)


def photon_sphere_radius(mass_kg: float) -> float:
    """Radius where light itself orbits in a circle: 1.5 * r_s.

    This is the bright ring you see in black hole images -- photons that
    grazed the hole and looped around before escaping toward you.
    """
    return 1.5 * schwarzschild_radius(mass_kg)


def isco_radius(mass_kg: float) -> float:
    """Innermost Stable Circular Orbit: 3 * r_s for a Schwarzschild hole.

    Inside this radius there are no stable orbits at all -- matter spirals in.
    It sets the inner edge of an accretion disk.
    """
    return 3.0 * schwarzschild_radius(mass_kg)


def hill_radius(m_small: float, m_large: float, separation: float) -> float:
    """Radius of gravitational dominance of a small body orbiting a large one.

    Moons must orbit inside this. Earth's is ~1.5 million km.
    """
    return separation * (m_small / (3.0 * m_large)) ** (1.0 / 3.0)


def escape_velocity(mass_kg: float, radius_m: float) -> float:
    """Speed needed to escape to infinity from a distance `radius_m`."""
    return (2.0 * G * mass_kg / radius_m) ** 0.5


def orbital_period(a: float, m_central: float, m_orbiting: float = 0.0) -> float:
    """Kepler's third law: T = 2*pi * sqrt(a^3 / (G*(M+m)))."""
    import math
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G * (m_central + m_orbiting)))
