"""
Physics verification.

Run:  python -m cosmos.tests.test_physics

These are the checks that tell you the simulator is real rather than
plausible-looking. Three things are tested:

1. Orbital periods come out right. If Earth doesn't take 365.25 days,
   nothing else matters.
2. Energy is conserved. Symplectic integrators should show bounded,
   oscillating error -- not a trend.
3. Euler fails. Included to make the point concrete.
"""

from __future__ import annotations

import numpy as np

from ..physics import constants as K
from ..physics.kepler import state_to_elements
from ..physics.simulation import Simulation
from ..scenes import solar_system

REAL_PERIODS_DAYS = {
    "Mercury": 87.969, "Venus": 224.701, "Earth": 365.256, "Mars": 686.980,
    "Jupiter": 4332.589, "Saturn": 10759.22, "Uranus": 30685.4,
    "Neptune": 60189.0,
}


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return ok


def test_periods():
    print("\n1. Orbital periods from the initial state vectors")
    store = solar_system.build(asteroids=0, include_moon=False)
    all_ok = True
    for name, real_days in REAL_PERIODS_DAYS.items():
        i = store.index(name)
        # The two-body problem's gravitational parameter uses the SUM of the
        # masses, not just the Sun's. For Jupiter that is a 0.1% correction --
        # small, but larger than the accuracy we are claiming, so it matters.
        mu = K.G * (K.M_SUN + store.mass[i])
        el = state_to_elements(store.pos[i] - store.pos[0],
                               store.vel[i] - store.vel[0], mu)
        days = el["period"] / K.DAY
        err = abs(days - real_days) / real_days
        all_ok &= check(f"{name:<8} {days:10.3f} d  (real {real_days:.3f})",
                        err < 0.002, f"err {err*100:.3f}%")
    return all_ok


def test_energy_conservation():
    print("\n2. Energy conservation over 20 simulated years")
    store = solar_system.build(asteroids=0, include_moon=True)
    sim = Simulation(store, dt=3600.0, integrator="verlet", collisions=False)
    target = 20 * K.YEAR
    drifts = []
    while sim.time < target:
        sim.step()
        if sim.steps % 2000 == 0:
            drifts.append(sim.energy_drift)
    peak = max(drifts)
    # A trend would mean the second half is systematically worse.
    first_half = np.mean(drifts[:len(drifts)//2])
    second_half = np.mean(drifts[len(drifts)//2:])
    ok = check(f"peak relative energy drift = {peak:.3e}", peak < 1e-6)
    ok &= check(f"no secular trend (first half {first_half:.2e} "
                f"vs second {second_half:.2e})",
                second_half < max(first_half * 3.0, 1e-12))
    return ok


def test_angular_momentum():
    print("\n3. Angular momentum conservation over 20 simulated years")
    store = solar_system.build(asteroids=0, include_moon=True)
    sim = Simulation(store, dt=3600.0, collisions=False)
    L0 = np.linalg.norm(store.angular_momentum())
    while sim.time < 20 * K.YEAR:
        sim.step()
    L1 = np.linalg.norm(store.angular_momentum())
    err = abs(L1 - L0) / L0
    return check(f"relative change = {err:.3e}", err < 1e-10)


def test_euler_is_bad():
    print("\n4. Euler integration, for contrast (this SHOULD be terrible)")
    store = solar_system.build(asteroids=0, include_moon=False)
    sim = Simulation(store, dt=3600.0, integrator="euler", collisions=False)
    i = store.index("Earth")
    r0 = np.linalg.norm(store.pos[i] - store.pos[0])
    while sim.time < 20 * K.YEAR:
        sim.step()
    r1 = np.linalg.norm(store.pos[i] - store.pos[0])
    growth = (r1 - r0) / r0
    print(f"       Earth's distance from the Sun changed by {growth*100:+.3f}% "
          f"in 20 years")
    print(f"       energy drift = {sim.energy_drift:.3e} "
          f"(vs ~1e-9 for Verlet)")
    return check("Euler drifts far more than Verlet", sim.energy_drift > 1e-6)


def test_mercury_precession():
    """The headline test: does our GR term reproduce the 1915 result?

    Isolate Sun + Mercury so no other planet contributes, switch the
    relativistic correction on, and measure how far the perihelion moves.
    Einstein's value is 42.98 arcseconds per century.
    """
    print("\n5. Mercury's perihelion precession from the GR correction")
    from ..physics.bodies import BodyStore

    full = solar_system.build(asteroids=0, include_moon=False)
    keep = [full.names.index("Sun"), full.names.index("Mercury")]

    class _Sub(BodyStore):
        def __init__(self, src, keep):
            self.n = len(keep)
            self.names = [src.names[k] for k in keep]
            for attr in ("mass", "radius", "pos", "vel", "acc", "kind",
                         "color", "luminosity", "trail_flag", "alive",
                         "softening"):
                setattr(self, attr, getattr(src, attr)[keep].copy())

    store = _Sub(full, keep)
    store.zero_momentum()
    sim = Simulation(store, dt=2000.0, integrator="yoshida4",
                     relativistic=True, collisions=False)
    mu = K.G * (store.mass[0] + store.mass[1])

    def peri_longitude():
        el = state_to_elements(store.pos[1] - store.pos[0],
                               store.vel[1] - store.vel[0], mu)
        return el["Omega"] + el["omega"]

    start = peri_longitude()
    years = 20.0
    while sim.time < years * K.YEAR:
        sim.step()
    delta = (peri_longitude() - start + 180) % 360 - 180
    arcsec_per_century = delta * 3600.0 * (100.0 / years)
    print(f"       measured {arcsec_per_century:.2f} arcsec/century "
          f"(Einstein: 42.98)")
    return check("matches GR to within 10%",
                 abs(arcsec_per_century - 42.98) < 4.3)


if __name__ == "__main__":
    results = [
        test_periods(),
        test_energy_conservation(),
        test_angular_momentum(),
        test_euler_is_bad(),
        test_mercury_precession(),
    ]
    print("\n" + ("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED"))
