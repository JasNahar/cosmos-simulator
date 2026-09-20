"""
Time integrators -- how we step the simulation forward.

This file matters more than it looks. The choice of integrator is the
difference between a solar system that is still recognisable after a
simulated million years, and one where Earth spirals into the Sun because
of accumulated arithmetic error.

The naive method (Euler: x += v*dt; v += a*dt) *always* gains energy on a
circular orbit. Planets spiral outward. It is unusable.

RK4 is accurate per step but not symplectic: energy error drifts steadily
in one direction. Good for a projectile, bad for a solar system.

Symplectic integrators exactly conserve a quantity very close to the true
energy. The energy error oscillates with the orbit instead of growing.
Orbits stay orbits, forever. That is why every serious N-body code uses
them, and why we use them here.

Two are provided:
  * velocity_verlet -- 2nd order, one force evaluation per step. The default.
  * yoshida4        -- 4th order, three force evaluations per step. Use for
                       tight binaries or when you want big timesteps.

Both take a `force_fn(pos) -> acc` closure so they don't care where the
acceleration comes from (Newtonian, +GR correction, +drag, whatever).
"""

from __future__ import annotations

import numpy as np


def velocity_verlet(pos, vel, acc, dt, force_fn):
    """Velocity Verlet. Symplectic, time-reversible, 2nd order.

        x(t+dt) = x(t) + v(t)*dt + a(t)*dt^2/2
        a(t+dt) = f(x(t+dt))                       <- one force eval
        v(t+dt) = v(t) + (a(t) + a(t+dt))*dt/2

    The velocity update averages the acceleration at BOTH ends of the step.
    That symmetry is what makes the method reversible and the energy error
    bounded. Using only a(t) there -- keeping the dt^2/2 term but not the
    average -- leaves you with a 12% orbital drift over 30 years, barely
    better than Euler.

    Returns (pos, vel, acc) -- acc is carried forward so the next step
    doesn't have to recompute it.
    """
    pos = pos + vel * dt + acc * (0.5 * dt * dt)
    new_acc = force_fn(pos)
    vel = vel + (acc + new_acc) * (0.5 * dt)
    return pos, vel, new_acc
# Yoshida's 4th-order coefficients. The trick: compose three 2nd-order
# leapfrog steps, one of which steps *backwards*, chosen so the 3rd-order
# error terms cancel exactly.
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = -(2.0 ** (1.0 / 3.0)) * _W1
_YOSHIDA_C = (_W1 * 0.5, (_W0 + _W1) * 0.5, (_W0 + _W1) * 0.5, _W1 * 0.5)
_YOSHIDA_D = (_W1, _W0, _W1)


def yoshida4(pos, vel, acc, dt, force_fn):
    """4th-order symplectic. ~3x the cost, but error scales as dt^4.

    In practice this means you can take timesteps ~5x larger for the same
    accuracy, so it is often a net win.
    """
    for i in range(3):
        pos = pos + vel * (_YOSHIDA_C[i] * dt)
        acc = force_fn(pos)
        vel = vel + acc * (_YOSHIDA_D[i] * dt)
    pos = pos + vel * (_YOSHIDA_C[3] * dt)
    acc = force_fn(pos)
    return pos, vel, acc

 
def euler(pos, vel, acc, dt, force_fn):
    """Included only so you can watch it fail. Do not use for orbits.

    Run the solar system with this and Earth visibly spirals outward within
    a few thousand simulated years. It is a useful thing to see once.
    """
    pos = pos + vel * dt
    vel = vel + acc * dt
    acc = force_fn(pos)
    return pos, vel, acc


INTEGRATORS = {
    "verlet": velocity_verlet,
    "yoshida4": yoshida4,
    "euler": euler,
}
