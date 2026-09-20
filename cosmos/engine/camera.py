"""
The camera.

Two things here are not standard tutorial material, and both exist because
this renderer has to work across fifteen orders of magnitude of distance --
from a 6,000 km planet to a 4.5-billion-km orbit.

1. CAMERA-RELATIVE RENDERING.
   The GPU works in float32, which has ~7 significant digits. Neptune sits
   at 4.5e12 m from the Sun; in float32 the smallest representable step at
   that magnitude is ~300 km. Neptune would visibly jitter and its surface
   would tear.

   The fix: do the subtraction (body_position - camera_position) on the CPU
   in float64, and hand the GPU only the *relative* vector. If you are
   10,000 km from Neptune the GPU sees 1e7 instead of 4.5e12, and float32
   resolves that to under a metre.

   A consequence: the view matrix here is rotation ONLY. The translation has
   already happened, on the CPU, in double precision.

2. LOGARITHMIC DEPTH.
   A standard depth buffer distributes its precision hyperbolically -- almost
   all of it is spent in the first few percent of the near-far range. With
   near=1 m and far=1e14 m, everything past a few kilometres collapses into
   the same depth value and objects flicker in front of each other at random
   ("z-fighting").

   Writing log2(w) instead spreads precision evenly across the orders of
   magnitude, which is exactly what a scene spanning many orders of magnitude
   needs. See the shaders for the two lines that implement it.
"""

from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self,
                 position=(0.0, 0.0, 0.0),
                 fov_deg: float = 60.0,
                 near: float = 1.0,
                 far: float = 1.0e14):
        # float64: this is a real position in metres, possibly 1e12 of them.
        self.position = np.array(position, dtype=np.float64)
        self.yaw = 0.0                 # radians, rotation about world +Y
        self.pitch = 0.0               # radians, positive = looking up
        self.fov = np.radians(fov_deg)
        self.near = near
        self.far = far
        self.aspect = 16.0 / 9.0

        # Follow mode: when set, the camera tracks a body index and keeps a
        # fixed offset from it. Essential for a solar system, where "sit
        # still in space" means the planet you were watching leaves at
        # 30 km/s.
        self.follow_index: int | None = None
        self.follow_distance: float = 1.0e9

    # -- orientation --------------------------------------------------------

    def basis(self):
        """Return (forward, right, up) unit vectors in world space.

        At yaw=pitch=0 this gives forward=(0,0,-1), right=(1,0,0), up=(0,1,0)
        -- the OpenGL convention, where the camera looks down -Z.
        """
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        forward = np.array([-sy * cp, sp, -cy * cp])
        right = np.array([cy, 0.0, -sy])
        up = np.cross(right, forward)
        return forward, right, up

    def rotate(self, dyaw: float, dpitch: float) -> None:
        self.yaw += dyaw
        # Clamp just short of vertical. At exactly 90 degrees the `right`
        # vector degenerates (forward becomes parallel to world up) and the
        # camera flips over -- the classic gimbal problem.
        self.pitch = float(np.clip(self.pitch + dpitch, -1.5533, 1.5533))

    # -- matrices -----------------------------------------------------------

    def view_rotation(self) -> np.ndarray:
        """3x3 world -> view rotation. No translation; see module docstring.

        The rows of this matrix are the camera's own axes. Multiplying a
        vector by it is the same as taking three dot products, which is
        literally what "where is this relative to my axes" means.
        """
        forward, right, up = self.basis()
        return np.array([right, up, -forward], dtype=np.float32)

    def projection(self) -> np.ndarray:
        """Standard perspective matrix, column-major for OpenGL.

        f = 1/tan(fov/2) is the scale factor that maps the edge of the field
        of view onto the edge of the screen. The -1 in row 3, column 2 is the
        whole trick: it copies -z into w, so the hardware's perspective
        divide becomes a divide by depth.

        The third row (depth mapping) is written for completeness but is
        overridden by the logarithmic depth code in the shaders.
        """
        f = 1.0 / np.tan(self.fov * 0.5)
        n, fa = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = (fa + n) / (n - fa)
        m[2, 3] = (2.0 * fa * n) / (n - fa)
        m[3, 2] = -1.0
        return m

    @property
    def log_depth_coef(self) -> float:
        """Fcoef = 2 / log2(far + 1), used by the log-depth shaders."""
        return 2.0 / np.log2(self.far + 1.0)

    # -- movement -----------------------------------------------------------

    def move_local(self, forward_amt: float, right_amt: float, up_amt: float) -> None:
        forward, right, _ = self.basis()
        # Vertical uses WORLD up, not camera up. This is what feels correct
        # when flying: "up" should mean up even when you are looking at your
        # feet. Try it with camera up instead -- the difference is obvious.
        world_up = np.array([0.0, 1.0, 0.0])
        self.position += (forward * forward_amt
                          + right * right_amt
                          + world_up * up_amt)

    def look_at(self, target: np.ndarray) -> None:
        d = np.asarray(target, dtype=np.float64) - self.position
        r = np.linalg.norm(d)
        if r < 1e-9:
            return
        d = d / r
        self.pitch = float(np.arcsin(np.clip(d[1], -1.0, 1.0)))
        self.yaw = float(np.arctan2(-d[0], -d[2]))
