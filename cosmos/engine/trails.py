"""
Orbit trails.

A trail is a ring buffer of past positions. Ring buffers matter here: a
naive `list.append` plus `list.pop(0)` is O(n) per frame and quietly becomes
the slowest thing in your renderer. Writing into a fixed numpy array at a
rotating index is O(1) and allocates nothing.

Positions are stored in float64 WORLD coordinates and converted to
camera-relative float32 only at draw time -- same reason as everywhere else
in this renderer.
"""

from __future__ import annotations

import numpy as np


class Trails:
    def __init__(self, indices, capacity: int = 2000, min_step: float = 0.0):
        """
        indices   -- which bodies to trail
        capacity  -- points kept per body
        min_step  -- only record a new point once the body has moved this far
                     (metres). Stops a slow outer planet from filling its
                     buffer with thousands of identical points.
        """
        self.indices = list(indices)
        self.capacity = capacity
        self.min_step = min_step
        n = len(self.indices)
        self.buf = np.zeros((n, capacity, 3), dtype=np.float64)
        self.count = np.zeros(n, dtype=np.int32)     # how many are valid
        self.head = np.zeros(n, dtype=np.int32)      # next write slot
        self._last = np.full((n, 3), np.nan)

    def record(self, positions: np.ndarray) -> None:
        for k, bi in enumerate(self.indices):
            p = positions[bi]
            if self.min_step > 0.0 and self.count[k] > 0:
                d = p - self._last[k]
                if d @ d < self.min_step * self.min_step:
                    continue
            self.buf[k, self.head[k]] = p
            self._last[k] = p
            self.head[k] = (self.head[k] + 1) % self.capacity
            if self.count[k] < self.capacity:
                self.count[k] += 1

    def polyline(self, k: int, camera_pos: np.ndarray):
        """Oldest-to-newest camera-relative points for trail k, plus ages."""
        c = int(self.count[k])
        if c < 2:
            return None, None
        h = int(self.head[k])
        if c < self.capacity:
            pts = self.buf[k, :c]
        else:
            # Unwrap the ring so the line is drawn in chronological order.
            pts = np.concatenate([self.buf[k, h:], self.buf[k, :h]], axis=0)
        rel = (pts - camera_pos).astype(np.float32)
        age = np.linspace(1.0, 0.0, len(rel), dtype=np.float32)
        return rel, age

    def clear(self) -> None:
        self.count[:] = 0
        self.head[:] = 0
        self._last[:] = np.nan
