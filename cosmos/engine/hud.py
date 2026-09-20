"""
Text overlay.

OpenGL has no concept of text. The standard approach is to render glyphs
into a texture and draw quads; here we let pygame's font engine draw the
whole overlay into an RGBA surface, upload that as a texture once per frame,
and blit it. Simple, and at a few thousand pixels of text the upload cost is
negligible.

The texture is only re-uploaded when the text actually changes, which for a
HUD updating a few times a second is most frames skipped.
"""

from __future__ import annotations

import numpy as np
import pygame


class Hud:
    def __init__(self, ctx, width, height, font_size=15):
        self.ctx = ctx
        self.width = width
        self.height = height
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace",
                                        font_size)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.texture = ctx.texture((width, height), 4)
        self.texture.filter = (9729, 9729)   # GL_LINEAR
        self._last = None

    def resize(self, w, h):
        self.width, self.height = w, h
        self.surface = pygame.Surface((w, h), pygame.SRCALPHA)
        self.texture.release()
        self.texture = self.ctx.texture((w, h), 4)
        self.texture.filter = (9729, 9729)
        self._last = None

    def draw(self, lines, help_lines=None):
        key = (tuple(lines), tuple(help_lines or ()))
        if key == self._last:
            return self.texture
        self._last = key

        self.surface.fill((0, 0, 0, 0))
        y = 10
        for text in lines:
            # Cheap drop shadow -- text over a starfield is otherwise
            # unreadable wherever a bright star sits behind a glyph.
            shadow = self.font.render(text, True, (0, 0, 0))
            self.surface.blit(shadow, (13, y + 1))
            img = self.font.render(text, True, (205, 226, 245))
            self.surface.blit(img, (12, y))
            y += self.font.get_linesize()

        if help_lines:
            y = self.height - 10 - self.font.get_linesize() * len(help_lines)
            for text in help_lines:
                shadow = self.font.render(text, True, (0, 0, 0))
                self.surface.blit(shadow, (13, y + 1))
                img = self.font.render(text, True, (130, 150, 170))
                self.surface.blit(img, (12, y))
                y += self.font.get_linesize()

        data = pygame.image.tostring(self.surface, "RGBA", False)
        self.texture.write(data)
        return self.texture


def fmt_distance(m: float) -> str:
    """Human-readable distance, choosing sensible astronomical units."""
    from ..physics import constants as K
    a = abs(m)
    if a >= 0.1 * K.LY:
        return f"{m / K.LY:.3f} ly"
    if a >= 0.01 * K.AU:
        return f"{m / K.AU:.4f} AU"
    if a >= 1e6:
        return f"{m / 1e3:,.0f} km"
    return f"{m:,.0f} m"


def fmt_time(seconds: float) -> str:
    from ..physics import constants as K
    a = abs(seconds)
    if a >= K.YEAR:
        return f"{seconds / K.YEAR:,.3f} yr"
    if a >= K.DAY:
        return f"{seconds / K.DAY:,.2f} d"
    if a >= 3600:
        return f"{seconds / 3600:,.2f} h"
    return f"{seconds:,.1f} s"
