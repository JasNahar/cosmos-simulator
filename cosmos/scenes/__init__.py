"""Scene registry. `python -m cosmos --scene <key>` looks names up here."""

from __future__ import annotations

from . import black_holes, solar_system

REGISTRY = {
    "solar": {"build": solar_system.build, "meta": solar_system.METADATA},
    "sgra": {"build": black_holes.sgr_a, "meta": black_holes.METADATA["sgr_a"]},
    "binary": {"build": black_holes.binary, "meta": black_holes.METADATA["binary"]},
}


def load(key: str, **kwargs):
    """Build a scene by name. Returns (BodyStore, metadata dict)."""
    if key not in REGISTRY:
        raise SystemExit(
            f"Unknown scene {key!r}. Available: {', '.join(REGISTRY)}")
    entry = REGISTRY[key]
    return entry["build"](**kwargs), entry["meta"]
