"""Straight-line constant-speed point-mass motion (paper §3.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def distance(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance in m between two 3D points."""
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def travel_time(dist: float, speed: float) -> float:
    """Flight time in s for ``dist`` m at ``speed`` m/s."""
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    if dist < 0:
        raise ValueError(f"distance must be >= 0, got {dist}")
    return dist / speed


def advance_toward(
    position: ArrayLike, destination: ArrayLike, speed: float, dt: float
) -> tuple[np.ndarray, float]:
    """Move from ``position`` toward ``destination`` for ``dt`` s at ``speed`` m/s.

    Motion stops at the destination; it never overshoots.

    Returns:
        The new position (a new array) and the distance actually flown in m.
    """
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    start = np.asarray(position, dtype=np.float64)
    goal = np.asarray(destination, dtype=np.float64)
    remaining = float(np.linalg.norm(goal - start))
    step = speed * dt
    if step >= remaining:
        return goal.copy(), remaining
    return start + (goal - start) * (step / remaining), step


def clip_to_airspace(position: ArrayLike, airspace: ArrayLike) -> np.ndarray:
    """Clip a position to the box [0, Lx] × [0, Ly] × [0, Lz]."""
    return np.clip(
        np.asarray(position, dtype=np.float64), 0.0, np.asarray(airspace, dtype=np.float64)
    )
