"""Probabilistic multi-sensor fusion (paper Eq. 5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def sensing_capability(available: ArrayLike, weather: ArrayLike) -> float:
    """ηᵢ = 1 − Π_m (1 − I_m · ω_m).

    Args:
        available: Sensor availability I (carried and working), one flag per sensor.
        weather: Weather effectiveness factors ω in (0, 1], one per sensor.

    Returns:
        Sensing capability in [0, 1). A UAV without working sensors has η = 0.
    """
    flags = np.asarray(available, dtype=np.float64)
    omega = np.asarray(weather, dtype=np.float64)
    if flags.shape != omega.shape:
        raise ValueError(f"shape mismatch: availability {flags.shape} vs weather {omega.shape}")
    if np.any((omega <= 0.0) | (omega > 1.0)):
        raise ValueError(f"weather factors must lie in (0, 1], got {omega}")
    return float(1.0 - np.prod(1.0 - flags * omega))


def target_sensing_capability(
    available: ArrayLike, weather: ArrayLike, required: int | None
) -> float:
    """Sensing capability of one UAV for one target.

    Without a requirement this is the paper's fusion model (Eq. 5). A target that requires
    sensor ``required`` (extended environment) can only be observed by that sensor, so the
    capability is I_m · ω_m: 0 for a UAV that lacks it or whose copy is damaged.
    """
    if required is None:
        return sensing_capability(available, weather)
    flags = np.asarray(available, dtype=np.float64)
    omega = np.asarray(weather, dtype=np.float64)
    if not 0 <= required < flags.shape[0]:
        raise ValueError(f"sensor index {required} out of range")
    return float(flags[required] * omega[required])
