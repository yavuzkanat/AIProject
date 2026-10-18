"""Common interface for allocation policies evaluated in the same environment."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from uavta.environment.env import MultiUAVDynamicEnv


class AllocationPolicy(Protocol):
    """A policy that picks one action per decision epoch.

    Baselines may read the *current* simulator state through ``env.sim`` (the control center
    has full knowledge of the present). They must never read the event schedule.
    """

    name: str

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None: ...

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int: ...


def wait_or_single(mask: np.ndarray) -> int | None:
    """Return the WAIT action if it is the only valid action, else ``None``."""
    valid = np.flatnonzero(mask[:-1])
    return len(mask) - 1 if valid.size == 0 else None
