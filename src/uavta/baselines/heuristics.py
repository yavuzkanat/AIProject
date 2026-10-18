"""Random and greedy dispatch baselines (phase1 §I.1).

These are newly implemented baselines, not algorithms from the paper.
"""

from __future__ import annotations

import numpy as np

from uavta.baselines.base import wait_or_single
from uavta.dynamics.kinematics import distance
from uavta.environment.env import MultiUAVDynamicEnv


class RandomPolicy:
    """Uniform choice among valid actions (seeded)."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self._rng.choice(np.flatnonzero(mask)))


class GreedyValueRatePolicy:
    """argmax_j expected value_j / (travel_j + t_obs,j) over valid targets.

    Inspired by the paper's spatiotemporal greedy seed (V / max(d, 1), §4.3), but uses the
    full expected-value term of Eq. 4 and the task duration. Ties go to the lowest slot.
    Limitations: myopic, ignores the other UAVs' future positions and the load balance.
    """

    name = "greedy_value_rate"

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        return None

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        wait = wait_or_single(mask)
        if wait is not None:
            return wait
        sim = env.sim
        assert sim is not None and sim.decision_uav is not None
        i = sim.decision_uav
        best, best_score = -1, -np.inf
        for slot in np.flatnonzero(mask[:-1]):
            _, duration = sim.task_duration(i, int(slot))
            score = sim.expected_value_now(i, int(slot)) / duration
            if score > best_score:
                best, best_score = int(slot), score
        return best


class GreedyNearestPolicy:
    """The closest valid target (isolates the effect of value awareness)."""

    name = "greedy_nearest"

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        return None

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        wait = wait_or_single(mask)
        if wait is not None:
            return wait
        sim = env.sim
        assert sim is not None and sim.decision_uav is not None
        pos = sim.uavs[sim.decision_uav].position
        slots = np.flatnonzero(mask[:-1])
        dists = [distance(pos, sim.targets[int(s)].position) for s in slots]
        return int(slots[int(np.argmin(dists))])
