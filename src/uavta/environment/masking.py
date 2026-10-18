"""Action masks for the UAV-centric discrete action space (phase1 §C, option C).

Action ``a < max_targets`` assigns target slot ``a`` to the decision UAV; action
``max_targets`` is WAIT.
"""

from __future__ import annotations

import numpy as np

from uavta.environment.simulator import Simulator


def action_mask(sim: Simulator, max_targets: int) -> np.ndarray:
    """Boolean mask of length ``max_targets + 1``; never all-False.

    WAIT is valid only when the decision UAV has no valid target (work-conserving, J3a), when
    voluntary waiting is enabled (J3b), or when no decision is pending (terminal state).
    """
    mask = np.zeros(max_targets + 1, dtype=bool)
    if sim.decision_uav is None:
        mask[max_targets] = True
        return mask
    mask[:max_targets] = sim.valid_targets(sim.decision_uav)
    mask[max_targets] = (not mask[:max_targets].any()) or sim.config.allow_voluntary_wait
    return mask
