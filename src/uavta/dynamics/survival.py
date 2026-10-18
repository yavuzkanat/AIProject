"""Path-dependent survival probability (paper Eqs. 6–7).

The segment risk is θ = 1 − exp(−λ_risk ξ d), so the survival factor of a segment is
1 − θ = exp(−λ_risk ξ d). Because this factor is multiplicative in distance, applying it to
partial segments (e.g. a redirected flight) composes exactly to the full-segment value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def segment_survival(dist: float, vulnerability: float, lambda_risk: float) -> float:
    """1 − θ_{i,k} = exp(−λ_risk · ξᵢ · d) for a segment of ``dist`` m (Eq. 6)."""
    if dist < 0:
        raise ValueError(f"distance must be >= 0, got {dist}")
    return math.exp(-lambda_risk * vulnerability * dist)


def cumulative_survival(
    distances: Iterable[float], vulnerability: float, lambda_risk: float, initial: float = 1.0
) -> float:
    """p^(K) = p^(0) · Π_k (1 − θ_{i,k}) over consecutive segments (Eq. 7).

    ``initial = 1.0`` gives the paper-literal recursion; the simulator passes the UAV's
    survival accumulated so far (J1).
    """
    p = initial
    for d in distances:
        p *= segment_survival(d, vulnerability, lambda_risk)
    return p
