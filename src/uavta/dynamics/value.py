"""Timeliness decay and expected collected value (paper Eq. 4)."""

from __future__ import annotations

import math


def decay_factor(arrival_time: float, reference_time: float, lambda_obse: float) -> float:
    """exp(−λ_obse · Δt) with Δt = arrival_time − reference_time (G1 resolved by J1)."""
    elapsed = arrival_time - reference_time
    if elapsed < 0:
        raise ValueError(f"arrival_time {arrival_time} precedes reference_time {reference_time}")
    return math.exp(-lambda_obse * elapsed)


def expected_collected_value(value: float, eta: float, survival: float, decay: float) -> float:
    """Vₖ · ηᵢ · p_surv,i · exp(−λ_obse Δt): one summand of the second term of Eq. 4."""
    for name, x in (("eta", eta), ("survival", survival), ("decay", decay)):
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {x}")
    return value * eta * survival * decay
