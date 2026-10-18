"""Time-based energy model (paper Eq. 11).

Energy is a fraction of a full battery. Flying and hovering consume the same
rate γᵢ = 1 / t_endu,i; no energy is reserved for returning to base.
"""

from __future__ import annotations


def consumption_rate(endurance: float) -> float:
    """γᵢ = 1 / t_endu,i in energy fraction per second."""
    if endurance <= 0:
        raise ValueError(f"endurance must be > 0, got {endurance}")
    return 1.0 / endurance


def residual_energy(energy: float, rate: float, duration: float) -> float:
    """Energy left after ``duration`` s of operation: E − γ·duration (gᵢ in Eq. 11).

    The result may be negative; callers decide whether that is allowed.
    """
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration}")
    return energy - rate * duration


def is_energy_feasible(energy: float, rate: float, duration: float, reserve: float = 0.0) -> bool:
    """Eq. 11 for one task: True iff E − γ·duration > reserve (strict, as in the paper)."""
    return residual_energy(energy, rate, duration) > reserve


def time_to_depletion(energy: float, rate: float) -> float:
    """Seconds until the energy reaches zero at constant consumption."""
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    return max(energy, 0.0) / rate
