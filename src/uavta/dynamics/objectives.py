"""Plan-level objectives and energy constraint of the paper (Eqs. 4, 8, 9–11).

This evaluates a complete allocation (an ordered task sequence per UAV) from a planning
instant ``t``. The RL environment does not use it to compute rewards; it is used by the
optional re-planning baseline and to cross-check the event-driven simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from uavta.dynamics.kinematics import distance
from uavta.dynamics.survival import segment_survival
from uavta.dynamics.value import decay_factor

DecayReference = Literal["absolute", "plan"]


@dataclass(frozen=True)
class PlanEvaluation:
    """Result of :func:`evaluate_plan`.

    Attributes:
        f1: Remaining target value (Eq. 4); minimize.
        f2: Makespan, the latest UAV completion time in absolute s (Eq. 8); minimize.
        completion_times: t_comp,i per UAV in absolute s.
        residual_energy: gᵢ = Eᵢ − γᵢ (t_comp,i − t_start,i) per UAV (Eq. 11).
        collected_value: Expected value collected per UAV.
        feasible: True iff every gᵢ > 0.
    """

    f1: float
    f2: float
    completion_times: np.ndarray
    residual_energy: np.ndarray
    collected_value: np.ndarray
    feasible: bool


def evaluate_plan(
    plan: Sequence[Sequence[int]],
    *,
    t: float,
    uav_positions: ArrayLike,
    uav_speeds: ArrayLike,
    uav_rates: ArrayLike,
    uav_energies: ArrayLike,
    uav_vulnerabilities: ArrayLike,
    uav_etas: ArrayLike,
    target_positions: ArrayLike,
    target_values: ArrayLike,
    target_observation_times: ArrayLike,
    target_reference_times: ArrayLike,
    unfinished: ArrayLike,
    lambda_obse: float,
    lambda_risk: float,
    uav_initial_survival: ArrayLike | None = None,
    uav_ready_times: ArrayLike | None = None,
    decay_reference: DecayReference = "absolute",
    require_complete: bool = True,
) -> PlanEvaluation:
    """Evaluate an allocation with the paper's objectives.

    Args:
        plan: ``plan[i]`` is the ordered list of target indices assigned to UAV i.
        t: Planning instant in s.
        uav_positions: (S, 3) positions where each sequence starts.
        uav_speeds, uav_rates, uav_energies, uav_vulnerabilities, uav_etas: (S,) vᵢ, γᵢ, Eᵢ, ξᵢ, ηᵢ.
        target_positions: (Z, 3) current target positions.
        target_values, target_observation_times, target_reference_times: (Z,) Vⱼ, t_obse,j, t_ref,j.
        unfinished: (Z,) boolean mask of unfinished targets (T_unfin).
        lambda_obse, lambda_risk: Model constants.
        uav_initial_survival: (S,) p^(0); defaults to 1 (paper-literal Eq. 7).
        uav_ready_times: (S,) absolute time each UAV starts its sequence; defaults to ``t``.
            Energies are taken at that time.
        decay_reference: ``"absolute"`` uses Δt = t_arr − t_ref,j (J1); ``"plan"`` uses
            Δt = t_arr − t (the planning-instant reading of G1).
        require_complete: Enforce the completeness constraint (Eq. 9).

    Raises:
        ValueError: If the plan assigns a target twice (Eq. 10), assigns a finished target,
            or, with ``require_complete``, leaves an unfinished target unassigned (Eq. 9).
    """
    up = np.asarray(uav_positions, dtype=np.float64)
    speeds = np.asarray(uav_speeds, dtype=np.float64)
    rates = np.asarray(uav_rates, dtype=np.float64)
    energies = np.asarray(uav_energies, dtype=np.float64)
    xis = np.asarray(uav_vulnerabilities, dtype=np.float64)
    etas = np.asarray(uav_etas, dtype=np.float64)
    tp = np.asarray(target_positions, dtype=np.float64)
    values = np.asarray(target_values, dtype=np.float64)
    t_obs = np.asarray(target_observation_times, dtype=np.float64)
    t_ref = np.asarray(target_reference_times, dtype=np.float64)
    unfin = np.asarray(unfinished, dtype=bool)
    n_uav = up.shape[0]
    p0 = np.ones(n_uav) if uav_initial_survival is None else np.asarray(uav_initial_survival, float)
    starts = np.full(n_uav, t) if uav_ready_times is None else np.asarray(uav_ready_times, float)

    if len(plan) != n_uav:
        raise ValueError(f"plan has {len(plan)} sequences for {n_uav} UAVs")
    assigned = [j for seq in plan for j in seq]
    if len(assigned) != len(set(assigned)):
        raise ValueError("collaboration constraint (Eq. 10) violated: a target is assigned twice")
    if any(not unfin[j] for j in assigned):
        raise ValueError("plan assigns a finished target")
    if require_complete and set(assigned) != set(np.flatnonzero(unfin).tolist()):
        raise ValueError("completeness constraint (Eq. 9) violated")

    completion = np.empty(n_uav)
    collected = np.zeros(n_uav)
    for i, seq in enumerate(plan):
        clock = float(starts[i])
        pos = up[i]
        survival = float(p0[i])
        for j in seq:
            d = distance(pos, tp[j])
            clock += d / speeds[i]
            survival *= segment_survival(d, xis[i], lambda_risk)
            ref = t_ref[j] if decay_reference == "absolute" else t
            collected[i] += values[j] * etas[i] * survival * decay_factor(clock, ref, lambda_obse)
            clock += t_obs[j]
            pos = tp[j]
        completion[i] = clock

    residual = energies - rates * (completion - starts)
    return PlanEvaluation(
        f1=float(values[unfin].sum() - collected.sum()),
        f2=float(completion.max()) if n_uav else t,
        completion_times=completion,
        residual_energy=residual,
        collected_value=collected,
        feasible=bool(np.all(residual > 0.0)),
    )
