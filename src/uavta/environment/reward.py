"""Scalar reward (phase1 §E).

r_k = w_value · ΔV_collected / V_ref − w_time · Δt_active / T_max − w_energy · ΔE / S − c · invalid

With the default weights the episode return equals
w_value · (collected value / V_ref) − w_time · (active mission time / T_max), so the paper's
two objectives (remaining value f₁ and makespan f₂) stay separately identifiable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from uavta.environment.simulator import StepAccumulator


class RewardConfig(BaseModel):
    """Reward weights. All terms are dimensionless after normalization."""

    model_config = ConfigDict(extra="forbid")

    w_value: float = Field(0.5, ge=0)
    w_time: float = Field(0.5, ge=0)
    w_energy: float = Field(0.0, ge=0)
    invalid_penalty: float = Field(0.1, ge=0)


def compute_reward(
    acc: StepAccumulator,
    config: RewardConfig,
    *,
    value_ref: float,
    horizon: float,
    num_uavs: int,
    invalid: bool = False,
) -> tuple[float, dict[str, float]]:
    """Return the reward and its unweighted components for one decision step."""
    if value_ref <= 0:
        raise ValueError("value_ref must be > 0")
    components = {
        "value": acc.collected_value / value_ref,
        "time": acc.active_time / horizon,
        "energy": acc.energy_used / max(num_uavs, 1),
        "invalid": float(invalid),
    }
    reward = (
        config.w_value * components["value"]
        - config.w_time * components["time"]
        - config.w_energy * components["energy"]
        - config.invalid_penalty * components["invalid"]
    )
    return reward, components
