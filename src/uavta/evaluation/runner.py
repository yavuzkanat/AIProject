"""Run allocation policies on identical scenarios and collect per-episode metrics."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from uavta.baselines.base import AllocationPolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.environment.simulator import Activity
from uavta.evaluation.metrics import finalize_metrics
from uavta.scenarios.scenario import Scenario


@dataclass
class EpisodeRecord:
    """Metrics plus the logs needed for plots of one episode."""

    metrics: dict[str, Any]
    event_log: list[dict[str, Any]] = field(default_factory=list)
    assignment_log: list[dict[str, Any]] = field(default_factory=list)
    activities: list[Activity] = field(default_factory=list)
    target_positions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioRef:
    """A scenario identified by a generator seed or given explicitly."""

    name: str
    scenario: Scenario | None = None
    seed: int | None = None

    def reset_options(self) -> dict[str, Any]:
        if self.scenario is not None:
            return {"scenario": self.scenario}
        if self.seed is None:
            raise ValueError("ScenarioRef needs a scenario or a seed")
        return {"scenario_seed": self.seed}


def seed_refs(base: int, count: int, prefix: str = "seed") -> list[ScenarioRef]:
    return [ScenarioRef(name=f"{prefix}_{base + k}", seed=base + k) for k in range(count)]


def run_episode(
    env: MultiUAVDynamicEnv,
    policy: AllocationPolicy,
    ref: ScenarioRef,
    *,
    policy_seed: int | None = None,
    keep_logs: bool = False,
) -> EpisodeRecord:
    """Run one episode and return its metrics (and logs if requested)."""
    obs, _ = env.reset(seed=0, options=ref.reset_options())
    policy.reset(env, seed=policy_seed)
    overrides_before = getattr(policy, "mask_overrides", 0)
    latencies: list[float] = []
    while True:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action = policy.act(env, obs, mask)
        latencies.append(time.perf_counter() - t0)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    sim = env.sim
    assert sim is not None
    metrics = finalize_metrics(info["mission"], sim, latencies)
    metrics["scenario_ref"] = ref.name
    # Unmasked policies whose argmax was masked out and replaced (never silently).
    metrics["mask_overrides"] = getattr(policy, "mask_overrides", 0) - overrides_before
    record = EpisodeRecord(metrics=metrics)
    if keep_logs:
        record.event_log = list(sim.event_log)
        record.assignment_log = list(sim.assignment_log)
        record.activities = list(sim.activities)
        record.target_positions = {
            tg.spec.id: {
                "position": tg.position.tolist(),
                "initial_position": list(tg.spec.position),
                "value": tg.value,
                "completed": tg.completed,
                "completion_time": tg.completion_time,
                "collected_value": tg.collected_value,
                "appear_time": tg.appear_time,
                "initial_value": tg.spec.value,
                "observation_time": tg.spec.observation_time,
                "slot": tg.slot,
            }
            for tg in sim.targets
        }
    return record


def evaluate_policy(
    env: MultiUAVDynamicEnv,
    policy: AllocationPolicy,
    refs: Sequence[ScenarioRef],
    *,
    method: str | None = None,
    extra: dict[str, Any] | None = None,
    policy_seed: int = 0,
) -> pd.DataFrame:
    """Evaluate ``policy`` on every scenario; one row per episode."""
    rows = []
    for k, ref in enumerate(refs):
        rec = run_episode(env, policy, ref, policy_seed=policy_seed + k)
        row = {"method": method or policy.name, **(extra or {}), **rec.metrics}
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_many(
    env: MultiUAVDynamicEnv,
    policies: Iterable[tuple[str, AllocationPolicy, dict[str, Any]]],
    refs: Sequence[ScenarioRef],
) -> pd.DataFrame:
    """Evaluate several (name, policy, extra-columns) triples on the same scenarios."""
    frames = [
        evaluate_policy(env, p, refs, method=name, extra=extra) for name, p, extra in policies
    ]
    return pd.concat(frames, ignore_index=True)
