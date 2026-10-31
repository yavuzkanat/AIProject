"""Experiment driver: evaluate PPO runs and baselines on named conditions.

A condition fixes the environment configuration and the scenario set; every method sees
exactly the same scenarios (same seeds, same pre-drawn events).
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from uavta.baselines.base import AllocationPolicy
from uavta.baselines.heuristics import GreedyNearestPolicy, GreedyValueRatePolicy, RandomPolicy
from uavta.baselines.rh_rk import RollingHorizonRandomKeyPolicy
from uavta.environment.env import EnvConfig, MultiUAVDynamicEnv
from uavta.evaluation.runner import ScenarioRef, evaluate_policy, seed_refs
from uavta.scenarios.generator import EventRates, GeneratorConfig
from uavta.scenarios.loader import load_paper_scenario
from uavta.utils.config import deep_merge, load_yaml

logger = logging.getLogger(__name__)

BASELINES: dict[str, Callable[[], AllocationPolicy]] = {
    "random": lambda: RandomPolicy(),
    "greedy_value_rate": GreedyValueRatePolicy,
    "greedy_nearest": GreedyNearestPolicy,
    "rh_rk": lambda: RollingHorizonRandomKeyPolicy(),
}
STOCHASTIC_BASELINES = {"random", "rh_rk"}


class ScenarioSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: str | None = None
    seed_base: int | None = None
    count: int = Field(100, gt=0)


class Condition(BaseModel):
    """One evaluation condition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scenarios: ScenarioSet
    generator: dict[str, Any] = Field(default_factory=dict)
    event_rate_scale: float | None = None
    event_class_only: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    repeats: int = Field(1, gt=0)
    baselines: list[str] | None = None


class LookaheadSpec(BaseModel):
    """Decision-time lookahead on top of every run matching ``pattern``."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    top_k: int = Field(2, gt=0)
    samples: int = Field(0, ge=0)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    output: str
    ppo_runs: dict[str, str] = Field(default_factory=dict)
    lookahead_runs: dict[str, LookaheadSpec] = Field(default_factory=dict)
    model_file: str = "best_model.zip"
    baselines: list[str] = Field(default_factory=lambda: list(BASELINES))
    # Base environment: the ``env`` section of a training configuration (so that evaluation
    # always uses the environment the models were trained in), then ``env`` overrides on top.
    env_from: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    conditions: list[Condition]

    def base_env(self) -> dict[str, Any]:
        base: dict[str, Any] = {}
        if self.env_from is not None:
            base = (load_yaml(self.env_from) or {}).get("env") or {}
        return deep_merge(base, self.env)


def condition_env_config(base_env: dict[str, Any], cond: Condition) -> EnvConfig:
    data = deep_merge(EnvConfig().model_dump(mode="json"), base_env)
    data = deep_merge(data, cond.env)
    gen = deep_merge(data["scenario"]["generator"], cond.generator)
    rates = EventRates(**gen["event_rates"])
    if cond.event_class_only is not None:
        rates = rates.only(cond.event_class_only)
    if cond.event_rate_scale is not None:
        rates = rates.scaled(cond.event_rate_scale)
    gen["event_rates"] = rates.model_dump()
    data["scenario"]["generator"] = GeneratorConfig.model_validate(gen).model_dump(mode="json")
    data["scenario"]["source"] = "generator"
    return EnvConfig.model_validate(data)


def condition_refs(cond: Condition) -> list[ScenarioRef]:
    s = cond.scenarios
    if s.paper is not None:
        return [ScenarioRef(f"paper_{s.paper}", scenario=load_paper_scenario(s.paper))]
    if s.seed_base is None:
        raise ValueError(f"condition {cond.name}: give scenarios.paper or scenarios.seed_base")
    return seed_refs(s.seed_base, s.count, cond.name)


def discover_runs(pattern: str, model_file: str) -> list[tuple[int, Path]]:
    """Return (seed, model path) for every run directory matching ``pattern``."""
    out = []
    for d in sorted(glob.glob(pattern)):
        path = Path(d) / model_file
        if path.exists():
            seed = int(Path(d).name.rsplit("seed", 1)[-1])
            out.append((seed, path))
    return out


def run_experiment(spec: ExperimentSpec) -> pd.DataFrame:
    """Evaluate every method on every condition and write one CSV (appending per condition)."""
    import torch

    from uavta.agents.ppo_factory import PPOPolicy

    torch.set_num_threads(1)  # fair single-thread decision latency
    ppo_models: dict[str, list[tuple[int, AllocationPolicy]]] = {
        name: [
            (seed, PPOPolicy.load(path, name=name))
            for seed, path in discover_runs(pat, spec.model_file)
        ]
        for name, pat in spec.ppo_runs.items()
    }
    from uavta.agents.lookahead import LookaheadPolicy

    for name, la in spec.lookahead_runs.items():
        ppo_models[name] = [
            (
                seed,
                LookaheadPolicy(
                    PPOPolicy.load(path), top_k=la.top_k, samples=la.samples, name=name
                ),
            )
            for seed, path in discover_runs(la.pattern, spec.model_file)
        ]
    for name, models in ppo_models.items():
        logger.info("%s: %d trained seeds", name, len(models))
    frames = []
    for cond in spec.conditions:
        env = MultiUAVDynamicEnv(condition_env_config(spec.base_env(), cond))
        refs = condition_refs(cond)
        for name in cond.baselines if cond.baselines is not None else spec.baselines:
            reps = cond.repeats if name in STOCHASTIC_BASELINES else 1
            for r in range(reps):
                df = evaluate_policy(
                    env,
                    BASELINES[name](),
                    refs,
                    method=name,
                    extra={"train_seed": -1, "repeat": r},
                    policy_seed=10_000 * r,
                )
                frames.append(df.assign(condition=cond.name))
        for name, models in ppo_models.items():
            for seed, policy in models:
                df = evaluate_policy(
                    env, policy, refs, method=name, extra={"train_seed": seed, "repeat": 0}
                )
                frames.append(df.assign(condition=cond.name))
        logger.info("condition %s done", cond.name)
    result = pd.concat(frames, ignore_index=True)
    out = Path(spec.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result
