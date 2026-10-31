"""Gymnasium environment ``MultiUAVDynamicEnv`` wrapping the event-driven simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field

from uavta.environment.masking import action_mask
from uavta.environment.observation import ObservationSpec, build_observation
from uavta.environment.reward import RewardConfig, compute_reward
from uavta.environment.simulator import InvalidActionError, Simulator, SimulatorConfig
from uavta.scenarios.generator import TRAIN_SEED_UPPER, GeneratorConfig, ScenarioGenerator
from uavta.scenarios.loader import load_paper_scenario, load_scenario
from uavta.scenarios.scenario import Scenario


class ScenarioSourceConfig(BaseModel):
    """Where reset() gets its scenario when ``options`` do not pin one."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["generator", "paper", "file"] = "generator"
    variant: str | None = None
    path: str | None = None
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)


class EnvConfig(BaseModel):
    """Complete environment configuration (loadable from YAML)."""

    model_config = ConfigDict(extra="forbid")

    max_uavs: int = Field(6, gt=0)
    max_targets: int = Field(24, gt=0)
    horizon: float = Field(2400.0, gt=0)
    energy_reserve: float = Field(0.0, ge=0, lt=1)
    idle_consumes_energy: bool = True
    allow_voluntary_wait: bool = False
    retask_on_event: bool = False  # extended environment: re-decisions at new-target/crash events
    min_operational_uavs: int = Field(1, ge=0)
    invalid_action_mode: Literal["strict", "penalize"] = "strict"
    max_decisions: int = Field(400, gt=0)
    value_max: float = Field(100.0, gt=0)
    speed_ref: float = Field(30.0, gt=0)
    extended_features: bool = False  # extra target features for the extended environment
    reward: RewardConfig = Field(default_factory=RewardConfig)
    scenario: ScenarioSourceConfig = Field(default_factory=ScenarioSourceConfig)

    def simulator_config(self) -> SimulatorConfig:
        return SimulatorConfig(
            horizon=self.horizon,
            energy_reserve=self.energy_reserve,
            idle_consumes_energy=self.idle_consumes_energy,
            allow_voluntary_wait=self.allow_voluntary_wait,
            min_operational_uavs=self.min_operational_uavs,
            retask_on_event=self.retask_on_event,
        )


class MultiUAVDynamicEnv(gym.Env):
    """Sequential dispatch of observation tasks to a heterogeneous UAV team.

    Each step assigns one target slot (or WAIT) to the decision UAV named in the observation;
    the simulator then advances continuous time to the next decision epoch.

    ``reset`` options:
        ``scenario``: a :class:`Scenario` to run instead of the configured source.
        ``scenario_seed``: seed for the configured generator (used for fixed evaluation sets).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = EnvConfig.model_validate(config)
        self.config = config
        airspace = config.scenario.generator.airspace
        self._fixed_scenario: Scenario | None = None
        if config.scenario.source == "paper":
            if config.scenario.variant is None:
                raise ValueError("scenario.source 'paper' needs scenario.variant")
            self._fixed_scenario = load_paper_scenario(config.scenario.variant)
        elif config.scenario.source == "file":
            if config.scenario.path is None:
                raise ValueError("scenario.source 'file' needs scenario.path")
            self._fixed_scenario = load_scenario(
                Path(config.scenario.path), config.scenario.variant
            )
        if self._fixed_scenario is not None:
            airspace = self._fixed_scenario.airspace
        self.generator = ScenarioGenerator(config.scenario.generator)
        self.obs_spec = ObservationSpec(
            max_uavs=config.max_uavs,
            max_targets=config.max_targets,
            horizon=config.horizon,
            airspace=airspace,
            value_max=config.value_max,
            speed_ref=config.speed_ref,
            extended=config.extended_features,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_spec.size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.max_targets + 1)
        self.wait_action = config.max_targets
        self.sim: Simulator | None = None
        self.scenario: Scenario | None = None
        self.n_invalid = 0
        self.n_steps = 0
        self.episode_return = 0.0
        self.value_ref = 1.0

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            scenario = options["scenario"]
        elif "scenario_seed" in options:
            scenario = self.generator.generate(int(options["scenario_seed"]))
        elif self._fixed_scenario is not None:
            scenario = self._fixed_scenario
        else:
            scenario = self.generator.generate(int(self.np_random.integers(TRAIN_SEED_UPPER)))
        if len(scenario.uavs) > self.config.max_uavs:
            raise ValueError(
                f"scenario {scenario.name} has {len(scenario.uavs)} UAVs > max_uavs "
                f"{self.config.max_uavs}"
            )
        self.scenario = scenario
        self.sim = Simulator(
            scenario, self.config.simulator_config(), max_targets=self.config.max_targets
        )
        self.sim.step_acc.reset()
        self.value_ref = max(self.sim.initial_target_value, 1e-9)
        self.n_invalid = 0
        self.n_steps = 0
        self.episode_return = 0.0
        return self._obs(), self._info({})

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        sim = self._require_sim()
        if sim.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise InvalidActionError(f"action {action} outside the action space")
        self.n_steps += 1
        mask = action_mask(sim, self.config.max_targets)
        sim.step_acc.reset()
        invalid = not mask[action]
        if invalid:
            if self.config.invalid_action_mode == "strict":
                raise InvalidActionError(
                    f"action {action} is masked out for UAV {sim.decision_uav} at t={sim.t}"
                )
            self.n_invalid += 1  # penalize mode: logged and penalized, state unchanged
        else:
            sim.assign(None if action == self.wait_action else action)
            if sim.decision_uav is None and not sim.done:
                sim.run_until_decision()
        reward, components = compute_reward(
            sim.step_acc,
            self.config.reward,
            value_ref=self.value_ref,
            horizon=self.config.horizon,
            num_uavs=self.config.max_uavs,
            invalid=invalid,
        )
        self.episode_return += reward
        terminated = sim.terminated
        truncated = sim.truncated or (not sim.done and self.n_steps >= self.config.max_decisions)
        if truncated and not sim.truncated:
            sim.termination_reason = "max_decisions"
        info = self._info({"reward_components": components, "invalid_action": invalid})
        if terminated or truncated:
            sim.close_activities()
            info["mission"] = self.mission_summary()
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Hook used by ``sb3_contrib.MaskablePPO``."""
        return action_mask(self._require_sim(), self.config.max_targets)

    # ------------------------------------------------------------------ helpers

    def mission_summary(self) -> dict[str, Any]:
        """Realized outcomes for evaluation (privileged; never part of the observation)."""
        sim = self._require_sim()
        summary = sim.episode_summary()
        w = self.config.reward
        summary.update(
            {
                "scenario": sim.scenario.name,
                "episode_return": self.episode_return,
                "n_invalid_actions": self.n_invalid,
                "n_steps": self.n_steps,
                "termination_reason": sim.termination_reason,
                "weighted_score": w.w_value * summary["value_fraction"]
                - w.w_time * summary["makespan"] / self.config.horizon,
            }
        )
        return summary

    def _require_sim(self) -> Simulator:
        if self.sim is None:
            raise RuntimeError("call reset() first")
        return self.sim

    def _obs(self) -> np.ndarray:
        return build_observation(self._require_sim(), self.obs_spec)

    def _info(self, extra: dict[str, Any]) -> dict[str, Any]:
        sim = self._require_sim()
        info = {
            "time": sim.t,
            "decision_uav": sim.decision_uav,
            "collected_value": sim.totals.collected_value,
            "n_completed": sim.totals.completions,
            "n_invalid_actions": self.n_invalid,
        }
        info.update(extra)
        return info


ENV_ID = "MultiUAVDynamic-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="uavta.environment.env:MultiUAVDynamicEnv")
