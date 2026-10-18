"""Builders for small, hand-checkable scenarios (importable from tests via pythonpath)."""

from __future__ import annotations

import copy
from typing import Any

from uavta.environment.env import EnvConfig, MultiUAVDynamicEnv
from uavta.environment.simulator import Simulator, SimulatorConfig
from uavta.scenarios.loader import scenario_from_dict
from uavta.scenarios.scenario import Scenario

LAM_OBSE = 0.001
LAM_RISK = 0.0002

BASE: dict[str, Any] = {
    "name": "tiny",
    "airspace": [10000.0, 10000.0, 200.0],
    "lambda_obse": LAM_OBSE,
    "lambda_risk": LAM_RISK,
    "uav_types": {
        "A": {
            "speed": 10.0,
            "endurance": 10000.0,
            "sensors": {"VL": 1, "IR": 0, "RD": 0},
            "vulnerability": 1.0,
        },
        "B": {
            "speed": 10.0,
            "endurance": 10000.0,
            "sensors": {"VL": 1, "IR": 1, "RD": 1},
            "vulnerability": 1.0,
        },
    },
    "uavs": [{"id": "U1", "type": "A", "position": [0.0, 0.0, 100.0]}],
    "targets": [
        {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]},
        {"id": "T2", "value": 80, "observation_time": 20, "position": [3000.0, 0.0, 100.0]},
    ],
    "events": [],
}


def make_scenario(**changes: Any) -> Scenario:
    data = copy.deepcopy(BASE)
    data.update(changes)
    return scenario_from_dict(data)


def make_sim(scenario: Scenario | None = None, max_targets: int = 8, **cfg: Any) -> Simulator:
    return Simulator(scenario or make_scenario(), SimulatorConfig(**cfg), max_targets=max_targets)


def make_env(scenario: Scenario | None = None, **cfg: Any) -> tuple[MultiUAVDynamicEnv, Any]:
    env = MultiUAVDynamicEnv(EnvConfig(max_targets=8, **cfg))
    obs, info = env.reset(seed=0, options={"scenario": scenario or make_scenario()})
    return env, obs


def step_to_done(sim: Simulator, chooser: Any = None) -> None:
    """Drive a simulator with ``chooser(sim) -> slot`` (default: lowest valid slot)."""
    import numpy as np

    while not sim.done:
        valid = sim.valid_targets(sim.decision_uav)
        slot = chooser(sim) if chooser else int(np.flatnonzero(valid)[0])
        sim.assign(slot)
        if not sim.done and sim.decision_uav is None:
            sim.run_until_decision()
