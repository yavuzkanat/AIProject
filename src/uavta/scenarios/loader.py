"""Load scenario YAML files into :class:`Scenario` objects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from uavta.entities.sensors import SENSORS
from uavta.entities.target import TargetSpec
from uavta.entities.uav import UAVSpec
from uavta.events.types import (
    Event,
    NewTarget,
    PreferenceShift,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    WeatherChange,
)
from uavta.scenarios.scenario import Scenario, ScenarioError
from uavta.scenarios.schema import (
    EventModel,
    NewTargetModel,
    PreferenceShiftModel,
    ScenarioFile,
    SensorDamageModel,
    TargetMoveModel,
    UAVCrashModel,
    ValueChangeModel,
    WeatherModel,
)
from uavta.utils.config import load_yaml

PAPER_SCENARIO_PATH = Path(__file__).resolve().parents[3] / "configs" / "scenarios" / "paper.yaml"


def _sensor_tuple(mapping: Mapping[str, float], what: str) -> tuple[float, ...]:
    if set(mapping) != set(SENSORS):
        raise ScenarioError(f"{what} must define exactly {SENSORS}, got {sorted(mapping)}")
    return tuple(float(mapping[s]) for s in SENSORS)


def _vec3(v: list[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _included(tags: list[str] | None, variant: str | None) -> bool:
    return tags is None or variant is None or variant in tags


def _convert_event(model: EventModel, target_specs: dict[str, TargetSpec]) -> Event:
    label = None if model.paper_no is None else str(model.paper_no)
    if isinstance(model, UAVCrashModel):
        return UAVCrash(time=model.time, label=label, uav=model.uav)
    if isinstance(model, SensorDamageModel):
        return SensorDamage(time=model.time, label=label, uav=model.uav, sensor=model.sensor)
    if isinstance(model, WeatherModel):
        omega = _sensor_tuple(model.omega, f"weather event at t={model.time}")
        return WeatherChange(time=model.time, label=label, omega=omega, mode=model.mode)
    if isinstance(model, TargetMoveModel):
        return TargetMove(
            time=model.time,
            label=label,
            target=model.target,
            destination=_vec3(model.to),
            expected_from=None if model.from_ is None else _vec3(model.from_),
        )
    if isinstance(model, ValueChangeModel):
        return ValueChange(
            time=model.time,
            label=label,
            target=model.target,
            new_value=model.to,
            expected_from=model.from_,
        )
    if isinstance(model, NewTargetModel):
        spec = target_specs.get(model.target)
        if spec is None or spec.initial:
            raise ScenarioError(
                f"new_target event at t={model.time} must reference a target with initial: false"
            )
        if model.position is not None and not np.allclose(model.position, spec.position):
            raise ScenarioError(
                f"new_target {model.target}: event position {model.position} differs from "
                f"target table position {spec.position}"
            )
        return NewTarget(time=model.time, label=label, target=spec)
    if isinstance(model, PreferenceShiftModel):
        return PreferenceShift(
            time=model.time,
            label=label,
            weights=(model.to[0], model.to[1]),
            expected_from=None if model.from_ is None else (model.from_[0], model.from_[1]),
        )
    raise TypeError(f"unhandled event model {type(model).__name__}")  # pragma: no cover


def scenario_from_dict(data: dict, variant: str | None = None) -> Scenario:
    """Build a :class:`Scenario` from parsed YAML.

    Args:
        data: Parsed scenario file.
        variant: Required if the file declares ``variants``; entities and events whose
            ``scenarios`` list does not contain the variant are dropped.

    Raises:
        pydantic.ValidationError: If the structure is invalid.
        ScenarioError: If the content is inconsistent.
    """
    spec = ScenarioFile.model_validate(data)
    if spec.variants is not None:
        if variant not in spec.variants:
            raise ScenarioError(
                f"{spec.name}: variant must be one of {spec.variants}, got {variant!r}"
            )
    elif variant is not None:
        raise ScenarioError(f"{spec.name}: file defines no variants, got {variant!r}")

    uavs: list[UAVSpec] = []
    for u in spec.uavs:
        if not _included(u.scenarios, variant):
            continue
        utype = spec.uav_types.get(u.type)
        if utype is None:
            raise ScenarioError(f"UAV {u.id}: unknown type {u.type!r}")
        uavs.append(
            UAVSpec(
                id=u.id,
                uav_type=u.type,
                speed=utype.speed,
                endurance=utype.endurance,
                sensors=tuple(bool(x) for x in _sensor_tuple(utype.sensors, f"UAV type {u.type}")),
                vulnerability=utype.vulnerability,
                start_position=_vec3(u.position),
            )
        )

    target_specs = {
        t.id: TargetSpec(
            id=t.id,
            value=t.value,
            observation_time=t.observation_time,
            position=_vec3(t.position),
            initial=t.initial,
            decay_rate=t.decay_rate,
            required_sensor=t.required_sensor,
            deadline=t.deadline,
            late_factor=t.late_factor,
        )
        for t in spec.targets
        if _included(t.scenarios, variant)
    }
    events = [
        _convert_event(e, target_specs) for e in spec.events if _included(e.scenarios, variant)
    ]
    name = spec.name if variant is None else f"{spec.name}_{variant}"
    return Scenario(
        name=name,
        airspace=_vec3(spec.airspace),
        lambda_obse=spec.lambda_obse,
        lambda_risk=spec.lambda_risk,
        uavs=tuple(uavs),
        targets=tuple(t for t in target_specs.values() if t.initial),
        events=tuple(events),
        initial_weather=_sensor_tuple(spec.initial_weather, "initial_weather"),
        initial_preference=(spec.initial_preference[0], spec.initial_preference[1]),
    )


def load_scenario(path: str | Path, variant: str | None = None) -> Scenario:
    """Load a scenario YAML file; see :func:`scenario_from_dict`."""
    return scenario_from_dict(load_yaml(path), variant)


def load_paper_scenario(variant: str) -> Scenario:
    """Load S1, S2 or S3 of the paper exactly as transcribed from Tables 3–5."""
    if not PAPER_SCENARIO_PATH.is_file():
        raise FileNotFoundError(
            f"paper scenario file not found at {PAPER_SCENARIO_PATH}; "
            "install the project in editable mode from the repository root"
        )
    return load_scenario(PAPER_SCENARIO_PATH, variant)
