"""Scenario loader and validation failures."""

import copy

import pydantic
import pytest

from uavta.scenarios.loader import scenario_from_dict
from uavta.scenarios.scenario import ScenarioError

BASE = {
    "name": "tiny",
    "airspace": [1000.0, 1000.0, 100.0],
    "lambda_obse": 0.001,
    "lambda_risk": 0.0002,
    "uav_types": {
        "A": {
            "speed": 10.0,
            "endurance": 600.0,
            "sensors": {"VL": 1, "IR": 0, "RD": 0},
            "vulnerability": 1.0,
        }
    },
    "uavs": [{"id": "U1", "type": "A", "position": [0.0, 0.0, 50.0]}],
    "targets": [
        {"id": "T1", "value": 50, "observation_time": 10, "position": [500.0, 0.0, 50.0]},
        {
            "id": "N1",
            "value": 40,
            "observation_time": 5,
            "position": [0.0, 500.0, 50.0],
            "initial": False,
        },
    ],
    "events": [],
}


def make(**changes):
    data = copy.deepcopy(BASE)
    data.update(changes)
    return data


def test_valid_minimal():
    sc = scenario_from_dict(make())
    assert sc.name == "tiny" and len(sc.uavs) == 1 and len(sc.targets) == 1
    assert sc.initial_weather == (1.0, 1.0, 1.0)


def test_unknown_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        scenario_from_dict(make(typo_field=1))


def test_unknown_uav_in_event():
    with pytest.raises(ScenarioError, match="unknown UAV"):
        scenario_from_dict(make(events=[{"time": 5, "type": "uav_crash", "uav": "U9"}]))


def test_sensor_not_carried():
    with pytest.raises(ScenarioError, match="does not carry"):
        scenario_from_dict(
            make(events=[{"time": 5, "type": "sensor_damage", "uav": "U1", "sensor": "IR"}])
        )


def test_move_before_target_appears():
    events = [
        {"time": 5, "type": "target_move", "target": "N1", "to": [10.0, 10.0, 10.0]},
        {"time": 9, "type": "new_target", "target": "N1"},
    ]
    with pytest.raises(ScenarioError, match="future target"):
        scenario_from_dict(make(events=events))


def test_new_target_must_be_non_initial():
    with pytest.raises(ScenarioError, match="initial: false"):
        scenario_from_dict(make(events=[{"time": 5, "type": "new_target", "target": "T1"}]))


def test_new_target_position_mismatch():
    with pytest.raises(ScenarioError, match="differs"):
        scenario_from_dict(
            make(
                events=[
                    {"time": 5, "type": "new_target", "target": "N1", "position": [1.0, 2.0, 3.0]}
                ]
            )
        )


def test_position_outside_airspace():
    uavs = [{"id": "U1", "type": "A", "position": [0.0, 0.0, 500.0]}]
    with pytest.raises(ScenarioError, match="outside"):
        scenario_from_dict(make(uavs=uavs))


def test_variant_required_when_declared():
    with pytest.raises(ScenarioError, match="variant"):
        scenario_from_dict(make(variants=["A", "B"]))


def test_variant_given_without_declaration():
    with pytest.raises(ScenarioError, match="no variants"):
        scenario_from_dict(make(), variant="A")
