"""Validation rules of event records."""

import pytest

from uavta.entities.target import TargetSpec
from uavta.events.types import (
    EventType,
    NewTarget,
    PreferenceShift,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    WeatherChange,
    sort_events,
)


def test_explicit_or_selector_exclusive():
    UAVCrash(time=1.0, uav="U1")
    UAVCrash(time=1.0, uav_selector=0.3)
    with pytest.raises(ValueError):
        UAVCrash(time=1.0)
    with pytest.raises(ValueError):
        UAVCrash(time=1.0, uav="U1", uav_selector=0.3)


@pytest.mark.parametrize("sel", [-0.1, 1.0])
def test_selector_range(sel):
    with pytest.raises(ValueError):
        UAVCrash(time=1.0, uav_selector=sel)


def test_time_must_be_positive():
    with pytest.raises(ValueError):
        UAVCrash(time=0.0, uav="U1")


def test_sensor_name_checked():
    with pytest.raises(ValueError):
        SensorDamage(time=1.0, uav="U1", sensor="LIDAR")


def test_weather_range():
    with pytest.raises(ValueError):
        WeatherChange(time=1.0, omega=(1.0, 0.0, 0.5))
    with pytest.raises(ValueError):
        WeatherChange(time=1.0, omega=(1.0, 0.5))


def test_move_needs_one_destination_form():
    with pytest.raises(ValueError):
        TargetMove(time=1.0, target="T1")
    with pytest.raises(ValueError):
        TargetMove(time=1.0, target="T1", destination=(1, 1, 1), displacement=(1, 1, 1))


def test_value_change_forms():
    ValueChange(time=1.0, target="T1", new_value=20.0)
    ValueChange(time=1.0, target_selector=0.5, factor=0.5)
    with pytest.raises(ValueError):
        ValueChange(time=1.0, target="T1", new_value=20.0, factor=0.5)


def test_preference_must_sum_to_one():
    with pytest.raises(ValueError):
        PreferenceShift(time=1.0, weights=(0.5, 0.6))


def test_events_are_immutable():
    e = UAVCrash(time=1.0, uav="U1")
    with pytest.raises(AttributeError):
        e.time = 2.0  # type: ignore[misc]


def test_sort_by_time_then_type_then_schedule_order():
    spec = TargetSpec(id="N", value=1.0, observation_time=1.0, position=(0, 0, 0), initial=False)
    a = NewTarget(time=5.0, target=spec, label="a")
    b = UAVCrash(time=5.0, uav="U1", label="b")
    c = UAVCrash(time=5.0, uav="U2", label="c")
    d = WeatherChange(time=1.0, omega=(1, 1, 1), label="d")
    assert [e.label for e in sort_events([a, b, c, d])] == ["d", "b", "c", "a"]
    assert b.event_type is EventType.UAV_CRASH
