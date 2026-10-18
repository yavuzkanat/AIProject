"""UAV and target data models."""

import numpy as np
import pytest

from uavta.entities import TargetSpec, TargetState, UAVSpec, UAVState, UAVStatus


def make_uav(**kw):
    base = dict(
        id="U1",
        uav_type="A",
        speed=15.0,
        endurance=1800.0,
        sensors=(True, True, False),
        vulnerability=0.9,
        start_position=(1.0, 2.0, 3.0),
    )
    base.update(kw)
    return UAVSpec(**base)


def test_uav_state_from_spec():
    st = UAVState.from_spec(make_uav())
    assert st.energy == 1.0 and st.survival == 1.0 and st.status is UAVStatus.IDLE
    np.testing.assert_array_equal(st.sensor_ok, [True, True, False])
    assert st.is_operational and not st.has_task
    assert make_uav().consumption_rate == pytest.approx(1 / 1800)


def test_uav_state_does_not_alias_spec():
    spec = make_uav()
    st = UAVState.from_spec(spec)
    st.position[0] = 999.0
    assert spec.start_position[0] == 1.0


@pytest.mark.parametrize("status", [UAVStatus.FAILED, UAVStatus.DEPLETED])
def test_non_operational_statuses(status):
    st = UAVState.from_spec(make_uav())
    st.status = status
    assert not st.is_operational


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed", 0.0),
        ("endurance", -1.0),
        ("vulnerability", 0.0),
        ("sensors", (True,)),
        ("start_position", (1.0, 2.0)),
    ],
)
def test_invalid_uav_spec(field, value):
    with pytest.raises(ValueError):
        make_uav(**{field: value})


def test_target_state_reference_time_is_appearance():
    spec = TargetSpec(
        id="NT1", value=76, observation_time=15, position=(2184, 6889, 83), initial=False
    )
    st = TargetState.from_spec(spec, slot=18, appear_time=100.0)
    assert st.reference_time == 100.0 and st.slot == 18 and st.is_available
    st.locked_by = 0
    assert not st.is_available


@pytest.mark.parametrize("field,value", [("value", 0.0), ("observation_time", -2.0)])
def test_invalid_target_spec(field, value):
    base = dict(id="T", value=50.0, observation_time=10.0, position=(0, 0, 0))
    base[field] = value
    with pytest.raises(ValueError):
        TargetSpec(**base)
