"""Checks of the S1–S3 transcription against facts stated in the paper (§5.1.1, Tables 3–5)."""

import pytest

from uavta.events.types import (
    NewTarget,
    PreferenceShift,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    WeatherChange,
)
from uavta.scenarios.loader import load_paper_scenario
from uavta.scenarios.scenario import ScenarioError


@pytest.fixture(scope="module", params=["S1", "S2", "S3"])
def scenario(request):
    return load_paper_scenario(request.param)


# §5.1.1: 3/4/5 UAVs and 10/14/18 initial targets. Table 4/5: NT1 in all, NT2 only in S3.
EXPECTED = {
    "S1": dict(uavs=3, targets=10, new=1, events=["2", "6", "10", "15"]),
    "S2": dict(uavs=4, targets=14, new=1, events=["2", "4", "6", "8", "10", "12", "15"]),
    "S3": dict(
        uavs=5,
        targets=18,
        new=2,
        events=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14", "15"],
    ),
}


def test_counts(scenario):
    variant = scenario.name.split("_")[-1]
    exp = EXPECTED[variant]
    assert len(scenario.uavs) == exp["uavs"]
    assert len(scenario.targets) == exp["targets"]
    assert scenario.num_new_targets == exp["new"]
    assert sorted(e.label for e in scenario.events) == sorted(exp["events"], key=str)


def test_model_constants(scenario):
    assert scenario.airspace == (8000.0, 8000.0, 200.0)
    assert scenario.lambda_obse == 0.001
    assert scenario.lambda_risk == 0.0002
    assert scenario.initial_preference == (0.5, 0.5)


def test_uav_types(scenario):
    for u in scenario.uavs:
        assert u.speed == 15.0
        if u.uav_type == "A":
            assert (u.endurance, u.vulnerability, u.sensors) == (1800.0, 0.9, (True, True, False))
        else:
            assert (u.endurance, u.vulnerability, u.sensors) == (2400.0, 1.0, (True, True, True))


def test_nested_subsets():
    s1, s2, s3 = (load_paper_scenario(v) for v in ("S1", "S2", "S3"))
    for small, big in ((s1, s2), (s2, s3)):
        assert {u.id for u in small.uavs} < {u.id for u in big.uavs}
        assert {t.id for t in small.targets} < {t.id for t in big.targets}
        # Entities shared across scenarios have identical attributes.
        big_t = {t.id: t for t in big.targets}
        assert all(t == big_t[t.id] for t in small.targets)


def test_spot_values_s3():
    s3 = load_paper_scenario("S3")
    uav = {u.id: u for u in s3.uavs}
    tgt = {t.id: t for t in s3.targets}
    assert uav["U5"].start_position == (70.0, 1400.0, 75.0)
    assert uav["U4"].uav_type == "B"
    assert (tgt["T8"].value, tgt["T8"].observation_time, tgt["T8"].position) == (
        59.0,
        10.0,
        (4886.0, 6545.0, 140.0),
    )
    assert (tgt["T17"].value, tgt["T17"].observation_time) == (99.0, 20.0)


def test_s3_event_sequence_and_ordering():
    ev = load_paper_scenario("S3").events
    # Simultaneous events are ordered crash -> sensor -> weather -> move -> value -> new -> pref.
    assert [type(e) for e in ev[:2]] == [UAVCrash, NewTarget]  # both t=100
    assert [type(e) for e in ev[2:4]] == [ValueChange, PreferenceShift]  # both t=170
    assert [type(e) for e in ev[4:6]] == [UAVCrash, SensorDamage]  # both t=250
    times = [e.time for e in ev]
    assert times == sorted(times)

    by_label = {e.label: e for e in ev}
    assert by_label["1"].uav == "U5"
    assert by_label["2"].target.id == "NT1" and by_label["2"].target.position == (
        2184.0,
        6889.0,
        83.0,
    )
    assert by_label["4"].new_value == 35 and by_label["4"].expected_from == 71
    assert (by_label["5"].uav, by_label["5"].sensor) == ("U2", "VL")
    assert by_label["7"].omega == (0.94, 0.58, 0.54) and by_label["7"].mode == "absolute"
    assert isinstance(by_label["9"], TargetMove)
    assert by_label["9"].destination == (1537.0, 1365.0, 125.0)
    assert by_label["14"].target.id == "NT2"
    assert isinstance(by_label["15"], WeatherChange) and by_label["15"].omega == (0.75, 0.76, 0.91)


def test_s2_moves_new_target_after_it_appears():
    ev = {e.label: e for e in load_paper_scenario("S2").events}
    assert ev["12"].target == "NT1" and ev["12"].time > ev["2"].time


def test_new_targets_are_not_initial():
    s3 = load_paper_scenario("S3")
    assert all(t.initial for t in s3.targets)
    assert {t.id for t in s3.targets}.isdisjoint({"NT1", "NT2"})
    assert s3.max_targets == 20


def test_unknown_variant_rejected():
    with pytest.raises(ScenarioError):
        load_paper_scenario("S4")
