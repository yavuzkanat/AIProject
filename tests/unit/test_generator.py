"""Random scenario generator."""

import pytest

from uavta.events.types import NewTarget, UAVCrash
from uavta.scenarios.generator import EventRates, GeneratorConfig, ScenarioGenerator


def test_deterministic():
    gen = ScenarioGenerator()
    assert gen.generate(5) == gen.generate(5)
    assert gen.generate(5) != gen.generate(6)


def test_layout_independent_of_event_model():
    a = ScenarioGenerator(GeneratorConfig()).generate(11)
    b = ScenarioGenerator(GeneratorConfig(event_rates=EventRates().scaled(0.0))).generate(11)
    assert a.uavs == b.uavs and a.targets == b.targets
    assert len(b.events) == 0


@pytest.mark.parametrize("seed", range(30))
def test_ranges_and_caps(seed):
    cfg = GeneratorConfig(event_rates=EventRates().scaled(4.0), max_total_targets=20)
    sc = ScenarioGenerator(cfg).generate(seed)
    assert 3 <= len(sc.uavs) <= 5
    assert 8 <= len(sc.targets) <= 18
    assert sc.max_targets <= 20
    assert all(0 < e.time <= cfg.event_window for e in sc.events)
    assert all(50 <= t.value <= 100 and 10 <= t.observation_time <= 20 for t in sc.targets)


def test_single_class_rates():
    rates = EventRates().only("uav_crash")
    sc = ScenarioGenerator(GeneratorConfig(event_rates=rates.scaled(5.0))).generate(3)
    assert sc.events and all(isinstance(e, UAVCrash) for e in sc.events)


def test_max_new_targets():
    cfg = GeneratorConfig(event_rates=EventRates(new_target=50.0), max_new_targets=2)
    sc = ScenarioGenerator(cfg).generate(0)
    assert sum(isinstance(e, NewTarget) for e in sc.events) == 2


@pytest.mark.parametrize("dist", ["uniform", "bimodal", "skewed"])
def test_value_distributions(dist):
    sc = ScenarioGenerator(GeneratorConfig(value_distribution=dist)).generate(1)
    assert all(50 <= t.value <= 100 for t in sc.targets)


def test_endurance_scale():
    sc = ScenarioGenerator(GeneratorConfig(endurance_scale=0.5)).generate(2)
    assert all(u.endurance in (900.0, 1200.0) for u in sc.uavs)


def test_invalid_config():
    with pytest.raises(ValueError):
        GeneratorConfig(n_uavs=(4, 2))
    with pytest.raises(ValueError):
        GeneratorConfig(uav_type_probs={"A": 0.5, "B": 0.6})
