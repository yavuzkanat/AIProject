"""Extended environment (v2): critical targets with deadlines, sensor requirements, urgency."""

import math

import numpy as np
import pytest
import torch
from builders import LAM_OBSE, make_env, make_scenario, make_sim, step_to_done

from uavta.agents.entity_policy import EntityExtractor
from uavta.baselines.rh_rk import RollingHorizonRandomKeyPolicy
from uavta.entities.target import TargetSpec
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.environment.observation import (
    EXTENDED_TARGET_FEATURES,
    EXTRA_TARGET_FEATURE_NAMES,
    TARGET_FEATURE_NAMES,
    TARGET_FEATURES,
    UAV_FEATURES,
)
from uavta.evaluation.runner import ScenarioRef, run_episode, seed_refs
from uavta.scenarios.generator import GeneratorConfig, ScenarioGenerator

UAV_B = {"id": "U2", "type": "B", "position": [0.0, 0.0, 100.0]}


def _target(**kw):
    base = {"id": "T1", "value": 100, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
    base.update(kw)
    return base


# ---------------------------------------------------------------- entities and generator


def test_target_spec_validates_extended_fields():
    with pytest.raises(ValueError):
        TargetSpec("T", 1.0, 1.0, (0, 0, 0), deadline=0.0)
    with pytest.raises(ValueError):
        TargetSpec("T", 1.0, 1.0, (0, 0, 0), late_factor=1.5)
    with pytest.raises(ValueError):
        TargetSpec("T", 1.0, 1.0, (0, 0, 0), required_sensor="XX")
    with pytest.raises(ValueError):
        TargetSpec("T", 1.0, 1.0, (0, 0, 0), decay_rate=-1.0)


def test_default_generator_draws_no_extended_attributes():
    """v1 scenarios are unchanged: no extra attributes and no extra random draws."""
    sc = ScenarioGenerator().generate(123)
    specs = list(sc.targets) + [e.target for e in sc.events if hasattr(e, "target")]
    assert all(
        t.deadline is None and t.required_sensor is None and t.decay_rate is None
        for t in specs
        if isinstance(t, TargetSpec)
    )


def test_generator_critical_targets_and_requirements():
    cfg = GeneratorConfig(
        critical_fraction=1.0,
        critical_value_range=(150.0, 250.0),
        critical_deadline_range=(200.0, 700.0),
        sensor_requirements={"none": 0.0, "RD": 1.0},
        uav_type_probs={"B": 1.0},
    )
    sc = ScenarioGenerator(cfg).generate(7)
    assert all(150 <= t.value <= 250 and 200 <= t.deadline <= 700 for t in sc.targets)
    assert all(t.required_sensor == "RD" for t in sc.targets)
    # Without any radar-carrying UAV the requirement cannot be drawn.
    cfg_a = cfg.model_copy(update={"sensor_requirements": {"none": 0.5, "RD": 0.5}})
    cfg_a = cfg_a.model_copy(update={"uav_type_probs": {"A": 1.0}})
    sc_a = ScenarioGenerator(cfg_a).generate(7)
    assert all(t.required_sensor is None for t in sc_a.targets)


def test_generator_rejects_bad_requirements():
    with pytest.raises(ValueError):
        GeneratorConfig(sensor_requirements={"none": 0.5, "XX": 0.5})
    with pytest.raises(ValueError):
        GeneratorConfig(sensor_requirements={"none": 0.5, "IR": 0.4})


# ---------------------------------------------------------------- simulator


def test_required_sensor_masks_incapable_uav_and_uses_single_sensor_eta():
    uav_a = {"id": "U1", "type": "A", "position": [0.0, 0.0, 100.0]}
    sim = make_sim(make_scenario(uavs=[uav_a, UAV_B], targets=[_target(required_sensor="RD")]))
    assert not sim.valid_targets(0)[0]  # type A carries no radar
    assert sim.valid_targets(1)[0]
    sim.weather = np.array([1.0, 0.8, 0.5])
    assert sim.eta_for(1, 0) == pytest.approx(0.5)  # only radar counts
    assert sim.eta(1) == pytest.approx(1 - 0.0 * 0.2 * 0.5)


def test_deadline_on_time_full_value_late_masked():
    # 1000 m at 10 m/s: arrival at t = 100 s.
    on_time = make_sim(make_scenario(targets=[_target(deadline=150.0)]))
    step_to_done(on_time)
    tg = on_time.targets[0]
    expected = 100 * math.exp(-LAM_OBSE * 100) * math.exp(-0.0002 * 1000)
    assert tg.completed and tg.collected_value == pytest.approx(expected)
    assert on_time.episode_summary()["critical_on_time"] == 1
    late = make_sim(make_scenario(targets=[_target(deadline=50.0)]))
    assert late.done and late.termination_reason == "no_feasible_target"
    assert late.episode_summary()["n_critical"] == 1
    assert late.episode_summary()["critical_on_time"] == 0


def test_deadline_late_factor_keeps_target_valid_with_reduced_value():
    sim = make_sim(make_scenario(targets=[_target(deadline=50.0, late_factor=0.5)]))
    assert sim.valid_targets(0)[0]
    step_to_done(sim)
    expected = 0.5 * 100 * math.exp(-LAM_OBSE * 100) * math.exp(-0.0002 * 1000)
    assert sim.targets[0].collected_value == pytest.approx(expected)
    assert sim.episode_summary()["critical_on_time"] == 0


def test_per_target_decay_rate():
    sim = make_sim(make_scenario(targets=[_target(decay_rate=0.01)]))
    step_to_done(sim)
    expected = 100 * math.exp(-0.01 * 100) * math.exp(-0.0002 * 1000)
    assert sim.targets[0].collected_value == pytest.approx(expected)


def test_sensor_damage_aborts_task_that_needs_the_sensor():
    sc = make_scenario(
        uavs=[UAV_B],
        targets=[_target(required_sensor="RD", position=[5000.0, 0.0, 100.0])],
        events=[{"type": "sensor_damage", "time": 50.0, "uav": "U2", "sensor": "RD"}],
    )
    sim = make_sim(sc)
    sim.assign(0)
    sim.run_until_decision()
    assert sim.done  # nobody can observe the radar-only target any more
    assert sim.targets[0].locked_by is None and not sim.targets[0].completed
    rec = next(r for r in sim.event_log if r["type"] == "sensor_damage")
    assert rec["aborted_uav"] == "U2" and rec["released_target"] == "T1"


def test_move_that_makes_deadline_unreachable_aborts_task():
    sc = make_scenario(
        targets=[_target(deadline=200.0)],
        events=[{"type": "target_move", "time": 50.0, "target": "T1", "to": [9000.0, 0.0, 100.0]}],
    )
    sim = make_sim(sc)
    sim.assign(0)
    sim.run_until_decision()
    rec = next(r for r in sim.event_log if r["type"] == "target_move")
    assert rec["aborted_uav"] == "U1"
    assert sim.done and not sim.targets[0].completed


# ---------------------------------------------------------------- observation and policy


def test_v1_observation_size_unchanged_and_v2_extended():
    assert MultiUAVDynamicEnv().observation_space.shape == (6 * 21 + 24 * 16 + 17,)
    env, obs = make_env(
        make_scenario(uavs=[UAV_B], targets=[_target(deadline=400.0, required_sensor="IR")]),
        extended_features=True,
    )
    assert EXTENDED_TARGET_FEATURES == TARGET_FEATURES + len(EXTRA_TARGET_FEATURE_NAMES)
    assert obs.shape == env.observation_space.shape
    row = obs[6 * UAV_FEATURES : 6 * UAV_FEATURES + EXTENDED_TARGET_FEATURES]
    names = TARGET_FEATURE_NAMES + EXTRA_TARGET_FEATURE_NAMES
    f = dict(zip(names, row, strict=True))
    assert f["has_deadline"] == 1.0 and f["requires_ir"] == 1.0 and f["requires_rd"] == 0.0
    assert f["time_to_deadline"] == pytest.approx(400.0 / 2400.0)
    assert f["deadline_slack"] == pytest.approx((400.0 - 100.0) / 2400.0)  # arrival at 100 s
    assert env.observation_space.contains(obs)


def test_entity_extractor_with_extended_features():
    env, obs = make_env(make_scenario(targets=[_target(deadline=400.0)]), extended_features=True)
    ext = EntityExtractor(6, 8, hidden=16, target_features=EXTENDED_TARGET_FEATURES)
    pi, vf = ext(torch.as_tensor(obs[None]))
    assert pi.shape == (1, 9) and vf.shape == (1, 16)


# ---------------------------------------------------------------- baselines on v2


def test_rh_rk_and_greedy_respect_v2_constraints():
    """Strict mode raises on any invalid action, so finishing proves feasibility."""
    import yaml

    cfg = yaml.safe_load(open("configs/train/ppo_entity_hard.yaml"))["env"]
    env = MultiUAVDynamicEnv(cfg)
    assert env.config.invalid_action_mode == "strict"
    policy = RollingHorizonRandomKeyPolicy()
    for ref in seed_refs(1_000_000, 2):
        m = run_episode(env, policy, ref, policy_seed=0).metrics
        assert m["n_critical"] > 0 and 0 <= m["critical_on_time"] <= m["n_critical"]


def test_replay_reports_deadlines():
    import yaml

    cfg = yaml.safe_load(open("configs/train/ppo_entity_hard.yaml"))["env"]
    env = MultiUAVDynamicEnv(cfg)
    from uavta.baselines.heuristics import GreedyNearestPolicy

    sc = env.generator.generate(5_000_000)
    rec = run_episode(env, GreedyNearestPolicy(), ScenarioRef("x", scenario=sc), keep_logs=True)
    assert any(t["deadline_time"] is not None for t in rec.target_positions.values())
    assert any(t["required_sensor"] in ("IR", "RD") for t in rec.target_positions.values())


# ---------------------------------------------------------------- re-tasking (retask_on_event)


def _retask_scenario():
    return make_scenario(
        targets=[
            _target(id="T1", position=[5000.0, 0.0, 100.0]),
            _target(id="N1", value=200, position=[1000.0, 1000.0, 100.0], initial=False),
        ],
        events=[{"type": "new_target", "time": 100.0, "target": "N1"}],
    )


def test_new_target_triggers_redecision_of_uav_in_transit():
    sim = make_sim(_retask_scenario(), retask_on_event=True)
    sim.assign(0)  # U1 → T1 (5000 m away)
    sim.run_until_decision()
    assert sim.t == pytest.approx(100.0) and sim.decision_uav == 0
    valid = sim.valid_targets(0)
    assert valid[0] and valid[1]  # continue to T1, or switch to N1
    sim.assign(1)
    assert sim.targets[1].locked_by == 0 and sim.targets[0].locked_by is None
    assert sim.assignment_log[-1]["retask"] is True and sim.totals.retasks == 1
    if sim.decision_uav is None:
        sim.run_until_decision()
    step_to_done(sim)
    assert sim.targets[1].completed and sim.targets[0].completed


def test_redecision_continue_keeps_task_and_is_not_repeated():
    sim = make_sim(_retask_scenario(), retask_on_event=True)
    sim.assign(0)
    sim.run_until_decision()
    sim.assign(0)  # continue
    assert sim.totals.retasks_kept == 1 and sim.uavs[0].task_slot == 0
    assert not sim.retask_pending
    # the next decision is at T1's completion, not another re-decision at t = 100
    if not sim.done and sim.decision_uav is None:
        sim.run_until_decision()
    assert sim.t > 100.0


def test_redecision_must_choose_a_target():
    from uavta.environment.simulator import InvalidActionError

    sim = make_sim(_retask_scenario(), retask_on_event=True)
    sim.assign(0)
    sim.run_until_decision()
    with pytest.raises(InvalidActionError):
        sim.assign(None)


def test_no_redecision_without_retask_option():
    sim = make_sim(_retask_scenario())
    sim.assign(0)
    sim.run_until_decision()
    assert sim.t > 100.0  # no decision at the event: U1 stays committed (v1 rule J2a)
