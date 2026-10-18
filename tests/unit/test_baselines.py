"""Baseline policies produce valid, reproducible decisions."""

import pytest

from builders import make_env
from uavta.baselines.heuristics import GreedyNearestPolicy, GreedyValueRatePolicy, RandomPolicy
from uavta.baselines.rh_rk import RHRKConfig, RollingHorizonRandomKeyPolicy, _Problem
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.runner import ScenarioRef, run_episode
from uavta.scenarios.loader import load_paper_scenario

POLICIES = [
    lambda: RandomPolicy(seed=0),
    GreedyValueRatePolicy,
    GreedyNearestPolicy,
    lambda: RollingHorizonRandomKeyPolicy(RHRKConfig(population=10, iterations=5)),
]


@pytest.mark.parametrize("make_policy", POLICIES)
@pytest.mark.parametrize("variant", ["S1", "S2", "S3"])
def test_policies_complete_paper_scenarios(make_policy, variant):
    env = MultiUAVDynamicEnv()
    rec = run_episode(
        env,
        make_policy(),
        ScenarioRef(variant, scenario=load_paper_scenario(variant)),
        policy_seed=1,
    )
    assert rec.metrics["n_invalid_actions"] == 0
    assert rec.metrics["termination_reason"] in {
        "all_targets_completed",
        "no_feasible_target",
        "no_operational_uav",
        "horizon",
    }


@pytest.mark.parametrize("make_policy", POLICIES)
def test_policies_reproducible(make_policy):
    env = MultiUAVDynamicEnv()
    ref = ScenarioRef("s", seed=42)
    a = run_episode(env, make_policy(), ref, policy_seed=3).metrics
    b = run_episode(env, make_policy(), ref, policy_seed=3).metrics
    for key in ("collected_value", "makespan", "n_decisions"):
        assert a[key] == b[key]


def test_greedy_value_rate_picks_best_ratio():
    env, obs = make_env()
    # T1: closer (110 s task) vs T2 (320 s task): T1 has the higher value rate.
    assert GreedyValueRatePolicy().act(env, obs, env.action_masks()) == 0


def test_greedy_nearest():
    env, obs = make_env()
    assert GreedyNearestPolicy().act(env, obs, env.action_masks()) == 0


def test_rh_rk_seeds_are_complete_plans():
    env = MultiUAVDynamicEnv()
    env.reset(seed=0, options={"scenario": load_paper_scenario("S3")})
    prob = _Problem(env)
    for plan in (prob.dispatch_greedy_seed(), prob.greedy_seed(), prob.round_robin_seed()):
        assigned = sorted(j for seq in plan for j in seq)
        assert assigned == list(range(prob.n))
        assert prob.decode(prob.encode(plan)) == plan


def test_rh_rk_encoding_decodes_back():
    env = MultiUAVDynamicEnv()
    env.reset(seed=0, options={"scenario": load_paper_scenario("S1")})
    prob = _Problem(env)
    plan = prob.round_robin_seed()
    assert prob.decode(prob.encode(plan)) == plan
