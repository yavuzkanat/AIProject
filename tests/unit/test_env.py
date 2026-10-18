"""Gymnasium environment: API compliance, spaces, masks, reward identity, leakage."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from builders import LAM_OBSE, LAM_RISK, make_env, make_scenario
from uavta.environment.env import EnvConfig, MultiUAVDynamicEnv
from uavta.environment.observation import (
    GLOBAL_FEATURES,
    TARGET_FEATURE_NAMES,
    TARGET_FEATURES,
    UAV_FEATURES,
)
from uavta.environment.simulator import InvalidActionError


def test_gymnasium_check_env():
    check_env(MultiUAVDynamicEnv({"invalid_action_mode": "penalize"}), skip_render_check=True)


def test_sb3_check_env():
    from stable_baselines3.common.env_checker import check_env as sb3_check

    sb3_check(MultiUAVDynamicEnv({"invalid_action_mode": "penalize"}), warn=True)


def test_space_shapes():
    env = MultiUAVDynamicEnv()
    assert env.observation_space.shape == (
        6 * UAV_FEATURES + 24 * TARGET_FEATURES + GLOBAL_FEATURES,
    )
    assert env.action_space.n == 25


@pytest.mark.parametrize("seed", range(5))
def test_random_rollout_obs_in_space_and_mask_never_empty(seed):
    env = MultiUAVDynamicEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        assert mask.any()
        obs, r, te, tr, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert np.isfinite(r)
        done = te or tr
    assert "mission" in info


def test_reset_seed_reproducible():
    def rollout(seed):
        env = MultiUAVDynamicEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        done = False
        while not done:
            obs, r, te, tr, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
            out.append(obs)
            done = te or tr
        return np.stack(out)

    np.testing.assert_array_equal(rollout(7), rollout(7))
    assert rollout(7).shape != rollout(8).shape or not np.array_equal(rollout(7), rollout(8))


def test_reward_sum_identity_closed_form():
    env, _ = make_env()
    total = 0.0
    done = False
    while not done:
        _, r, te, tr, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
        total += r
        done = te or tr
    v = 50 * math.exp(-LAM_RISK * 1000 - LAM_OBSE * 100) + 80 * math.exp(
        -LAM_RISK * 3000 - LAM_OBSE * 310
    )
    expected = 0.5 * v / 130.0 - 0.5 * 330.0 / 2400.0
    assert total == pytest.approx(expected, abs=1e-9)
    m = info["mission"]
    assert m["episode_return"] == pytest.approx(expected, abs=1e-9)
    assert m["weighted_score"] == pytest.approx(0.5 * v / 130.0 - 0.5 * 330 / 2400, abs=1e-9)


def test_strict_mode_raises_on_masked_action():
    env, _ = make_env()
    with pytest.raises(InvalidActionError):
        env.step(env.wait_action)


def test_penalize_mode_penalizes_without_state_change():
    env, obs = make_env(invalid_action_mode="penalize")
    t_before = env.sim.t
    obs2, r, te, tr, info = env.step(env.wait_action)
    assert r == pytest.approx(-0.1)
    assert info["invalid_action"] and env.n_invalid == 1
    assert env.sim.t == t_before
    np.testing.assert_array_equal(obs, obs2)


def test_max_decisions_truncation():
    env, _ = make_env(invalid_action_mode="penalize", max_decisions=3)
    for _ in range(3):
        _, _, te, tr, info = env.step(env.wait_action)
    assert tr and not te and info["mission"]["termination_reason"] == "max_decisions"


def test_step_after_done_raises():
    env, _ = make_env()
    done = False
    while not done:
        _, _, te, tr, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        done = te or tr
    with pytest.raises(RuntimeError):
        env.step(0)


def test_no_targets_episode_wait_only():
    env, _ = make_env(make_scenario(targets=[]))
    mask = env.action_masks()
    assert mask.tolist() == [False] * 8 + [True]


def test_new_target_enters_fixed_slot_without_shape_change():
    sc = make_scenario(
        targets=[
            {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]},
            {
                "id": "N1",
                "value": 60,
                "observation_time": 5,
                "position": [0.0, 1000.0, 100.0],
                "initial": False,
            },
        ],
        events=[{"time": 40, "type": "new_target", "target": "N1"}],
    )
    env, obs0 = make_env(sc)
    tgt0 = 6 * UAV_FEATURES
    present = TARGET_FEATURE_NAMES.index("present")
    assert obs0[tgt0 + TARGET_FEATURES * 1 + present] == 0.0
    obs1, *_ = env.step(0)
    assert obs1.shape == obs0.shape
    assert obs1[tgt0 + TARGET_FEATURES * 1 + present] == 1.0


def test_no_future_event_leakage():
    """Same past, different future: observations must match until the first difference."""
    a = make_scenario(events=[{"time": 500, "type": "uav_crash", "uav": "U1"}])
    b = make_scenario(
        events=[{"time": 900, "type": "weather", "omega": {"VL": 0.3, "IR": 0.3, "RD": 0.3}}]
    )
    env_a, obs_a = make_env(a)
    env_b, obs_b = make_env(b)
    np.testing.assert_array_equal(obs_a, obs_b)
    obs_a, *_ = env_a.step(0)
    obs_b, *_ = env_b.step(0)
    assert env_a.sim.t == env_b.sim.t == pytest.approx(110.0)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_paper_source_config():
    env = MultiUAVDynamicEnv(EnvConfig(scenario={"source": "paper", "variant": "S2"}))
    env.reset(seed=0)
    assert env.scenario.name == "paper_S2" and len(env.sim.uavs) == 4


def test_too_many_uavs_rejected():
    env = MultiUAVDynamicEnv(EnvConfig(max_uavs=2))
    sc = make_scenario(
        uavs=[{"id": f"U{k}", "type": "A", "position": [0.0, 0.0, 100.0]} for k in range(3)]
    )
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"scenario": sc})
