"""Lookahead never uses future events and only returns valid actions."""

import numpy as np
import pytest

from builders import make_env, make_scenario
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.scenarios.generator import ScenarioGenerator


class FirstValid:
    """Stand-in for PPOPolicy: prefers the lowest valid slot."""

    name = "first"

    def reset(self, env, seed=None):
        return None

    def act(self, env, obs, mask):
        return int(np.flatnonzero(mask)[0])

    def action_probabilities(self, obs, mask):
        p = np.where(mask, 1.0 / (1 + np.arange(mask.size)), 0.0)
        return p / p.sum()


def test_clone_without_future_drops_only_future_events():
    env = MultiUAVDynamicEnv()
    sc = ScenarioGenerator().generate(7)
    env.reset(seed=0, options={"scenario": sc})
    while env.sim.t < 200 and not env.sim.done:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    sim = env.sim
    clone = sim.clone_without_future()
    assert not clone.has_pending_events()
    assert len(clone._events) == sim._next_event
    assert clone.uavs[0] is not sim.uavs[0] and clone.scenario is sim.scenario
    clone.uavs[0].position[0] = -1.0
    assert sim.uavs[0].position[0] != -1.0


def test_rollout_does_not_change_the_real_episode():
    from uavta.agents.lookahead import LookaheadPolicy

    env, obs = make_env()
    before = (env.sim.t, env.sim.totals.collected_value, env.sim.decision_uav)
    LookaheadPolicy(FirstValid(), top_k=2).rollout_return(env, 1)
    assert (env.sim.t, env.sim.totals.collected_value, env.sim.decision_uav) == before


def test_lookahead_is_blind_to_future_events():
    """Same past, different future: the lookahead must make identical decisions."""
    from uavta.agents.lookahead import LookaheadPolicy

    future_a = [{"time": 900, "type": "uav_crash", "uav": "U1"}]
    future_b = [{"time": 900, "type": "weather", "omega": {"VL": 0.2, "IR": 0.2, "RD": 0.2}}]
    env_a, obs_a = make_env(make_scenario(events=future_a))
    env_b, obs_b = make_env(make_scenario(events=future_b))
    pol = LookaheadPolicy(FirstValid(), top_k=2)
    assert pol.act(env_a, obs_a, env_a.action_masks()) == pol.act(
        env_b, obs_b, env_b.action_masks()
    )
    assert pol.rollout_return(env_a, 1) == pytest.approx(pol.rollout_return(env_b, 1))


def test_lookahead_picks_better_order_and_valid_actions():
    from uavta.agents.lookahead import LookaheadPolicy

    env, obs = make_env()
    pol = LookaheadPolicy(FirstValid(), top_k=2)
    # Going to the far target first (slot 1) is worse than the near one (slot 0).
    assert pol.rollout_return(env, 0) > pol.rollout_return(env, 1)
    done = False
    while not done:
        mask = env.action_masks()
        a = pol.act(env, obs, mask)
        assert mask[a]
        obs, _, te, tr, _ = env.step(a)
        done = te or tr


def test_top_k_validation():
    from uavta.agents.lookahead import LookaheadPolicy

    with pytest.raises(ValueError):
        LookaheadPolicy(FirstValid(), top_k=0)
