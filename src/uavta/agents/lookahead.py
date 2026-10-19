"""Decision-time lookahead on top of a trained policy (improvement 1).

At each decision the policy's ``top_k`` most probable valid actions are each tried in a copy of
the simulator from which all future events have been removed
(:meth:`Simulator.clone_without_future`), and the rest of the mission is played out with the
same policy acting greedily. The action whose rollout earns the highest return under the
environment's own reward weights is executed.

No information about the realized future is used. With ``samples = 0`` the rollouts assume no
further disruptions (like RH-RK). With ``samples = N`` every candidate is instead evaluated on the
same N hypothetical futures drawn from the *known event statistics* of the training
distribution (``ScenarioGenerator.sample_future_events``), and the returns are averaged
(common random numbers). The policy itself is unchanged (no retraining).
"""

from __future__ import annotations

import numpy as np

from uavta.agents.ppo_factory import PPOPolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.environment.masking import action_mask
from uavta.environment.observation import build_observation
from uavta.environment.simulator import Simulator
from uavta.events.types import Event, sort_events


class LookaheadPolicy:
    """Wraps a :class:`PPOPolicy` with ``top_k`` full-episode rollouts per decision."""

    def __init__(
        self, base: PPOPolicy, top_k: int = 3, samples: int = 0, name: str | None = None
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if samples < 0:
            raise ValueError("samples must be >= 0")
        self.base = base
        self.top_k = top_k
        self.samples = samples
        self.name = name or f"{base.name}_lookahead"
        self.rollouts = 0
        self._rng = np.random.default_rng(0)

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        self.base.reset(env, seed)
        self._rng = np.random.default_rng(seed)

    def _greedy(self, env: MultiUAVDynamicEnv, sim: Simulator) -> int:
        mask = action_mask(sim, env.config.max_targets)
        valid = np.flatnonzero(mask)
        if valid.size == 1:
            return int(valid[0])
        obs = build_observation(sim, env.obs_spec)
        return self.base.act(env, obs, mask)

    def _futures(self, env: MultiUAVDynamicEnv) -> list[list[Event]]:
        if self.samples == 0:
            return [[]]
        sim = env.sim
        assert sim is not None and sim.max_targets is not None
        free = sim.max_targets - len(sim.targets)
        return [
            env.generator.sample_future_events(self._rng, after=sim.t, free_slots=free)
            for _ in range(self.samples)
        ]

    def rollout_return(
        self, env: MultiUAVDynamicEnv, action: int, future: list[Event] | None = None
    ) -> float:
        """Return of executing ``action`` now and following the base policy afterwards, in a
        copy of the current state whose future is ``future`` (default: no further events)."""
        assert env.sim is not None
        sim = env.sim.clone_without_future()
        if future:
            sim._events = sim._events + sort_events(list(future))
        value0, active0 = sim.totals.collected_value, sim.totals.active_time
        w = env.config.reward
        a = action
        for _ in range(env.config.max_decisions):
            sim.assign(None if a == env.wait_action else a)
            if not sim.done and sim.decision_uav is None:
                sim.run_until_decision()
            if sim.done:
                break
            a = self._greedy(env, sim)
        self.rollouts += 1
        return (
            w.w_value * (sim.totals.collected_value - value0) / env.value_ref
            - w.w_time * (sim.totals.active_time - active0) / env.config.horizon
        )

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 1:
            return int(valid[0])
        probs = self.base.action_probabilities(obs, mask)
        candidates = sorted(valid.tolist(), key=lambda a: -probs[a])[: self.top_k]
        if len(candidates) == 1:
            return int(candidates[0])
        futures = self._futures(env)
        scored = [
            (float(np.mean([self.rollout_return(env, a, f) for f in futures])), probs[a], a)
            for a in candidates
        ]
        # Highest rollout return; ties go to the more probable action.
        return int(max(scored)[2])
