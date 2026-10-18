"""Run Gymnasium's and Stable-Baselines3's environment checkers and one paper episode.

Example:
    python scripts/check_env.py
"""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from uavta.baselines.heuristics import GreedyValueRatePolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.runner import ScenarioRef, run_episode
from uavta.scenarios.loader import load_paper_scenario


def main() -> None:
    # The checkers sample unmasked actions, so they run in penalize mode (docs/assumptions.md N9).
    check_env(MultiUAVDynamicEnv({"invalid_action_mode": "penalize"}), skip_render_check=True)
    sb3_check_env(MultiUAVDynamicEnv({"invalid_action_mode": "penalize"}), warn=True)
    env = MultiUAVDynamicEnv()
    print(
        "environment checks passed; observation",
        env.observation_space.shape,
        "action",
        env.action_space,
    )
    rec = run_episode(
        env, GreedyValueRatePolicy(), ScenarioRef("S3", scenario=load_paper_scenario("S3"))
    )
    keys = ("weighted_score", "value_fraction", "makespan", "completion_rate", "termination_reason")
    print("greedy on paper S3:", {k: rec.metrics[k] for k in keys})


if __name__ == "__main__":
    main()
