"""Short MaskablePPO training and evaluation runs end to end."""

import pytest

from uavta.agents.ppo_factory import PPOPolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.runner import evaluate_policy, seed_refs
from uavta.scenarios.generator import VALIDATION_SEED_BASE
from uavta.training.train import TrainConfig, train


@pytest.mark.parametrize("use_masking", [True, False])
def test_short_training_and_evaluation(tmp_path, use_masking):
    cfg = TrainConfig(
        name="it",
        total_timesteps=1024,
        n_envs=2,
        vec_env="dummy",
        eval_freq=512,
        n_validation_scenarios=3,
        env={"invalid_action_mode": "strict" if use_masking else "penalize"},
        ppo={"n_steps": 256, "batch_size": 64, "n_epochs": 2, "use_masking": use_masking},
    )
    run_dir = train(cfg, tmp_path)
    for name in (
        "final_model.zip",
        "best_model.zip",
        "config.yaml",
        "versions.json",
        "validation.csv",
        "training_summary.json",
        "progress.csv",
    ):
        assert (run_dir / name).exists(), name
    policy = PPOPolicy.load(run_dir / "final_model.zip")
    assert policy.use_masking == use_masking
    df = evaluate_policy(MultiUAVDynamicEnv(), policy, seed_refs(VALIDATION_SEED_BASE, 3))
    assert (df["n_invalid_actions"] == 0).all()
    assert len(df) == 3
