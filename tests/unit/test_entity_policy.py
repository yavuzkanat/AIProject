"""Shared per-target scoring policy: shapes, masking compatibility, permutation equivariance."""

import numpy as np
import torch

from uavta.agents.entity_policy import EntityExtractor
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.environment.observation import TARGET_FEATURES, UAV_FEATURES


def _obs_batch(n=4):
    env = MultiUAVDynamicEnv()
    obs = [env.reset(seed=s)[0] for s in range(n)]
    return torch.as_tensor(np.stack(obs))


def test_shapes():
    ext = EntityExtractor(6, 24)
    obs = _obs_batch()
    pi, vf = ext(obs)
    assert pi.shape == (4, 25) and vf.shape == (4, ext.latent_dim_vf)


def test_target_permutation_equivariance():
    torch.manual_seed(0)
    ext = EntityExtractor(6, 24).eval()
    obs = _obs_batch(1)
    start = 6 * UAV_FEATURES
    tgt = obs[:, start : start + 24 * TARGET_FEATURES].reshape(1, 24, TARGET_FEATURES)
    perm = torch.randperm(24)
    obs_p = obs.clone()
    obs_p[:, start : start + 24 * TARGET_FEATURES] = tgt[:, perm].reshape(1, -1)
    with torch.no_grad():
        logits, logits_p = ext.forward_actor(obs), ext.forward_actor(obs_p)
        v, v_p = ext.forward_critic(obs), ext.forward_critic(obs_p)
    torch.testing.assert_close(logits_p[0, :24], logits[0, :24][perm])
    torch.testing.assert_close(logits_p[0, 24], logits[0, 24])
    torch.testing.assert_close(v_p, v, rtol=1e-5, atol=1e-6)


def test_maskable_ppo_with_entity_policy_respects_mask():
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from uavta.agents.ppo_factory import PPOConfig, make_model

    vec = DummyVecEnv([lambda: MultiUAVDynamicEnv()])
    model = make_model(
        vec, PPOConfig(policy="entity", n_steps=64, batch_size=32), seed=0, device="cpu"
    )
    assert isinstance(model, MaskablePPO)
    model.learn(128)
    env = MultiUAVDynamicEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(5):
        mask = env.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=False)
        assert mask[int(a)]
        obs = env.step(int(a))[0]
