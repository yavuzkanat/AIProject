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


def test_attention_extractor_equivariant_and_masks_padding():
    from uavta.agents.entity_policy import AttentionExtractor

    torch.manual_seed(0)
    ext = AttentionExtractor(6, 24, hidden=32, layers=2, heads=4).eval()
    obs = _obs_batch(2)
    start = 6 * UAV_FEATURES
    tgt = obs[:, start : start + 24 * TARGET_FEATURES].reshape(2, 24, TARGET_FEATURES)
    perm = torch.randperm(24)
    obs_p = obs.clone()
    obs_p[:, start : start + 24 * TARGET_FEATURES] = tgt[:, perm].reshape(2, -1)
    with torch.no_grad():
        a, b = ext.forward_actor(obs), ext.forward_actor(obs_p)
        va, vb = ext.forward_critic(obs), ext.forward_critic(obs_p)
    torch.testing.assert_close(b[:, :24], a[:, :24][:, perm], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(vb, va, rtol=1e-4, atol=1e-5)
    # Changing a padded (absent) target slot must not change any present slot's score.
    present = tgt[0, :, 0] > 0.5
    absent = int(torch.nonzero(~present)[0])
    obs_q = obs.clone()
    obs_q[0, start + absent * TARGET_FEATURES + 3] = 0.77
    with torch.no_grad():
        c = ext.forward_actor(obs_q)
    torch.testing.assert_close(c[0, :24][present], a[0, :24][present])


def test_attention_handles_observation_without_targets():
    from uavta.agents.entity_policy import AttentionExtractor

    ext = AttentionExtractor(6, 24, hidden=32, layers=1, heads=4)
    obs = torch.zeros(1, 6 * UAV_FEATURES + 24 * TARGET_FEATURES + 17)
    pi, vf = ext(obs)
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()


def test_attention_policy_trains_saves_and_loads(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from uavta.agents.ppo_factory import PPOConfig, PPOPolicy, make_model

    vec = DummyVecEnv([lambda: MultiUAVDynamicEnv()])
    cfg = PPOConfig(policy="attention", n_steps=64, batch_size=32, attention_layers=1)
    model = make_model(vec, cfg, seed=0, device="cpu")
    model.learn(64)
    model.save(tmp_path / "m.zip")
    pol = PPOPolicy.load(tmp_path / "m.zip")
    env = MultiUAVDynamicEnv()
    obs, _ = env.reset(seed=1)
    assert env.action_masks()[pol.act(env, obs, env.action_masks())]
