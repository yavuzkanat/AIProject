"""MaskablePPO construction and a policy wrapper for evaluation.

Uses ``sb3_contrib.MaskablePPO``: the categorical distribution over Z_max + 1 actions sets the
logits of masked actions to −∞, so invalid actions have probability zero during both rollout
and evaluation. Masks are obtained from the environment's ``action_masks()`` method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from uavta.environment.env import MultiUAVDynamicEnv


class PPOConfig(BaseModel):
    """PPO hyper-parameters (phase1 §G.4). Tuned only on the validation split."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(3e-4, gt=0)
    linear_lr_decay: bool = True
    n_steps: int = Field(1024, gt=0)
    batch_size: int = Field(256, gt=0)
    n_epochs: int = Field(10, gt=0)
    gamma: float = Field(0.995, gt=0, le=1)
    gae_lambda: float = Field(0.95, gt=0, le=1)
    clip_range: float = Field(0.2, gt=0)
    ent_coef: float = Field(0.01, ge=0)
    vf_coef: float = Field(0.5, ge=0)
    max_grad_norm: float = Field(0.5, gt=0)
    net_arch_pi: list[int] = Field(default_factory=lambda: [256, 256])
    net_arch_vf: list[int] = Field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    use_masking: bool = True
    policy: str = "mlp"  # "mlp" (flat MLP) | "entity" (shared per-target scorer) | "attention"
    entity_hidden: int = Field(128, gt=0)
    attention_layers: int = Field(2, gt=0)
    attention_heads: int = Field(4, gt=0)


def make_model(
    vec_env: Any, config: PPOConfig, *, seed: int, device: str, tensorboard_log: str | None = None
) -> Any:
    """Create a MaskablePPO model (or plain PPO when ``use_masking`` is False)."""
    import torch.nn as nn
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO

    activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[config.activation]
    lr: Any = config.learning_rate
    if config.linear_lr_decay:
        base = config.learning_rate

        def lr(progress_remaining: float) -> float:
            return base * progress_remaining

    algo = MaskablePPO if config.use_masking else PPO
    policy: Any = "MlpPolicy"
    policy_kwargs: dict[str, Any] = {
        "net_arch": {"pi": config.net_arch_pi, "vf": config.net_arch_vf},
        "activation_fn": activation,
    }
    if config.policy in ("entity", "attention"):
        if not config.use_masking:
            raise ValueError("the entity policy is implemented for MaskablePPO only")
        from uavta.agents.entity_policy import EntityMaskablePolicy

        env_cfg = vec_env.get_attr("config")[0]
        policy = EntityMaskablePolicy
        policy_kwargs = {
            "max_uavs": env_cfg.max_uavs,
            "max_targets": env_cfg.max_targets,
            "entity_hidden": config.entity_hidden,
        }
        if config.policy == "attention":
            policy_kwargs["attention_layers"] = config.attention_layers
            policy_kwargs["attention_heads"] = config.attention_heads
    elif config.policy != "mlp":
        raise ValueError(f"unknown policy {config.policy!r}")
    return algo(
        policy,
        vec_env,
        learning_rate=lr,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=0,
    )


class PPOPolicy:
    """Deterministic (argmax) evaluation wrapper implementing ``AllocationPolicy``.

    Masking is applied at evaluation for masked models. For an unmasked model, a masked-out
    argmax is replaced by the most probable valid action and counted in ``mask_overrides``, so
    the evaluation never executes an invalid action silently.
    """

    def __init__(self, model: Any, name: str = "ppo", use_masking: bool = True) -> None:
        self.model = model
        self.name = name
        self.use_masking = use_masking
        self.mask_overrides = 0

    @classmethod
    def load(cls, path: str | Path, name: str = "ppo", device: str = "cpu") -> PPOPolicy:
        from sb3_contrib import MaskablePPO
        from stable_baselines3 import PPO

        try:
            return cls(MaskablePPO.load(str(path), device=device), name=name, use_masking=True)
        except (ValueError, KeyError, TypeError):
            return cls(PPO.load(str(path), device=device), name=name, use_masking=False)

    def reset(self, env: MultiUAVDynamicEnv, seed: int | None = None) -> None:
        return None

    def action_probabilities(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Policy probabilities over all actions; masked actions get probability 0."""
        import torch

        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            if self.use_masking:
                dist = self.model.policy.get_distribution(obs_t, action_masks=mask)
            else:
                dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs[0].cpu().numpy().astype(np.float64)
        probs[~mask] = 0.0
        return probs

    def act(self, env: MultiUAVDynamicEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.use_masking:
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        action, _ = self.model.predict(obs, deterministic=True)
        if mask[int(action)]:
            return int(action)
        import torch

        self.mask_overrides += 1
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            probs = self.model.policy.get_distribution(obs_t).distribution.probs[0].cpu().numpy()
        probs[~mask] = -1.0
        return int(np.argmax(probs))
