"""Shared per-target scoring policy ("pointer-style" head), phase1 §D trade-off discussion.

The flat MLP must learn a separate mapping for each of the Z_max target slots. This policy
instead applies ONE small network to every target slot, conditioned on a context vector built
from the UAV block, the decision UAV and the global block, and uses the resulting scores as
action logits. It is equivariant to permutations of target slots and reuses what it learns
about one slot for all others. The WAIT logit comes from the context alone.

It plugs into ``sb3_contrib.MaskablePPO`` by replacing the MLP extractor; masking is unchanged
(masked logits are set to −∞ by the maskable categorical distribution).
"""

from __future__ import annotations

from typing import Any

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from uavta.environment.observation import GLOBAL_FEATURES, TARGET_FEATURES, UAV_FEATURES

UAV_PRESENT, UAV_IS_DECISION = 0, UAV_FEATURES - 1
TARGET_PRESENT = 0


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers)


class EntityExtractor(nn.Module):
    """Produces per-action logits (actor latent) and a pooled value latent (critic)."""

    def __init__(self, max_uavs: int, max_targets: int, hidden: int = 128) -> None:
        super().__init__()
        self.max_uavs, self.max_targets = max_uavs, max_targets
        self.uav_enc = _mlp([UAV_FEATURES, hidden, hidden])
        self.ctx = _mlp([2 * hidden + GLOBAL_FEATURES, hidden, hidden])
        self.tgt_enc = _mlp([TARGET_FEATURES + hidden, hidden, hidden])
        self.tgt_score = nn.Linear(hidden, 1)
        self.wait_score = nn.Linear(hidden, 1)
        self.value_head = _mlp([2 * hidden, hidden, hidden])
        self.latent_dim_pi = max_targets + 1
        self.latent_dim_vf = hidden

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = obs.shape[0]
        nu = self.max_uavs * UAV_FEATURES
        nt = self.max_targets * TARGET_FEATURES
        uav = obs[:, :nu].reshape(b, self.max_uavs, UAV_FEATURES)
        tgt = obs[:, nu : nu + nt].reshape(b, self.max_targets, TARGET_FEATURES)
        return uav, tgt, obs[:, nu + nt :]

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uav, tgt, glob = self._split(obs)
        u_emb = self.uav_enc(uav)
        present = uav[..., UAV_PRESENT : UAV_PRESENT + 1]
        pooled = (u_emb * present).sum(1) / present.sum(1).clamp(min=1.0)
        dec = uav[..., UAV_IS_DECISION : UAV_IS_DECISION + 1]
        dec_emb = (u_emb * dec).sum(1)
        context = self.ctx(torch.cat([pooled, dec_emb, glob], dim=-1))
        ctx_rep = context.unsqueeze(1).expand(-1, self.max_targets, -1)
        t_emb = self.tgt_enc(torch.cat([tgt, ctx_rep], dim=-1))
        return context, t_emb, tgt[..., TARGET_PRESENT : TARGET_PRESENT + 1]

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        context, t_emb, _ = self._encode(obs)
        scores = self.tgt_score(t_emb).squeeze(-1)
        return torch.cat([scores, self.wait_score(context)], dim=-1)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        context, t_emb, present = self._encode(obs)
        pooled = (t_emb * present).sum(1) / present.sum(1).clamp(min=1.0)
        return self.value_head(torch.cat([context, pooled], dim=-1))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)


class AttentionExtractor(EntityExtractor):
    """Entity extractor whose target embeddings also attend to each other and to the UAVs.

    The plain entity scorer rates every target independently given a pooled context, so it
    cannot express relations such as "these two targets are close together" or "another UAV is
    about to be free next to this target". Here UAV and target tokens pass through
    ``attention_layers`` Transformer encoder layers (padded slots masked out) before scoring.
    Target-slot permutation equivariance is preserved.
    """

    def __init__(
        self, max_uavs: int, max_targets: int, hidden: int = 128, layers: int = 2, heads: int = 4
    ) -> None:
        super().__init__(max_uavs, max_targets, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.token_type = nn.Parameter(torch.zeros(2, hidden))  # 0 = UAV token, 1 = target token

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uav, tgt, glob = self._split(obs)
        u_emb = self.uav_enc(uav)
        u_present = uav[..., UAV_PRESENT : UAV_PRESENT + 1]
        pooled = (u_emb * u_present).sum(1) / u_present.sum(1).clamp(min=1.0)
        dec = uav[..., UAV_IS_DECISION : UAV_IS_DECISION + 1]
        dec_emb = (u_emb * dec).sum(1)
        context = self.ctx(torch.cat([pooled, dec_emb, glob], dim=-1))
        ctx_rep = context.unsqueeze(1).expand(-1, self.max_targets, -1)
        t_emb = self.tgt_enc(torch.cat([tgt, ctx_rep], dim=-1))
        t_present = tgt[..., TARGET_PRESENT : TARGET_PRESENT + 1]
        tokens = torch.cat([u_emb + self.token_type[0], t_emb + self.token_type[1]], dim=1)
        ignore = torch.cat([u_present, t_present], dim=1).squeeze(-1) < 0.5
        # A sequence with every token masked would produce NaNs; keep one token visible.
        empty = ignore.all(dim=1)
        if empty.any():
            ignore = ignore.clone()
            ignore[empty, 0] = False
        out = self.attn(tokens, src_key_padding_mask=ignore)
        t_out = t_emb + out[:, self.max_uavs :]  # residual keeps the plain entity signal
        return context, t_out, t_present


class EntityMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy whose action logits are the shared per-target scores."""

    def __init__(
        self,
        *args: Any,
        max_uavs: int = 6,
        max_targets: int = 24,
        entity_hidden: int = 128,
        attention_layers: int = 0,
        attention_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        self._entity_cfg = (max_uavs, max_targets, entity_hidden)
        self._attention_cfg = (attention_layers, attention_heads)
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        max_uavs, max_targets, hidden = self._entity_cfg
        layers, heads = getattr(self, "_attention_cfg", (0, 4))
        extractor: EntityExtractor = (
            AttentionExtractor(max_uavs, max_targets, hidden, layers, heads)
            if layers > 0
            else EntityExtractor(max_uavs, max_targets, hidden)
        )
        self.mlp_extractor = extractor  # type: ignore[assignment]

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        # Scores are already logits; drop the extra linear layer so the head stays shared.
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
