"""MAPPO actor feature extractor (local bot obs only; ignores concatenated global tail)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from village_ai_war.agents.bot_obs_builder import BotObsBuilder
from village_ai_war.models.mappo_layout import mappo_local_dim

# Role one-hot in BotObsBuilder: indices 98:102 (same as RoleConditionedExtractor).
_ROLE_START = 98
_ROLE_END = 102
_BACKBONE_DIM = _ROLE_START + (BotObsBuilder.OBS_DIM - _ROLE_END)


class MAPPOActorExtractor(BaseFeaturesExtractor):
    """Role-conditioned trunk on the first ``K * local_dim`` components (K bot slots)."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        backbone_hidden: int = 128,
        role_embed_dim: int = 16,
        n_roles: int = 4,
        local_dim: int | None = None,
        n_bot_slots: int = 1,
    ) -> None:
        obs_dim = int(np.prod(observation_space.shape))
        ld = int(local_dim) if local_dim is not None else mappo_local_dim()
        k = int(n_bot_slots)
        if k < 1:
            raise ValueError("n_bot_slots must be >= 1")
        if obs_dim < k * ld:
            raise ValueError(f"Expected observation dim >= {k * ld}, got {obs_dim}")
        per_bot = int(features_dim)
        super().__init__(observation_space, k * per_bot)
        self._local_dim = ld
        self._n_bot_slots = k
        self._per_bot_features = per_bot

        self.backbone = nn.Sequential(
            nn.Linear(_BACKBONE_DIM, backbone_hidden),
            nn.LayerNorm(backbone_hidden),
            nn.ReLU(),
            nn.Linear(backbone_hidden, backbone_hidden),
            nn.LayerNorm(backbone_hidden),
            nn.ReLU(),
        )
        self.role_embedding = nn.Embedding(n_roles, role_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(backbone_hidden + role_embed_dim, per_bot),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        k = self._n_bot_slots
        ld = self._local_dim
        bsz = observations.shape[0]
        bk = bsz * k
        local = observations[:, : k * ld].reshape(bk, ld)
        role_oh = local[:, _ROLE_START:_ROLE_END]
        role_ids = role_oh.argmax(dim=1).clamp(0, self.role_embedding.num_embeddings - 1)
        rest_before = local[:, :_ROLE_START]
        rest_after = local[:, _ROLE_END:]
        obs_no_role = torch.cat([rest_before, rest_after], dim=1)
        hidden = self.backbone(obs_no_role)
        role_vec = self.role_embedding(role_ids)
        feat = self.head(torch.cat([hidden, role_vec], dim=1))
        return feat.reshape(bsz, k * self._per_bot_features)
