"""Role-conditioned actor-critic policy (shared backbone + role embedding)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from village_ai_war.agents.bot_obs_builder import BotObsBuilder

# Role one-hot in BotObsBuilder: indices 98:102; backbone uses the rest (177 dims).
_ROLE_START = 98
_ROLE_END = 102
_BACKBONE_DIM = _ROLE_START + (BotObsBuilder.OBS_DIM - _ROLE_END)


class RoleConditionedExtractor(BaseFeaturesExtractor):
    """Shared MLP backbone plus learned role embedding (from one-hot slice).

    Observation layout matches ``BotObsBuilder``: terrain/resources patches and
    scalars except the role one-hot at ``[98:102]``, which is converted to a
    discrete role id for ``nn.Embedding``.

    Args:
        observation_space: Box observation for a single bot.
        features_dim: Output size of the extractor (input to policy/value heads).
        backbone_hidden: Hidden width of the shared trunk.
        role_embed_dim: Embedding size per role.
        n_roles: Number of discrete roles (default ``4``).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        backbone_hidden: int = 128,
        role_embed_dim: int = 16,
        n_roles: int = 4,
    ) -> None:
        obs_dim = int(np.prod(observation_space.shape))
        if obs_dim != BotObsBuilder.OBS_DIM:
            raise ValueError(
                f"Expected observation dim {BotObsBuilder.OBS_DIM}, got {obs_dim}"
            )
        super().__init__(observation_space, features_dim)

        backbone_in = _BACKBONE_DIM
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, backbone_hidden),
            nn.LayerNorm(backbone_hidden),
            nn.ReLU(),
            nn.Linear(backbone_hidden, backbone_hidden),
            nn.LayerNorm(backbone_hidden),
            nn.ReLU(),
        )
        self.role_embedding = nn.Embedding(n_roles, role_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(backbone_hidden + role_embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Concatenate backbone features with role embedding."""
        role_oh = observations[:, _ROLE_START:_ROLE_END]
        role_ids = role_oh.argmax(dim=1).clamp(0, self.role_embedding.num_embeddings - 1)
        rest_before = observations[:, :_ROLE_START]
        rest_after = observations[:, _ROLE_END:]
        obs_no_role = torch.cat([rest_before, rest_after], dim=1)
        hidden = self.backbone(obs_no_role)
        role_vec = self.role_embedding(role_ids)
        combined = torch.cat([hidden, role_vec], dim=1)
        return self.head(combined)


class RoleConditionedPolicy(ActorCriticPolicy):
    """PPO policy using :class:`RoleConditionedExtractor` as feature extractor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", RoleConditionedExtractor)
        kwargs.setdefault(
            "features_extractor_kwargs",
            {
                "features_dim": 256,
                "backbone_hidden": 128,
                "role_embed_dim": 16,
                "n_roles": 4,
            },
        )
        super().__init__(*args, **kwargs)
