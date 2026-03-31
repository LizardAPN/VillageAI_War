"""MAPPO policy: decentralized actor features + centralized critic on global tail."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from village_ai_war.models.mappo_actor import MAPPOActorExtractor
from village_ai_war.models.mappo_critic import MAPPOCentralizedCritic
from village_ai_war.models.mappo_layout import mappo_local_dim, mappo_map_flat, mappo_village_total


class MAPPOPolicy(ActorCriticPolicy):
    """PPO policy with actor on local bot obs and critic on global map + villages."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        map_size: int,
        critic_hidden_dim: int = 256,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._mappo_map_size = int(map_size)
        self._local_dim = mappo_local_dim()
        self._map_flat = mappo_map_flat(self._mappo_map_size)
        self._village_total = mappo_village_total()
        tail = self._map_flat + self._village_total
        expected = self._local_dim + tail
        if isinstance(observation_space, spaces.Box):
            od = int(np.prod(observation_space.shape))
            if od != expected:
                raise ValueError(
                    f"MAPPOPolicy expects observation dim {expected} "
                    f"(local {self._local_dim} + global {tail}), got {od}"
                )

        kwargs.setdefault("features_extractor_class", MAPPOActorExtractor)
        kwargs.setdefault(
            "features_extractor_kwargs",
            {
                "features_dim": 128,
                "backbone_hidden": 128,
                "role_embed_dim": 16,
                "n_roles": 4,
                "local_dim": self._local_dim,
            },
        )
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        map_shape = (self._mappo_map_size, self._mappo_map_size, 6)
        self.centralized_critic = MAPPOCentralizedCritic(
            map_shape=map_shape,
            village_vec_dim=self._village_total,
            hidden_dim=int(critic_hidden_dim),
        )
        self.centralized_critic.to(self.device)

    def _global_tensors(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        rest = obs[:, self._local_dim :]
        map_flat = rest[:, : self._map_flat]
        vil = rest[:, self._map_flat :]
        n = self._mappo_map_size
        map_b = map_flat.reshape(-1, n, n, 6)
        return map_b, vil

    def _centralized_values(self, obs: th.Tensor) -> th.Tensor:
        map_b, vil = self._global_tensors(obs)
        return self.centralized_critic(map_b, vil)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[union-attr]
        values = self._centralized_values(obs)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        obs_t = obs if isinstance(obs, th.Tensor) else th.as_tensor(obs, device=self.device)
        features = self.extract_features(obs_t)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._centralized_values(obs_t)
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        obs_t = obs if isinstance(obs, th.Tensor) else th.as_tensor(obs, device=self.device)
        return self._centralized_values(obs_t)
