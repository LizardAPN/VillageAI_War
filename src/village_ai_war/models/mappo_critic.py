"""Centralized MAPPO critic: global map tensor + both village vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class MAPPOCentralizedCritic(nn.Module):
    """Value network over full map (CNN) and concatenated village state (MLP)."""

    def __init__(
        self,
        map_shape: tuple[int, int, int],
        village_vec_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        n, n2, c = map_shape
        if n != n2:
            raise ValueError("Map must be square")
        self.map_shape = map_shape

        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        map_out_dim = 64 * 4 * 4

        self.village_encoder = nn.Sequential(
            nn.Linear(village_vec_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(map_out_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, map_obs: torch.Tensor, village_obs: torch.Tensor) -> torch.Tensor:
        # map_obs: (B, H, W, C) -> (B, C, H, W)
        x = map_obs.permute(0, 3, 1, 2)
        map_features = self.map_encoder(x)
        village_features = self.village_encoder(village_obs)
        combined = torch.cat([map_features, village_features], dim=1)
        return self.value_head(combined)
