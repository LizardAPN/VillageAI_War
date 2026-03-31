"""Observation layout for MAPPO: local bot vector + flattened global map + two village vectors."""

from __future__ import annotations

import numpy as np

from village_ai_war.agents.bot_obs_builder import BotObsBuilder
from village_ai_war.agents.village_obs_builder import VillageObsBuilder


def mappo_local_dim() -> int:
    return int(BotObsBuilder.OBS_DIM)


def mappo_map_flat(map_size: int) -> int:
    return int(map_size) * int(map_size) * int(VillageObsBuilder.N_CHANNELS)


def mappo_village_total() -> int:
    return int(VillageObsBuilder.VEC_DIM) * 2


def mappo_obs_dim(map_size: int) -> int:
    return mappo_local_dim() + mappo_map_flat(map_size) + mappo_village_total()


def pack_mappo_obs(
    local_obs: np.ndarray,
    map_obs: np.ndarray,
    village0: np.ndarray,
    village1: np.ndarray,
) -> np.ndarray:
    """Concatenate local bot obs, flattened map (team-0 POV), and both village vectors."""
    flat_map = np.asarray(map_obs, dtype=np.float32).reshape(-1)
    v = np.concatenate(
        [np.asarray(village0, dtype=np.float32), np.asarray(village1, dtype=np.float32)],
        axis=0,
    )
    loc = np.asarray(local_obs, dtype=np.float32).reshape(-1)
    return np.concatenate([loc, flat_map, v], axis=0).astype(np.float32, copy=False)
