"""MAPPO observation layout shared by MAPPOBotEnv and human-vs-MAPPO play."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from village_ai_war.agents.village_obs_builder import VillageObsBuilder
from village_ai_war.models.mappo_layout import pack_mappo_obs_slots
from village_ai_war.state.game_state import GameState

if TYPE_CHECKING:
    from village_ai_war.env.game_env import GameEnv


def build_mappo_global_state(state: GameState, vil_obs: VillageObsBuilder) -> dict[str, np.ndarray]:
    """Team-0 map POV plus concatenated village vectors (order matches MAPPO critic)."""
    mp = vil_obs.build_map(state, team=0)
    v0 = vil_obs.build_village_vec(state, team=0)
    v1 = vil_obs.build_village_vec(state, team=1)
    return {"map": mp, "village": np.concatenate([v0, v1], axis=0).astype(np.float32)}


def build_mappo_locals_matrix(
    state: GameState,
    game_env: GameEnv,
    *,
    mappo_team: int,
    n_bot_slots: int,
) -> np.ndarray:
    """K × local_dim; alive bots on ``mappo_team`` sorted by ``bot_id`` fill low rows."""
    k = int(n_bot_slots)
    d = int(game_env.observation_space.shape[0])
    out = np.zeros((k, d), dtype=np.float32)
    village = state.villages[int(mappo_team)]
    alive = sorted((b for b in village.bots if b.is_alive), key=lambda b: int(b.bot_id))
    for i, bot in enumerate(alive[:k]):
        out[i] = game_env._get_single_bot_obs(bot.bot_id, int(mappo_team))
    return out


def pack_mappo_observation_vector(locals_k: np.ndarray, gs: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten packed MAPPO vector (K locals + map + two village vecs)."""
    v = gs["village"]
    half = VillageObsBuilder.VEC_DIM
    return pack_mappo_obs_slots(
        locals_k,
        gs["map"],
        v[:half],
        v[half:],
    )
