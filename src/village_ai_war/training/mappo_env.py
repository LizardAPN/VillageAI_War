"""
MAPPO training env: all allied bots act each tick, all opponent bots, then one sim tick.

Observation stacks K local bot vectors (``game.max_bots_for_role_change``) plus global map/village tail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from village_ai_war.agents.village_obs_builder import VillageObsBuilder
from village_ai_war.env.game_env import GameEnv
from village_ai_war.models.mappo_layout import mappo_obs_dim
from village_ai_war.play.mappo_obs import (
    build_mappo_global_state,
    build_mappo_locals_matrix,
    pack_mappo_observation_vector,
)
from village_ai_war.state.game_state import GameState


def _as_plain_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class MAPPOBotEnv(gym.Env):
    """MAPPO with K local slots + global state for centralized critic (all allies move per tick)."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        team: int = 0,
        *,
        vec_env_index: int = 0,
    ) -> None:
        super().__init__()
        self._flat_cfg = _as_plain_config(config)
        n = int(self._flat_cfg["map"]["size"])
        self.inner = GameEnv(self._flat_cfg, mode="bot", team=team, render_mode=None)
        self.team = int(team)
        self.opponent_team = 1 - self.team

        self._local_dim = int(self.inner.observation_space.shape[0])
        self._n_bot_slots = int(self._flat_cfg["game"]["max_bots_for_role_change"])
        self._obs_dim = mappo_obs_dim(n, self._n_bot_slots)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [GameEnv.BOT_ACTIONS] * self._n_bot_slots
        )

        self._vec_env_index = int(vec_env_index)

        self.village_obs_builder = VillageObsBuilder(n)

    def _global_state(self, state: GameState) -> dict[str, np.ndarray]:
        """Canonical team-0 map POV plus both village vectors (for critic and logging)."""
        return build_mappo_global_state(state, self.village_obs_builder)

    def _locals_matrix(self, state: GameState) -> np.ndarray:
        """K x local_dim; alive bots sorted by bot_id fill low rows, then zeros."""
        return build_mappo_locals_matrix(
            state,
            self.inner,
            mappo_team=self.team,
            n_bot_slots=self._n_bot_slots,
        )

    def _pack_slots(self, locals_k: np.ndarray, gs: dict[str, np.ndarray]) -> np.ndarray:
        return pack_mappo_observation_vector(locals_k, gs)

    def _get_global_from_state(self) -> dict[str, np.ndarray]:
        st = self.inner.game_state
        assert st is not None
        return self._global_state(st)

    def _step_opponent_bots(self) -> None:
        st = self.inner.game_state
        assert st is not None
        village = st.villages[self.opponent_team]
        for bot in village.bots:
            if not bot.is_alive:
                continue
            act_int = int(self.inner.action_space.sample())
            self.inner.queue_bot_action(self.opponent_team, bot.bot_id, act_int)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _, info = self.inner.reset(seed=seed, options=options)
        gs = self._get_global_from_state()
        st = self.inner.game_state
        assert st is not None
        info = dict(info)
        info["global_state"] = gs
        mat = self._locals_matrix(st)
        packed = self._pack_slots(mat, gs)
        return packed, info

    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        st0 = self.inner.game_state
        assert st0 is not None
        village = st0.villages[self.team]
        alive = sorted((b for b in village.bots if b.is_alive), key=lambda b: int(b.bot_id))

        acts = np.asarray(action, dtype=np.int64).reshape(-1)
        if acts.shape[0] != self._n_bot_slots:
            raise ValueError(
                f"Expected {self._n_bot_slots} actions, got shape {acts.shape}"
            )

        if not alive:
            gs = self._get_global_from_state()
            z = np.zeros((self._n_bot_slots, self._local_dim), dtype=np.float32)
            info = {**self.inner._info_dict(), "global_state": gs}
            return self._pack_slots(z, gs), 0.0, True, False, info

        self.inner.snapshot_bot_positions_for_tick()
        self.inner.begin_mappo_tick()

        controlled: list[tuple[int, int]] = []
        for i, bot in enumerate(alive[: self._n_bot_slots]):
            a = int(acts[i])
            self.inner.queue_bot_action(self.team, bot.bot_id, a)
            controlled.append((bot.bot_id, a))
        if controlled:
            self.inner._controlled_bot_id = controlled[0][0]

        self._step_opponent_bots()

        learner_bot_actions = {bid: ac for bid, ac in controlled}
        _, reward, terminated, truncated, info = self.inner._simulation_tick(
            manager_action=None,
            learner_bot_action=None,
            learner_bot_actions=learner_bot_actions,
        )

        gs = self._get_global_from_state()
        info = dict(info)
        info["global_state"] = gs

        st1 = self.inner.game_state
        assert st1 is not None
        mat = self._locals_matrix(st1)
        if not any(b.is_alive for b in st1.villages[self.team].bots):
            terminated = True

        packed = self._pack_slots(mat, gs)
        return packed, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()
