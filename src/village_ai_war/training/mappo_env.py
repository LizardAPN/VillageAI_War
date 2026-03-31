"""
MAPPO training env: all allied bots act each tick, all opponent bots, then one sim tick.

Observation stacks K local bot vectors (``game.max_bots_for_role_change``) plus global map/village tail.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from stable_baselines3 import PPO

from village_ai_war.agents.village_obs_builder import VillageObsBuilder
from village_ai_war.env.game_env import GameEnv
from village_ai_war.models.mappo_layout import mappo_obs_dim, pack_mappo_obs_slots
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
        opponent_pool_dir: str = "checkpoints/pool/bots",
        opponent_sampling: str = "uniform",
        *,
        vec_env_index: int = 0,
    ) -> None:
        super().__init__()
        self._flat_cfg = _as_plain_config(config)
        n = int(self._flat_cfg["map"]["size"])
        self.inner = GameEnv(self._flat_cfg, mode="bot", team=team, render_mode=None)
        self.team = int(team)
        self.opponent_team = 1 - self.team

        self._local_dim = int(self.inner.observation_space.shape[0])  # BotObsBuilder.OBS_DIM
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

        self.opponent_pool_dir = Path(opponent_pool_dir)
        self.opponent_sampling = opponent_sampling
        self._vec_env_index = int(vec_env_index)
        self._opponent_policy: PPO | None = None
        self._skip_opponent_logged: set[str] = set()
        self._warned_no_compatible_opponent: bool = False

        self.village_obs_builder = VillageObsBuilder(n)
        self._load_opponent()

    def _opponent_obs_compatible(self, model: PPO) -> bool:
        """Opponent policy must match ``inner`` bot Box (181-dim), not MAPPO extended obs."""
        pe = model.observation_space
        ee = self.inner.observation_space
        if not isinstance(pe, spaces.Box) or not isinstance(ee, spaces.Box):
            return False
        return tuple(int(x) for x in pe.shape) == tuple(int(x) for x in ee.shape)

    def _load_opponent(self) -> None:
        self.opponent_pool_dir.mkdir(parents=True, exist_ok=True)
        all_zips = sorted(self.opponent_pool_dir.glob("*.zip"))
        # MAPPO zips live in pool/bots_mappo/; skip mappo_bot* here to avoid loading them.
        checkpoints = [p for p in all_zips if not p.name.startswith("mappo_bot")]
        if not all_zips:
            self._opponent_policy = None
            return
        if not checkpoints:
            self._opponent_policy = None
            self._emit_no_opponent_message(
                "opponent pool has only mappo_bot_*.zip (skipped); add 181-dim zips or use "
                "pool/bots_mappo/ for MAPPO — random opponent moves",
            )
            return

        rng = np.random.default_rng()
        if self.opponent_sampling == "latest":
            order = list(reversed(checkpoints))
        else:
            perm = rng.permutation(len(checkpoints))
            order = [checkpoints[int(i)] for i in perm]

        self._opponent_policy = None
        for ckpt in order:
            try:
                model = PPO.load(str(ckpt))
            except Exception as e:  # noqa: BLE001
                logger.debug("Skip opponent {}: {}", ckpt.name, e)
                continue
            if not self._opponent_obs_compatible(model):
                key = str(ckpt.resolve())
                if key not in self._skip_opponent_logged:
                    self._skip_opponent_logged.add(key)
                    logger.debug(
                        "Skip opponent {}: policy obs {} != inner bot obs {}",
                        ckpt.name,
                        getattr(model.observation_space, "shape", None),
                        self.inner.observation_space.shape,
                    )
                continue
            self._opponent_policy = model
            self._warned_no_compatible_opponent = False
            logger.debug("Loaded opponent: {}", ckpt.name)
            return

        self._emit_no_opponent_message(
            "no compatible opponent in {} (need Box shape {}); using random opponent moves",
            self.opponent_pool_dir,
            self.inner.observation_space.shape,
        )

    def _emit_no_opponent_message(self, msg: str, *args: Any) -> None:
        """SubprocVecEnv: each process has its own memory — log only from worker 0, once (no DEBUG spam)."""
        if self._vec_env_index != 0:
            return
        if self._warned_no_compatible_opponent:
            return
        self._warned_no_compatible_opponent = True
        logger.warning(msg, *args)

    def _global_state(self, state: GameState) -> dict[str, np.ndarray]:
        """Canonical team-0 map POV plus both village vectors (for critic and logging)."""
        mp = self.village_obs_builder.build_map(state, team=0)
        v0 = self.village_obs_builder.build_village_vec(state, team=0)
        v1 = self.village_obs_builder.build_village_vec(state, team=1)
        return {"map": mp, "village": np.concatenate([v0, v1], axis=0).astype(np.float32)}

    def _locals_matrix(self, state: GameState) -> np.ndarray:
        """K x local_dim; alive bots sorted by bot_id fill low rows, then zeros."""
        k = self._n_bot_slots
        d = self._local_dim
        out = np.zeros((k, d), dtype=np.float32)
        village = state.villages[self.team]
        alive = sorted((b for b in village.bots if b.is_alive), key=lambda b: int(b.bot_id))
        for i, bot in enumerate(alive[:k]):
            out[i] = self.inner._get_single_bot_obs(bot.bot_id, self.team)
        return out

    def _pack_slots(self, locals_k: np.ndarray, gs: dict[str, np.ndarray]) -> np.ndarray:
        return pack_mappo_obs_slots(
            locals_k, gs["map"], gs["village"][:20], gs["village"][20:]
        )

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
            obs_local = self.inner._get_single_bot_obs(bot.bot_id, self.opponent_team)
            if self._opponent_policy is not None:
                act, _ = self._opponent_policy.predict(obs_local, deterministic=False)
                act_int = int(np.asarray(act).reshape(-1)[0])
            else:
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
        if self.np_random.random() < 0.1:
            self._load_opponent()
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
