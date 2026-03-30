"""Gym wrappers for self-play training (bots and village manager)."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from village_ai_war.agents.bot_obs_builder import BotObsBuilder
from village_ai_war.env.game_env import GameEnv
from village_ai_war.rewards.bot_reward import BotRewardCalculator
from village_ai_war.state import GlobalRewardMode


def _maskable_village_obs_matches_env(policy: MaskablePPO, env_obs_space: gym.Space) -> bool:
    """True if ``policy`` was trained on the same observation space as ``env_obs_space``."""
    try:
        return policy.observation_space == env_obs_space
    except Exception:  # noqa: BLE001
        return False


def _obs_map_tensor_shape(obs_space: gym.Space) -> tuple[int, ...] | None:
    """Shape of the ``map`` tensor in a village Dict obs space, or ``None``."""
    if isinstance(obs_space, spaces.Dict) and "map" in obs_space.spaces:
        m = obs_space.spaces["map"]
        if isinstance(m, spaces.Box):
            return tuple(int(x) for x in m.shape)
    return None


# One WARNING per (policy map shape, env map shape): many envs/zips share the same mismatch.
_village_obs_mismatch_warned_shapes: set[
    tuple[tuple[int, ...] | None, tuple[int, ...] | None]
] = set()


def _warn_village_obs_mismatch_once(
    path: Path,
    *,
    role: str,
    policy_space: gym.Space,
    env_space: gym.Space,
) -> None:
    pshape = _obs_map_tensor_shape(policy_space)
    eshape = _obs_map_tensor_shape(env_space)
    sig = (pshape, eshape)
    if sig in _village_obs_mismatch_warned_shapes:
        return
    _village_obs_mismatch_warned_shapes.add(sig)
    logger.warning(
        "Village checkpoints do not match env obs (first hit: {} side, example {}): "
        "policy {} != env {}; further incompatible zips are skipped without extra logs. "
        "Using random village manager actions until compatible checkpoints exist.",
        role,
        path,
        policy_space,
        env_space,
    )


def _as_plain_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class SelfPlayBotEnv(gym.Env):
    """Bot-mode env: team 0 learns; team 1 actions come from a checkpoint pool."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        opponent_pool_dir: str = "checkpoints/pool/bots",
        opponent_sampling: str = "uniform",
    ) -> None:
        super().__init__()
        self._flat_cfg = _as_plain_config(config)
        self.inner = GameEnv(self._flat_cfg, mode="bot", team=0, render_mode=None)
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.opponent_pool_dir = Path(opponent_pool_dir)
        self.opponent_sampling = opponent_sampling
        self._opponent_policy: PPO | None = None
        self._load_opponent()

    def _load_opponent(self) -> None:
        self.opponent_pool_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(self.opponent_pool_dir.glob("*.zip"))
        if not checkpoints:
            self._opponent_policy = None
            return
        if self.opponent_sampling == "latest":
            ckpt = checkpoints[-1]
        elif self.opponent_sampling == "random":
            ckpt = checkpoints[int(np.random.randint(len(checkpoints)))]
        else:
            ckpt = checkpoints[int(np.random.randint(len(checkpoints)))]
        try:
            self._opponent_policy = PPO.load(str(ckpt))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load opponent bot policy from {}: {}", ckpt, e)
            self._opponent_policy = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.np_random.random() < 0.1:
            self._load_opponent()
        obs, info = self.inner.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        blue_obs = self.inner._get_bot_obs(1)
        if self._opponent_policy is not None and blue_obs is not None:
            blue_act, _ = self._opponent_policy.predict(blue_obs, deterministic=False)
            blue_action = int(np.asarray(blue_act).reshape(-1)[0])
        else:
            blue_action = int(self.inner.action_space.sample())
        return self.inner.step_with_opponent(int(action), blue_action)

    def render(self) -> Any:
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()


class SelfPlayVillageEnv(gym.Env):
    """Village-mode env: RL bots frozen checkpoint + self-play opponent manager."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        bot_checkpoint_dir: str = "checkpoints/bots",
        opponent_pool_dir: str = "checkpoints/pool/village",
        opponent_sampling: str = "uniform",
    ) -> None:
        super().__init__()
        self._flat_cfg = _as_plain_config(config)
        self.inner = GameEnv(self._flat_cfg, mode="village", team=0, render_mode=None)
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.bot_checkpoint_dir = Path(bot_checkpoint_dir)
        self.opponent_pool_dir = Path(opponent_pool_dir)
        self.opponent_sampling = opponent_sampling
        self._bot_policy: PPO | None = None
        self._opponent_policy: MaskablePPO | None = None
        self._load_bot_policy()
        self._load_opponent()

    def _load_bot_policy(self) -> None:
        for name in ("bot_final.zip", "bot_final"):
            ckpt = self.bot_checkpoint_dir / name
            if ckpt.is_file():
                try:
                    self._bot_policy = PPO.load(str(ckpt))
                    return
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load bot policy from {}: {}", ckpt, e)
        self._bot_policy = None

    def _load_opponent(self) -> None:
        self.opponent_pool_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(self.opponent_pool_dir.glob("*.zip"))
        if not checkpoints:
            self._opponent_policy = None
            return
        if self.opponent_sampling == "latest":
            ckpt = checkpoints[-1]
        else:
            ckpt = checkpoints[int(np.random.randint(len(checkpoints)))]
        try:
            self._opponent_policy = MaskablePPO.load(str(ckpt))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load opponent village policy from {}: {}", ckpt, e)
            self._opponent_policy = None

    def action_masks(self) -> np.ndarray:
        return self.inner.action_masks(team=0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if self.np_random.random() < 0.1:
            self._load_opponent()
        obs, info = self.inner.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        melee_intents: list[tuple[int, int, tuple[int, int]]] = []
        self.inner._step_all_bots_with_policy(self._bot_policy, melee_intents, exclude=None)
        blue_obs = self.inner._get_village_obs(1)
        if self._opponent_policy is not None:
            blue_masks = self.inner.action_masks(team=1)
            blue_act, _ = self._opponent_policy.predict(
                blue_obs, action_masks=blue_masks, deterministic=False
            )
            blue_action = int(np.asarray(blue_act).reshape(-1)[0])
        else:
            blue_action = int(self.inner.action_space.sample())
        return self.inner.step_village_only(int(action), blue_action, melee_intents)

    def render(self) -> Any:
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()


class UnifiedBotSelfPlayEnv(gym.Env):
    """Full-game env for unified training: one bot on team 0 learns, everything else frozen.

    Each tick mirrors ``SelfPlayVillageEnv`` (bots move, then both village managers act),
    but the learner controls a single bot on team 0 and receives bot-level reward.

    Frozen policies loaded from paths/pools:
    * Other team-0 bots: ``bot_policy_holder["model"]`` (mutable dict shared with trainer).
    * Team-1 bots: opponent pool ``checkpoints/pool/bots``.
    * Red village manager: checkpoint at ``village_checkpoint_path`` (or random if missing).
    * Blue village manager: opponent pool ``checkpoints/pool/village`` (or random).
    """

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        bot_policy_holder: MutableMapping[str, Any] | None = None,
        village_checkpoint_path: str = "checkpoints/unified/village_latest",
        opponent_bot_pool_dir: str = "checkpoints/pool/bots",
        opponent_village_pool_dir: str = "checkpoints/pool/village",
        opponent_sampling: str = "uniform",
    ) -> None:
        super().__init__()
        self._flat_cfg = _as_plain_config(config)
        self.inner = GameEnv(self._flat_cfg, mode="village", team=0, render_mode=None)

        self.action_space = spaces.Discrete(GameEnv.BOT_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(BotObsBuilder.OBS_DIM,), dtype=np.float32,
        )

        self._bot_policy_holder = bot_policy_holder
        self._village_ckpt_path = Path(village_checkpoint_path)
        self._opponent_bot_pool_dir = Path(opponent_bot_pool_dir)
        self._opponent_village_pool_dir = Path(opponent_village_pool_dir)
        self._opponent_sampling = opponent_sampling

        self._opponent_bot_policy: PPO | None = None
        self._red_village_policy: MaskablePPO | None = None
        self._blue_village_policy: MaskablePPO | None = None
        self._controlled_bot_id: int = 0

        self._load_opponent_bot()
        self._load_red_village()
        self._load_blue_village()

    def _load_opponent_bot(self) -> None:
        self._opponent_bot_pool_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(self._opponent_bot_pool_dir.glob("*.zip"))
        if not checkpoints:
            self._opponent_bot_policy = None
            return
        ckpt = checkpoints[-1] if self._opponent_sampling == "latest" else checkpoints[
            int(np.random.randint(len(checkpoints)))
        ]
        try:
            self._opponent_bot_policy = PPO.load(str(ckpt))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load opponent bot policy from {}: {}", ckpt, e)
            self._opponent_bot_policy = None

    def _load_red_village(self) -> None:
        env_space = self.inner.observation_space
        for suffix in ("", ".zip"):
            p = self._village_ckpt_path.parent / (self._village_ckpt_path.name + suffix)
            if p.is_file():
                try:
                    loaded = MaskablePPO.load(str(p))
                    if not _maskable_village_obs_matches_env(loaded, env_space):
                        _warn_village_obs_mismatch_once(
                            p,
                            role="red",
                            policy_space=loaded.observation_space,
                            env_space=env_space,
                        )
                        continue
                    self._red_village_policy = loaded
                    return
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load red village policy from {}: {}", p, e)
        self._red_village_policy = None

    def _load_blue_village(self) -> None:
        self._opponent_village_pool_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(self._opponent_village_pool_dir.glob("*.zip"))
        if not checkpoints:
            self._blue_village_policy = None
            return
        ckpt = checkpoints[-1] if self._opponent_sampling == "latest" else checkpoints[
            int(np.random.randint(len(checkpoints)))
        ]
        env_space = self.inner.observation_space
        try:
            loaded = MaskablePPO.load(str(ckpt))
            if not _maskable_village_obs_matches_env(loaded, env_space):
                _warn_village_obs_mismatch_once(
                    ckpt,
                    role="blue",
                    policy_space=loaded.observation_space,
                    env_space=env_space,
                )
                self._blue_village_policy = None
                return
            self._blue_village_policy = loaded
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load blue village policy from {}: {}", ckpt, e)
            self._blue_village_policy = None

    def _pick_controlled_bot(self) -> None:
        assert self.inner._state is not None
        alive = [b for b in self.inner._state.villages[0].bots if b.is_alive]
        self._controlled_bot_id = alive[0].bot_id if alive else 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.np_random.random() < 0.1:
            self._load_opponent_bot()
            self._load_blue_village()
        obs, info = self.inner.reset(seed=seed, options=options)
        self._pick_controlled_bot()
        bot_obs = self.inner._get_single_bot_obs(self._controlled_bot_id)
        return bot_obs, info

    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.inner._state is not None and self.inner._rng is not None
        melee_intents: list[tuple[int, int, tuple[int, int]]] = []

        # --- bot phase ---
        learner_action = int(action)
        self.inner._apply_bot_action(0, self._controlled_bot_id, learner_action, melee_intents)

        friendly_policy = (
            self._bot_policy_holder.get("model") if self._bot_policy_holder else None
        )
        for bot in self.inner._state.villages[0].bots:
            if not bot.is_alive or bot.bot_id == self._controlled_bot_id:
                continue
            obs_b = self.inner._get_single_bot_obs(bot.bot_id)
            if friendly_policy is not None:
                a, _ = friendly_policy.predict(obs_b, deterministic=False)
                act_int = int(np.asarray(a).reshape(-1)[0])
            else:
                act_int = int(self.inner._rng.integers(0, GameEnv.BOT_ACTIONS))
            self.inner._apply_bot_action(0, bot.bot_id, act_int, melee_intents)

        for bot in self.inner._state.villages[1].bots:
            if not bot.is_alive:
                continue
            obs_b = self.inner._get_single_bot_obs(bot.bot_id)
            if self._opponent_bot_policy is not None:
                a, _ = self._opponent_bot_policy.predict(obs_b, deterministic=False)
                act_int = int(np.asarray(a).reshape(-1)[0])
            else:
                act_int = int(self.inner._rng.integers(0, GameEnv.BOT_ACTIONS))
            self.inner._apply_bot_action(1, bot.bot_id, act_int, melee_intents)

        # --- village manager phase ---
        red_masks = self.inner.action_masks(team=0)
        if self._red_village_policy is not None:
            red_obs = self.inner._get_village_obs(0)
            red_act, _ = self._red_village_policy.predict(
                red_obs, action_masks=red_masks, deterministic=False,
            )
            red_village_action = int(np.asarray(red_act).reshape(-1)[0])
        else:
            valid = np.flatnonzero(red_masks)
            red_village_action = int(self.inner._rng.choice(valid)) if len(valid) else 0

        blue_masks = self.inner.action_masks(team=1)
        if self._blue_village_policy is not None:
            blue_obs = self.inner._get_village_obs(1)
            blue_act, _ = self._blue_village_policy.predict(
                blue_obs, action_masks=blue_masks, deterministic=False,
            )
            blue_village_action = int(np.asarray(blue_act).reshape(-1)[0])
        else:
            valid = np.flatnonzero(blue_masks)
            blue_village_action = int(self.inner._rng.choice(valid)) if len(valid) else 0

        _, _, terminated, truncated, info = self.inner.step_village_only(
            red_village_action, blue_village_action, melee_intents,
        )

        # --- bot-level obs & reward ---
        bot_obs = self.inner._get_single_bot_obs(self._controlled_bot_id)
        bot_state = next(
            (b for b in self.inner._state.villages[0].bots if b.bot_id == self._controlled_bot_id),
            None,
        )
        if bot_state is not None:
            mode = self.inner._state.villages[0].global_reward_mode
            bev = GameEnv._bot_events_for(bot_state, self.inner._last_tick_merged, learner_action)
            reward = float(BotRewardCalculator.compute(bev, bot_state, mode, self.inner.config))
        else:
            reward = 0.0

        return bot_obs, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()
