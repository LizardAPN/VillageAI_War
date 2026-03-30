"""Gym wrappers for self-play training (bots and village manager)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from village_ai_war.env.game_env import GameEnv


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
