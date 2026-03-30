"""Stage 2: train village manager with MaskablePPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from village_ai_war.env.game_env import GameEnv


class _MaskableMonitor(Monitor):
    """``Monitor`` that forwards ``action_masks`` for MaskablePPO."""

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def run_village_training(cfg: Any, team: int = 0) -> None:
    """Train MaskablePPO on high-level village MDP (bots use heuristics)."""
    flat = _cfg_to_dict(cfg)
    tcfg = flat["training"]
    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    ckpt_dir = Path(tcfg["checkpoint_dir"]) / "village"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def make_one(_rank: int) -> Any:
        def _init() -> _MaskableMonitor:
            env = GameEnv(dict(flat), mode="village", team=team, render_mode=None)
            return _MaskableMonitor(env)

        return _init

    if n_envs > 1:
        venv = SubprocVecEnv([make_one(i) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_one(0)])
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        tensorboard_log=str(Path(tcfg["log_dir"]) / "tb_village"),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(total // 10, 1000),
        save_path=str(ckpt_dir),
        name_prefix="maskable_ppo_village",
    )
    model.learn(total_timesteps=total, callback=ckpt_cb)
    model.save(str(ckpt_dir / "village_final"))
    venv.close()
    logger.info("Saved village policy to {}", ckpt_dir / "village_final")
