"""Stage 2: train village manager (MaskablePPO) with frozen RL bots and self-play."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from village_ai_war.training.pool_manager import PoolManager
from village_ai_war.training.self_play_env import SelfPlayVillageEnv


class _MaskableMonitor(Monitor):
    """Monitor that exposes ``action_masks`` for MaskablePPO."""

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def _flat_cfg(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _tensorboard_log_dir(flat: dict[str, Any], tcfg: dict[str, Any], subdir: str) -> str | None:
    if not bool(flat.get("logging", {}).get("use_tensorboard", True)):
        return None
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning(
            "logging.use_tensorboard is true but tensorboard is not installed; "
            "training without tensorboard_log"
        )
        return None
    return str(Path(tcfg["log_dir"]) / subdir)


def run_village_selfplay_training(cfg: Any) -> None:
    """MaskablePPO self-play for the village manager."""
    flat = _flat_cfg(cfg)
    tcfg = flat["training"]
    pool_dir = Path(tcfg["pool_dir"]) / "village"
    pool_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(tcfg["checkpoint_dir"]) / "village"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bot_ckpt_dir = str(Path(tcfg["checkpoint_dir"]) / "bots")
    pool_manager = PoolManager(pool_dir, max_size=int(tcfg.get("pool_max_size", 10)))

    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    iterations = int(tcfg.get("selfplay_iterations", 1))
    steps_per_iter = max(total // iterations, n_envs)

    def make_env(_rank: int) -> Any:
        def _init() -> _MaskableMonitor:
            env = SelfPlayVillageEnv(
                flat,
                bot_checkpoint_dir=bot_ckpt_dir,
                opponent_pool_dir=str(pool_dir),
                opponent_sampling="uniform",
            )
            return _MaskableMonitor(env)

        return _init

    use_subproc = n_envs > 1
    vec_env: DummyVecEnv | SubprocVecEnv = (
        SubprocVecEnv([make_env(i) for i in range(n_envs)])
        if use_subproc
        else DummyVecEnv([make_env(0)])
    )

    tb_log = _tensorboard_log_dir(flat, tcfg, "village")

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        n_steps=int(tcfg.get("n_steps", 2048)),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        tensorboard_log=tb_log,
    )

    save_freq = max(int(tcfg.get("checkpoint_interval", 25_000)) // n_envs, 1)

    user_eval_freq = int(tcfg.get("eval_freq", 10_000))
    eval_cb_freq = max(user_eval_freq // max(n_envs, 1), 1) if user_eval_freq > 0 else 0
    eval_sampling = str(tcfg.get("eval_opponent_sampling", "latest"))

    eval_callback: MaskableEvalCallback | None = None
    eval_venv: DummyVecEnv | None = None
    if eval_cb_freq > 0:

        def make_eval_env() -> _MaskableMonitor:
            env = SelfPlayVillageEnv(
                flat,
                bot_checkpoint_dir=bot_ckpt_dir,
                opponent_pool_dir=str(pool_dir),
                opponent_sampling=eval_sampling,
            )
            return _MaskableMonitor(env)

        eval_venv = DummyVecEnv([make_eval_env])
        eval_log = Path(tcfg["log_dir"]) / "village_eval"
        best_path = checkpoint_dir / "best_village"
        eval_callback = MaskableEvalCallback(
            eval_venv,
            best_model_save_path=str(best_path),
            log_path=str(eval_log),
            eval_freq=eval_cb_freq,
            n_eval_episodes=int(tcfg.get("n_eval_episodes", 5)),
            deterministic=True,
            verbose=1,
        )

    for iteration in range(iterations):
        logger.info("Village self-play iteration {} / {}", iteration + 1, iterations)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"village_iter{iteration}",
        )
        cbs: list[Any] = [checkpoint_callback]
        if eval_callback is not None:
            cbs.append(eval_callback)
        model.learn(
            total_timesteps=steps_per_iter,
            callback=cbs,
            reset_num_timesteps=(iteration == 0),
            tb_log_name="village_selfplay",
        )
        stem = pool_dir / f"village_iter{iteration}"
        model.save(str(stem))
        pool_manager.add(Path(str(stem) + ".zip"))

    model.save(str(checkpoint_dir / "village_last"))
    best_zip = checkpoint_dir / "best_village" / "best_model.zip"
    if user_eval_freq > 0 and best_zip.is_file():
        shutil.copy2(best_zip, checkpoint_dir / "village_best.zip")
        shutil.copy2(best_zip, checkpoint_dir / "village_final.zip")
        logger.info(
            "Exported best eval policy to {} and {} (last weights: {}.zip)",
            checkpoint_dir / "village_final.zip",
            checkpoint_dir / "village_best.zip",
            checkpoint_dir / "village_last",
        )
    else:
        model.save(str(checkpoint_dir / "village_final"))
        logger.info(
            "Exported last-iteration policy to {}.zip (eval disabled or no best checkpoint yet)",
            checkpoint_dir / "village_final",
        )

    if eval_venv is not None:
        eval_venv.close()
    vec_env.close()
