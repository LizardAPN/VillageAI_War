"""Stage 1: train a single role-conditioned bot policy via self-play."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from village_ai_war.models.role_conditioned_policy import RoleConditionedPolicy
from village_ai_war.training.pool_manager import PoolManager
from village_ai_war.training.self_play_env import SelfPlayBotEnv


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


def run_bots_selfplay_training(cfg: Any) -> None:
    """PPO self-play for bots with a growing opponent pool."""
    flat = _flat_cfg(cfg)
    tcfg = flat["training"]
    pool_dir = Path(tcfg["pool_dir"]) / "bots"
    pool_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(tcfg["checkpoint_dir"]) / "bots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pool_manager = PoolManager(pool_dir, max_size=int(tcfg.get("pool_max_size", 10)))

    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    iterations = int(tcfg.get("selfplay_iterations", 1))
    steps_per_iter = max(total // iterations, n_envs)

    def make_env(_rank: int) -> Any:
        def _init() -> SelfPlayBotEnv:
            return SelfPlayBotEnv(
                flat,
                opponent_pool_dir=str(pool_dir),
                opponent_sampling="uniform",
            )

        return _init

    use_subproc = n_envs > 1
    vec_env: DummyVecEnv | SubprocVecEnv = (
        SubprocVecEnv([make_env(i) for i in range(n_envs)])
        if use_subproc
        else DummyVecEnv([make_env(0)])
    )
    vec_env = VecMonitor(vec_env)

    gae_lambda = float(tcfg.get("gae_lambda", 0.95))
    clip_range = float(tcfg.get("clip_range", 0.2))
    ent_coef = float(tcfg.get("ent_coef", 0.0))

    tb_log = _tensorboard_log_dir(flat, tcfg, "bots")

    model = PPO(
        RoleConditionedPolicy,
        vec_env,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        n_steps=int(tcfg.get("n_steps", 2048)),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        tensorboard_log=tb_log,
    )

    callbacks_extra: list[Any] = []
    if bool(flat.get("logging", {}).get("use_wandb", False)):
        try:
            from wandb.integration.sb3 import WandbCallback

            callbacks_extra.append(WandbCallback(verbose=0))
        except Exception as e:  # noqa: BLE001
            logger.warning("WandbCallback unavailable: {}", e)

    save_freq = max(int(tcfg.get("checkpoint_interval", 50_000)) // n_envs, 1)

    user_eval_freq = int(tcfg.get("eval_freq", 10_000))
    eval_cb_freq = max(user_eval_freq // max(n_envs, 1), 1) if user_eval_freq > 0 else 0
    eval_sampling = str(tcfg.get("eval_opponent_sampling", "latest"))

    eval_callback: EvalCallback | None = None
    eval_venv: DummyVecEnv | None = None
    if eval_cb_freq > 0:

        def make_eval_env() -> SelfPlayBotEnv:
            return SelfPlayBotEnv(
                flat,
                opponent_pool_dir=str(pool_dir),
                opponent_sampling=eval_sampling,
            )

        eval_venv = DummyVecEnv([make_eval_env])
        eval_venv = VecMonitor(eval_venv)
        eval_log = Path(tcfg["log_dir"]) / "bots_eval"
        best_path = checkpoint_dir / "best_bot"
        eval_callback = EvalCallback(
            eval_venv,
            best_model_save_path=str(best_path),
            log_path=str(eval_log),
            eval_freq=eval_cb_freq,
            n_eval_episodes=int(tcfg.get("n_eval_episodes", 5)),
            deterministic=True,
            verbose=1,
        )

    for iteration in range(iterations):
        logger.info("Bot self-play iteration {} / {}", iteration + 1, iterations)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"bot_iter{iteration}",
        )
        cbs: list[Any] = [checkpoint_callback]
        if eval_callback is not None:
            cbs.append(eval_callback)
        cbs.extend(callbacks_extra)
        model.learn(
            total_timesteps=steps_per_iter,
            callback=cbs,
            reset_num_timesteps=(iteration == 0),
            tb_log_name="bot_selfplay",
        )
        stem = pool_dir / f"bot_iter{iteration}"
        model.save(str(stem))
        pool_manager.add(Path(str(stem) + ".zip"))

    model.save(str(checkpoint_dir / "bot_last"))
    best_zip = checkpoint_dir / "best_bot" / "best_model.zip"
    if user_eval_freq > 0 and best_zip.is_file():
        shutil.copy2(best_zip, checkpoint_dir / "bot_best.zip")
        shutil.copy2(best_zip, checkpoint_dir / "bot_final.zip")
        logger.info(
            "Exported best eval policy to {} and {} (last weights: {}.zip)",
            checkpoint_dir / "bot_final.zip",
            checkpoint_dir / "bot_best.zip",
            checkpoint_dir / "bot_last",
        )
    else:
        model.save(str(checkpoint_dir / "bot_final"))
        logger.info(
            "Exported last-iteration policy to {}.zip (eval disabled or no best checkpoint yet)",
            checkpoint_dir / "bot_final",
        )

    if eval_venv is not None:
        eval_venv.close()
    vec_env.close()
