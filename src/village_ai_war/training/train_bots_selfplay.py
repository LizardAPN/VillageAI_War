"""Stage 1: train a single role-conditioned bot policy via self-play."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from village_ai_war.models.role_conditioned_policy import RoleConditionedPolicy
from village_ai_war.training.pool_manager import PoolManager
from village_ai_war.training.self_play_env import SelfPlayBotEnv


def _flat_cfg(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


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

    tb_log: str | None = None
    if importlib.util.find_spec("tensorboard") is not None:
        tb_log = str(Path(tcfg["log_dir"]) / "bots")

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

    for iteration in range(iterations):
        logger.info("Bot self-play iteration {} / {}", iteration + 1, iterations)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"bot_iter{iteration}",
        )
        cbs = [checkpoint_callback, *callbacks_extra]
        model.learn(
            total_timesteps=steps_per_iter,
            callback=cbs,
            reset_num_timesteps=(iteration == 0),
            tb_log_name="bot_selfplay",
        )
        stem = pool_dir / f"bot_iter{iteration}"
        model.save(str(stem))
        pool_manager.add(Path(str(stem) + ".zip"))

    final_stem = checkpoint_dir / "bot_final"
    model.save(str(final_stem))
    vec_env.close()
    logger.info("Saved bot policy to {}.zip", final_stem)
