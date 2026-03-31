"""MAPPO bot training: decentralized actor + centralized critic, bot self-play pool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from village_ai_war.models.mappo_policy import MAPPOPolicy
from village_ai_war.training.mappo_env import MAPPOBotEnv
from village_ai_war.training.mappo_episode_metrics_callback import MAPPOEpisodeMetricsCallback
from village_ai_war.training.pool_manager import PoolManager


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


def run_mappo_bots_training(
    cfg: Any, *, return_metrics: bool = False
) -> dict[str, Any] | None:
    """MAPPO (PPO + centralized critic) with a rolling pool of past MAPPO snapshots.

    If ``return_metrics`` is True, returns ``win_frac`` and ``outcome_fractions`` from the
    episode metrics callback; otherwise returns ``None``.
    """
    flat = _flat_cfg(cfg)
    tcfg = flat["training"]
    pool_root = Path(tcfg["pool_dir"])
    mappo_pool_dir = pool_root / str(tcfg.get("mappo_pool_subdir", "bots_mappo"))
    mappo_pool_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(tcfg["checkpoint_dir"]) / "bots_mappo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pool_manager = PoolManager(mappo_pool_dir, max_size=int(tcfg.get("pool_max_size", 15)))

    n = int(flat["map"]["size"])
    n_bot_slots = int(flat["game"]["max_bots_for_role_change"])

    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    iterations = int(tcfg.get("selfplay_iterations", 1))
    steps_per_iter = max(total // max(iterations, 1), n_envs)

    def make_env(_rank: int) -> Any:
        def _init() -> MAPPOBotEnv:
            return MAPPOBotEnv(
                flat,
                team=0,
                vec_env_index=_rank,
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
    ent_coef = float(tcfg.get("ent_coef", 0.01))
    vf_coef = float(tcfg.get("vf_coef", 0.5))
    critic_h = int(tcfg.get("critic_hidden_dim", 256))

    tb_log = _tensorboard_log_dir(flat, tcfg, "mappo_bots")

    model = PPO(
        MAPPOPolicy,
        vec_env,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        n_steps=int(tcfg.get("n_steps", 512)),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        tensorboard_log=tb_log,
        policy_kwargs={
            "map_size": n,
            "critic_hidden_dim": critic_h,
            "n_bot_slots": n_bot_slots,
        },
    )

    save_freq = max(int(tcfg.get("checkpoint_interval", 100_000)) // max(n_envs, 1), 1)
    metrics_window = int(tcfg.get("mappo_metrics_window", 512))
    metrics_cb = MAPPOEpisodeMetricsCallback(window=metrics_window)

    for iteration in range(iterations):
        logger.info("MAPPO self-play iteration {} / {}", iteration + 1, iterations)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"mappo_bot_iter{iteration}",
        )
        learn_cb = CallbackList([checkpoint_callback, metrics_cb])
        model.learn(
            total_timesteps=steps_per_iter,
            callback=learn_cb,
            reset_num_timesteps=(iteration == 0),
            tb_log_name="mappo_bot_selfplay",
        )
        stem = mappo_pool_dir / f"mappo_bot_iter{iteration}"
        model.save(str(stem))
        pool_manager.add(Path(str(stem) + ".zip"))

    model.save(str(checkpoint_dir / "mappo_bot_final"))
    logger.info("Saved MAPPO policy to {}.zip", checkpoint_dir / "mappo_bot_final")

    vec_env.close()

    if return_metrics:
        fracs = metrics_cb.outcome_fractions()
        return {
            "win_frac": float(fracs.get("win", 0.0)),
            "outcome_fractions": fracs,
        }
    return None
