"""Unified training loop: alternating PPO (bots) and MaskablePPO (village) in full-game episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from village_ai_war.models.role_conditioned_policy import RoleConditionedPolicy
from village_ai_war.training.pool_manager import PoolManager
from village_ai_war.training.self_play_env import SelfPlayVillageEnv, UnifiedBotSelfPlayEnv


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


def run_unified_training(cfg: Any) -> None:
    """Alternating bot PPO / village MaskablePPO in full-game episodes."""
    flat = _flat_cfg(cfg)
    tcfg = flat["training"]
    ucfg = flat.get("unified", {})

    n_cycles = int(ucfg.get("n_cycles", 10))
    bot_steps = int(ucfg.get("bot_steps_per_turn", 20_000))
    village_steps = int(ucfg.get("village_steps_per_turn", 20_000))
    push_to_pool = bool(ucfg.get("push_to_pool", True))
    first_phase = str(ucfg.get("first_phase", "bot"))

    n_envs = int(tcfg["n_envs"])
    checkpoint_dir = Path(tcfg["checkpoint_dir"]) / "unified"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bot_ckpt_path = checkpoint_dir / "bot_latest"
    village_ckpt_path = checkpoint_dir / "village_latest"

    pool_dir = Path(tcfg.get("pool_dir", "checkpoints/pool"))
    bot_pool_dir = pool_dir / "bots"
    village_pool_dir = pool_dir / "village"

    bot_pool = PoolManager(bot_pool_dir, max_size=int(tcfg.get("pool_max_size", 10)))
    village_pool = PoolManager(village_pool_dir, max_size=int(tcfg.get("pool_max_size", 10)))

    tb_log_bot = _tensorboard_log_dir(flat, tcfg, "unified_bots")
    tb_log_vil = _tensorboard_log_dir(flat, tcfg, "unified_village")

    bot_policy_holder: dict[str, Any] = {"model": None}

    # --- bot vec-env (DummyVecEnv only — holder is in-process) ---
    def _make_bot_env() -> Any:
        def _init() -> UnifiedBotSelfPlayEnv:
            return UnifiedBotSelfPlayEnv(
                flat,
                bot_policy_holder=bot_policy_holder,
                village_checkpoint_path=str(village_ckpt_path),
                opponent_bot_pool_dir=str(bot_pool_dir),
                opponent_village_pool_dir=str(village_pool_dir),
            )
        return _init

    bot_venv = DummyVecEnv([_make_bot_env() for _ in range(n_envs)])
    bot_venv = VecMonitor(bot_venv)

    gae_lambda = float(tcfg.get("gae_lambda", 0.95))
    clip_range = float(tcfg.get("clip_range", 0.2))
    ent_coef = float(tcfg.get("ent_coef", 0.0))

    model_bot = PPO(
        RoleConditionedPolicy,
        bot_venv,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        n_steps=int(tcfg.get("n_steps", 2048)),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        tensorboard_log=tb_log_bot,
    )
    bot_policy_holder["model"] = model_bot

    # --- village vec-env (SubprocVecEnv OK — bot loaded from disk) ---
    def _make_vil_env(_rank: int) -> Any:
        def _init() -> _MaskableMonitor:
            env = SelfPlayVillageEnv(
                flat,
                bot_checkpoint_dir=str(checkpoint_dir),
                opponent_pool_dir=str(village_pool_dir),
                opponent_sampling="uniform",
            )
            return _MaskableMonitor(env)
        return _init

    use_subproc = n_envs > 1
    vil_venv: DummyVecEnv = (
        DummyVecEnv([_make_vil_env(i) for i in range(n_envs)])
        if not use_subproc
        else DummyVecEnv([_make_vil_env(i) for i in range(n_envs)])
    )

    model_vil = MaskablePPO(
        "MultiInputPolicy",
        vil_venv,
        verbose=1,
        learning_rate=float(tcfg["learning_rate"]),
        n_steps=int(tcfg.get("n_steps", 2048)),
        batch_size=int(tcfg["batch_size"]),
        n_epochs=int(tcfg["n_epochs"]),
        gamma=float(tcfg["gamma"]),
        tensorboard_log=tb_log_vil,
    )

    # --- WandB callback (shared) ---
    callbacks_extra: list[Any] = []
    if bool(flat.get("logging", {}).get("use_wandb", False)):
        try:
            from wandb.integration.sb3 import WandbCallback
            callbacks_extra.append(WandbCallback(verbose=0))
        except Exception as e:  # noqa: BLE001
            logger.warning("WandbCallback unavailable: {}", e)

    save_freq_bot = max(bot_steps // 4, 1)
    save_freq_vil = max(village_steps // 4, 1)

    phases = ["bot", "village"] if first_phase == "bot" else ["village", "bot"]

    for cycle in range(n_cycles):
        logger.info("Unified cycle {} / {}", cycle + 1, n_cycles)

        for phase in phases:
            if phase == "bot":
                logger.info("  Bot phase: {} steps", bot_steps)
                bot_policy_holder["model"] = model_bot
                cb = CheckpointCallback(
                    save_freq=save_freq_bot,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"bot_cycle{cycle}",
                )
                model_bot.learn(
                    total_timesteps=bot_steps,
                    callback=[cb, *callbacks_extra],
                    reset_num_timesteps=(cycle == 0 and phase == phases[0]),
                    tb_log_name="unified_bot",
                )
                model_bot.save(str(bot_ckpt_path))
                # SelfPlayVillageEnv expects bot_final.zip inside bot_checkpoint_dir
                model_bot.save(str(checkpoint_dir / "bot_final"))
                if push_to_pool:
                    stem = bot_pool_dir / f"unified_bot_c{cycle}"
                    model_bot.save(str(stem))
                    bot_pool.add(Path(str(stem) + ".zip"))

            else:
                logger.info("  Village phase: {} steps", village_steps)
                cb = CheckpointCallback(
                    save_freq=save_freq_vil,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"village_cycle{cycle}",
                )
                model_vil.learn(
                    total_timesteps=village_steps,
                    callback=[cb, *callbacks_extra],
                    reset_num_timesteps=(cycle == 0 and phase == phases[0]),
                    tb_log_name="unified_village",
                )
                model_vil.save(str(village_ckpt_path))
                model_vil.save(str(checkpoint_dir / "village_final"))
                if push_to_pool:
                    stem = village_pool_dir / f"unified_village_c{cycle}"
                    model_vil.save(str(stem))
                    village_pool.add(Path(str(stem) + ".zip"))

    model_bot.save(str(checkpoint_dir / "bot_final"))
    model_vil.save(str(checkpoint_dir / "village_final"))
    bot_venv.close()
    vil_venv.close()
    logger.info("Unified training complete — checkpoints in {}", checkpoint_dir)
