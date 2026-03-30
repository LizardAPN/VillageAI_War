"""Stage 1: train one PPO policy per bot role."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from village_ai_war.env.game_env import GameEnv
from village_ai_war.state import Role


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _make_env_fn(
    flat_cfg: Mapping[str, Any],
    team: int,
    role: Role,
) -> Any:
    def _init() -> GameEnv:
        return GameEnv(
            dict(flat_cfg),
            mode="bot",
            team=team,
            render_mode=None,
            bot_role=role,
        )

    return _init


def run_bot_training(cfg: Any, team: int = 0) -> None:
    """Train four PPO policies (warrior, gatherer, farmer, builder)."""
    flat = _cfg_to_dict(cfg)
    tcfg = flat["training"]
    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    ckpt_dir = Path(tcfg["checkpoint_dir"]) / "bots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_subproc = n_envs > 1
    for role in Role:
        logger.info("Training bot policy for role {}", role.name)
        fns = [_make_env_fn(flat, team, role) for _ in range(n_envs)]
        venv: DummyVecEnv | SubprocVecEnv = (
            SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)
        )
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            learning_rate=float(tcfg["learning_rate"]),
            batch_size=int(tcfg["batch_size"]),
            n_epochs=int(tcfg["n_epochs"]),
            gamma=float(tcfg["gamma"]),
            tensorboard_log=str(Path(tcfg["log_dir"]) / "tb_bots"),
        )
        cb = CheckpointCallback(
            save_freq=max(total // 10, 1000),
            save_path=str(ckpt_dir),
            name_prefix=f"ppo_{role.name.lower()}",
        )
        model.learn(total_timesteps=total, callback=cb)
        out = ckpt_dir / f"{role.name.lower()}_final"
        model.save(str(out))
        venv.close()
        logger.info("Saved {}", out)
