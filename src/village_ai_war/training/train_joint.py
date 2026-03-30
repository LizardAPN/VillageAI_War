"""Stage 3: joint fine-tuning (village MaskablePPO with reduced learning rate).

    Low-level units are controlled by the frozen stage-1 bot policy when
    ``checkpoints/bots/bot_final.zip`` exists; otherwise actions are random.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from village_ai_war.env.game_env import GameEnv


class _MaskableMonitor(Monitor):
    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def run_joint_training(cfg: Any, team: int = 0) -> None:
    """Fine-tune village MaskablePPO in ``full`` mode (RL bots, scripted opponent manager)."""
    flat = _cfg_to_dict(cfg)
    game = dict(flat.get("game", {}))
    bot_ckpt = Path(flat["training"]["checkpoint_dir"]) / "bots" / "bot_final.zip"
    game["bot_rl_checkpoint"] = str(bot_ckpt)
    flat = {**flat, "game": game}
    tcfg = flat["training"]
    total = int(tcfg["total_timesteps"])
    base_lr = float(tcfg["learning_rate"])
    lr = base_lr * float(tcfg.get("bot_joint_lr_factor", 0.1))
    ckpt_dir = Path(tcfg["checkpoint_dir"]) / "joint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def make_env() -> _MaskableMonitor:
        env = GameEnv(dict(flat), mode="full", team=team, render_mode=None)
        return _MaskableMonitor(env)

    venv = DummyVecEnv([make_env])
    prev = ckpt_dir.parent / "village" / "village_final.zip"
    if prev.exists():
        model = MaskablePPO.load(str(prev), env=venv)
        model.learning_rate = lr
        logger.info("Loaded village checkpoint for joint fine-tune")
    else:
        logger.warning("No village checkpoint at {}; training from scratch", prev)
        model = MaskablePPO(
            "MultiInputPolicy",
            venv,
            verbose=1,
            learning_rate=lr,
            batch_size=int(tcfg["batch_size"]),
            n_epochs=int(tcfg["n_epochs"]),
            gamma=float(tcfg["gamma"]),
        )
    cb = CheckpointCallback(
        save_freq=max(total // 5, 500),
        save_path=str(ckpt_dir),
        name_prefix="joint_maskable_ppo",
    )
    model.learn(total_timesteps=total, callback=cb)
    model.save(str(ckpt_dir / "joint_final"))
    venv.close()
    logger.info("Joint training saved to {}", ckpt_dir / "joint_final")
