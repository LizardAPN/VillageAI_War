#!/usr/bin/env python3
"""Launch training stages via Hydra (``training.stage`` 1 / 2 / 3)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from village_ai_war.training.train_bots_selfplay import run_bots_selfplay_training
from village_ai_war.training.train_joint import run_joint_training
from village_ai_war.training.train_village_selfplay import run_village_selfplay_training


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    """Dispatch to stage trainers."""
    flat = OmegaConf.to_container(cfg, resolve=True)
    use_wb = bool(flat.get("logging", {}).get("use_wandb", False))
    if use_wb:
        try:
            import wandb

            wandb.init(
                project=str(flat.get("logging", {}).get("project_name", "village-ai-war")),
                config=flat,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("wandb init failed: {}", e)

    stage = int(flat["training"]["stage"])
    logger.info("Starting training stage {}", stage)
    if stage == 1:
        run_bots_selfplay_training(cfg)
    elif stage == 2:
        run_village_selfplay_training(cfg)
    elif stage == 3:
        run_joint_training(cfg)
    else:
        raise ValueError(f"Unknown training.stage={stage}")


if __name__ == "__main__":
    main()
