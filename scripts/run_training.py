#!/usr/bin/env python3
"""Launch training stages via Hydra (``training.stage`` 0 unified, 1–3 legacy)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from village_ai_war.training.train_bots_selfplay import run_bots_selfplay_training
from village_ai_war.training.train_joint import run_joint_training
from village_ai_war.training.train_unified import run_unified_training
from village_ai_war.training.train_village_selfplay import run_village_selfplay_training


class _InterceptHandler(logging.Handler):
    """Send stdlib logging (e.g. Stable-Baselines3) through Loguru so one file sink works."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _configure_run_log_file() -> None:
    """Hydra's default ``run_training.log`` only hooks stdlib logging; we use Loguru."""
    out = Path(HydraConfig.get().runtime.output_dir)
    log_path = out / "run_training.log"
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="DEBUG",
        enqueue=True,
    )
    logging.root.handlers.clear()
    logging.root.addHandler(_InterceptHandler())
    logging.root.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    """Dispatch to stage trainers."""
    _configure_run_log_file()
    flat = OmegaConf.to_container(cfg, resolve=True)

    stage = int(flat["training"]["stage"])
    logger.info("Starting training stage {}", stage)
    if stage == 0:
        run_unified_training(cfg)
    elif stage == 1:
        run_bots_selfplay_training(cfg)
    elif stage == 2:
        run_village_selfplay_training(cfg)
    elif stage == 3:
        run_joint_training(cfg)
    else:
        raise ValueError(f"Unknown training.stage={stage}")


if __name__ == "__main__":
    main()
