"""Load merged Hydra config (same as ``@hydra.main`` defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_project_config(
    project_root: Path,
    config_name: str = "default",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose ``config_name`` from ``project_root/configs`` and return a plain dict."""
    from hydra import compose, initialize_config_dir

    cfg_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides or [])
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
