#!/usr/bin/env python3
"""Play a short match with random or loaded policies (pygame if available)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from loguru import logger

from village_ai_war.config_load import load_project_config
from village_ai_war.env.game_env import GameEnv


def main() -> None:
    flat = load_project_config(_ROOT)

    try:
        env = GameEnv(flat, mode="village", team=0, render_mode="human")
    except Exception as e:  # noqa: BLE001
        logger.warning("pygame/human render unavailable ({}); using rgb_array off-screen", e)
        env = GameEnv(flat, mode="village", team=0, render_mode=None)

    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for t in range(500):
        m = env.action_masks()
        valid = np.flatnonzero(m)
        a = int(rng.choice(valid))
        obs, r, term, trunc, info = env.step(a)
        if env.render_mode == "human":
            env.render()
        if term or trunc:
            logger.info("Done at t={} info={}", t, info)
            break
    env.close()


if __name__ == "__main__":
    main()
