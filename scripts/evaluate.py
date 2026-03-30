#!/usr/bin/env python3
"""Run N episodes and report win rate, mean return, and length."""

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
    n_episodes = 100
    team = 0
    returns: list[float] = []
    lengths: list[int] = []
    wins = 0

    env = GameEnv(flat, mode="village", team=team, render_mode=None)
    rng = np.random.default_rng(123)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))
        done = False
        G = 0.0
        steps = 0
        while not done:
            m = env.action_masks()
            a = int(rng.choice(np.flatnonzero(m)))
            obs, r, term, trunc, info = env.step(a)
            G += float(r)
            steps += 1
            done = term or trunc
        returns.append(G)
        lengths.append(steps)
        st = env._state
        if st is not None and st.winner == team:
            wins += 1
    env.close()

    logger.info(
        "Episodes={} win_rate={:.3f} mean_return={:.3f} mean_len={:.1f}",
        n_episodes,
        wins / n_episodes,
        float(np.mean(returns)),
        float(np.mean(lengths)),
    )


if __name__ == "__main__":
    main()
