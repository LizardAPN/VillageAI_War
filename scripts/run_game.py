#!/usr/bin/env python3
"""Play a match with optional trained village / bot policies (pygame if available)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from village_ai_war.config_load import load_project_config  # noqa: E402
from village_ai_war.env.game_env import GameEnv  # noqa: E402


def _resolve_ckpt(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.suffix and not p.exists():
        p = p.with_suffix(".zip")
    return p if p.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Village AI War with optional RL checkpoints.")
    parser.add_argument(
        "--village-checkpoint",
        default="checkpoints/village/village_final.zip",
        help="MaskablePPO zip for red manager; if missing, random valid actions.",
    )
    parser.add_argument(
        "--opponent-village-checkpoint",
        default="",
        help="MaskablePPO zip for blue manager; if empty or missing, random valid actions.",
    )
    parser.add_argument(
        "--bot-checkpoint",
        default="checkpoints/bots/bot_final.zip",
        help="PPO zip for low-level bots; if missing, random bot moves.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy.predict for loaded MaskablePPO.",
    )
    args = parser.parse_args()

    flat = load_project_config(_ROOT)
    bot_path = _resolve_ckpt(args.bot_checkpoint)
    if bot_path is not None:
        game = dict(flat.get("game", {}))
        game["bot_rl_checkpoint"] = str(bot_path)
        flat = {**flat, "game": game}

    village_path = _resolve_ckpt(args.village_checkpoint)
    opp_path = _resolve_ckpt(args.opponent_village_checkpoint or None)

    red_model = None
    blue_model = None
    if village_path is not None:
        try:
            from sb3_contrib import MaskablePPO

            red_model = MaskablePPO.load(str(village_path), device="auto")
            logger.info("Loaded red village policy from {}", village_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load red village policy ({}); using random red actions", e)
    if opp_path is not None:
        try:
            from sb3_contrib import MaskablePPO

            blue_model = MaskablePPO.load(str(opp_path), device="auto")
            logger.info("Loaded blue village policy from {}", opp_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load blue village policy ({}); using random blue actions", e)

    bot_policy = None
    if bot_path is not None:
        try:
            from stable_baselines3 import PPO

            bot_policy = PPO.load(str(bot_path), device="auto")
            logger.info("Loaded bot policy from {}", bot_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load bot policy ({}); using random bot moves", e)

    try:
        env = GameEnv(flat, mode="village", team=0, render_mode="human")
    except Exception as e:  # noqa: BLE001
        logger.warning("pygame/human render unavailable ({}); using rgb_array off-screen", e)
        env = GameEnv(flat, mode="village", team=0, render_mode=None)

    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    use_trained_tick = red_model is not None or blue_model is not None or bot_policy is not None

    for t in range(args.max_steps):
        if use_trained_tick:
            m0 = env.action_masks(team=0)
            m1 = env.action_masks(team=1)
            obs0 = env.get_village_observation(0)
            if red_model is not None:
                a0, _ = red_model.predict(
                    obs0,
                    action_masks=m0,
                    deterministic=args.deterministic,
                )
                a0 = int(np.asarray(a0).reshape(-1)[0])
            else:
                a0 = int(rng.choice(np.flatnonzero(m0)))
            if blue_model is not None:
                obs1 = env.get_village_observation(1)
                a1, _ = blue_model.predict(
                    obs1,
                    action_masks=m1,
                    deterministic=args.deterministic,
                )
                a1 = int(np.asarray(a1).reshape(-1)[0])
            else:
                a1 = int(rng.choice(np.flatnonzero(m1)))
            obs, r, term, trunc, info = env.run_bots_then_village_decisions(bot_policy, a0, a1)
        else:
            m = env.action_masks()
            a = int(rng.choice(np.flatnonzero(m)))
            obs, r, term, trunc, info = env.step(a)
        if env.render_mode == "human":
            env.render()
        if term or trunc:
            logger.info("Done at t={} info={}", t, info)
            break
    env.close()


if __name__ == "__main__":
    main()
