#!/usr/bin/env python3
"""Play: human vs MAPPO (2D), MAPPO vs same MAPPO (no human), or random village demo."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from loguru import logger

from village_ai_war.config_load import load_project_config
from village_ai_war.env.game_env import GameEnv
from village_ai_war.play.human_controls import collect_blue_bot_actions_for_tick
from village_ai_war.play.mappo_human_tick import (
    play_mappo_human_tick,
    play_mappo_self_play_tick,
)


def _resolve_ckpt(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.suffix and not p.exists():
        p = p.with_suffix(".zip")
    return p if p.is_file() else None


def _load_mappo_policy(path: Path, _flat: dict) -> object:
    from stable_baselines3 import PPO

    import village_ai_war.models.mappo_policy  # noqa: F401 — SB3 unpickle

    return PPO.load(
        str(path),
        device="auto",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )


def _run_random_village_demo(
    flat: dict,
    *,
    seed: int,
    max_steps: int,
    human_3d: bool,
) -> None:
    render_mode = "human_3d" if human_3d else "human"
    try:
        env = GameEnv(flat, mode="village", team=0, render_mode=render_mode)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Display render unavailable ({}); falling back to no window",
            e,
        )
        env = GameEnv(flat, mode="village", team=0, render_mode=None)

    # 3D render needs a real GameState; it is only set after reset().
    rng = np.random.default_rng(seed)
    _obs, _ = env.reset(seed=seed)

    if env.render_mode == "human_3d":
        try:
            env.render()
        except (OSError, RuntimeError) as e:
            msg = str(e)
            if (
                "OpenGL libraries missing" in msg
                or "Could not load OpenGL" in msg
                or "libGL" in msg
                or "libEGL" in msg
                or "libgl.so" in msg.lower()
                or "libegl.so" in msg.lower()
            ):
                logger.warning(
                    "3D view failed ({}). Install OpenGL on Linux/WSL "
                    "(e.g. sudo apt install -y libgl1 libegl1), or use 2D. "
                    "Falling back to pygame window.",
                    e,
                )
                env.close()
                env = GameEnv(flat, mode="village", team=0, render_mode="human")
                _obs, _ = env.reset(seed=seed)
            else:
                raise

    if env.render_mode is not None:
        logger.info(
            "Random village demo | render_mode={} | max_steps={} | close window or Ctrl+C to stop",
            env.render_mode,
            max_steps,
        )

    for t in range(max_steps):
        m = env.action_masks()
        a = int(rng.choice(np.flatnonzero(m)))
        _obs, _r, term, trunc, info = env.step(a)
        if env.render_mode is not None:
            env.render()
        if term or trunc:
            logger.info("Done at t={} info={}", t, info)
            break
    env.close()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pygame.pkgdata",
    )
    parser = argparse.ArgumentParser(
        description="Village AI War: human vs MAPPO, MAPPO self-play, or random village demo."
    )
    parser.add_argument(
        "--mappo-opponent",
        default="",
        help="Path to MAPPO zip (e.g. checkpoints/bots_mappo/mappo_bot_final.zip). "
        "Human plays BLUE vs MAPPO on RED; no village AI (training-faithful micro).",
    )
    parser.add_argument(
        "--mappo-self-play",
        default="",
        help="Path to MAPPO zip: RED and BLUE both use this checkpoint (no human). "
        "Opens 2D pygame if a display is available, otherwise runs headless.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic policy.predict for MAPPO opponent.",
    )
    parser.add_argument(
        "--human-3d",
        action="store_true",
        help="OpenGL 3D board for the random demo only (ignored with --mappo-opponent).",
    )
    args = parser.parse_args()

    flat = load_project_config(_ROOT)
    mappo_path = _resolve_ckpt(args.mappo_opponent or None)
    self_play_path = _resolve_ckpt(args.mappo_self_play.strip() or None)

    if str(args.mappo_opponent).strip() and str(args.mappo_self_play).strip():
        logger.error("Use only one of --mappo-opponent or --mappo-self-play.")
        sys.exit(1)

    if str(args.mappo_opponent).strip() and mappo_path is None:
        logger.error("MAPPO checkpoint not found: {}", args.mappo_opponent)
        sys.exit(1)

    if str(args.mappo_self_play).strip() and self_play_path is None:
        logger.error("MAPPO checkpoint not found: {}", args.mappo_self_play)
        sys.exit(1)

    if mappo_path is not None:
        if args.human_3d:
            logger.warning("MAPPO human play uses 2D pygame only; ignoring --human-3d.")
        render_mode = "human"
        try:
            env = GameEnv(flat, mode="bot", team=0, render_mode=render_mode)
        except Exception as e:  # noqa: BLE001
            logger.warning("Display unavailable ({}); headless MAPPO play", e)
            env = GameEnv(flat, mode="bot", team=0, render_mode=None)

        try:
            mappo_model = _load_mappo_policy(mappo_path, flat)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load MAPPO from {}: {}", mappo_path, e)
            sys.exit(1)
        logger.info("Loaded MAPPO policy from {}", mappo_path)

        if env.render_mode is None:
            logger.error("Human vs MAPPO needs a display (pygame 2D window).")
            sys.exit(1)

        n_slots = int(flat["game"]["max_bots_for_role_change"])
        env.reset(seed=args.seed)
        import pygame  # noqa: PLC0415

        def _render(overlay_lines: tuple[str, ...] | None = None) -> None:
            if env.render_mode is not None:
                env.render(overlay_lines=overlay_lines or ())

        def _render_cb(overlay_lines: tuple[str, ...] = ()) -> None:
            _render(overlay_lines)

        if env.render_mode is not None:
            logger.info(
                "Human vs MAPPO | BLUE=you | max_steps={} | ESC/close to quit",
                args.max_steps,
            )

        for t in range(args.max_steps):
            blue_actions = collect_blue_bot_actions_for_tick(
                env,
                pygame,
                render=_render_cb,
            )
            _obs, _r, term, trunc, info = play_mappo_human_tick(
                env,
                mappo_model,
                blue_actions,
                n_bot_slots=n_slots,
                deterministic=args.deterministic,
            )
            if env.render_mode is not None:
                env.render(
                    overlay_lines=(
                        f"tick={info.get('tick', '?')}  t={t}",
                        "Next: choose BLUE bot actions",
                    )
                )
            if term or trunc:
                logger.info("Done at t={} info={}", t, info)
                break
        env.close()
        return

    if self_play_path is not None:
        if args.human_3d:
            logger.warning(
                "MAPPO self-play uses 2D pygame only when a display is available; "
                "ignoring --human-3d."
            )
        try:
            env = GameEnv(flat, mode="bot", team=0, render_mode="human")
        except Exception as e:  # noqa: BLE001
            logger.warning("Display unavailable ({}); headless MAPPO self-play", e)
            env = GameEnv(flat, mode="bot", team=0, render_mode=None)

        try:
            mappo_model = _load_mappo_policy(self_play_path, flat)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load MAPPO from {}: {}", self_play_path, e)
            sys.exit(1)
        logger.info("Loaded MAPPO for self-play from {}", self_play_path)

        n_slots = int(flat["game"]["max_bots_for_role_change"])
        env.reset(seed=args.seed)

        if env.render_mode is not None:
            logger.info(
                "MAPPO vs MAPPO (same checkpoint) | max_steps={} | ESC/close to quit",
                args.max_steps,
            )
        else:
            logger.info("MAPPO self-play (headless) | max_steps={}", args.max_steps)

        for t in range(args.max_steps):
            _obs, _r, term, trunc, info = play_mappo_self_play_tick(
                env,
                mappo_model,
                n_bot_slots=n_slots,
                deterministic=args.deterministic,
            )
            if env.render_mode is not None:
                env.render(
                    overlay_lines=(
                        f"tick={info.get('tick', '?')}  t={t}",
                        "RED vs BLUE — same MAPPO weights",
                    )
                )
            if term or trunc:
                logger.info("Done at t={} info={}", t, info)
                break
        env.close()
        return

    _run_random_village_demo(
        flat,
        seed=args.seed,
        max_steps=args.max_steps,
        human_3d=args.human_3d,
    )


if __name__ == "__main__":
    main()
