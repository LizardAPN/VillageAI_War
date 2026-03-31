#!/usr/bin/env python3
"""Play a match with optional trained village / bot policies, human vs AI, or vs MAPPO."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from village_ai_war.config_load import load_project_config  # noqa: E402
from village_ai_war.env.game_env import GameEnv  # noqa: E402
from village_ai_war.play.human_controls import (  # noqa: E402
    collect_blue_bot_actions_for_tick,
    collect_team_bot_actions_for_tick,
    collect_village_action_for_tick,
)
from village_ai_war.play.mappo_human_tick import play_mappo_human_tick  # noqa: E402
from village_ai_war.training.self_play_env import _maskable_village_obs_matches_env  # noqa: E402


def _resolve_ckpt(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.suffix and not p.exists():
        p = p.with_suffix(".zip")
    return p if p.is_file() else None


def _load_mappo_policy(path: Path, _flat: dict) -> object:
    import village_ai_war.models.mappo_policy  # noqa: F401 — SB3 unpickle

    from stable_baselines3 import PPO

    return PPO.load(
        str(path),
        device="auto",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pygame.pkgdata",
    )
    parser = argparse.ArgumentParser(description="Run Village AI War with optional RL checkpoints.")
    parser.add_argument(
        "--mappo-opponent",
        default="",
        help="Path to MAPPO zip (e.g. checkpoints/bots_mappo/mappo_bot_final.zip). "
        "Human plays BLUE vs MAPPO on RED; no village AI (training-faithful micro).",
    )
    parser.add_argument(
        "--human",
        choices=("red", "blue", ""),
        default="",
        help="Play as this team (village + bots). Leave empty for AI-only demo. "
        "With --mappo-opponent only 'blue' is allowed.",
    )
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
        help="Use deterministic policy.predict for loaded policies.",
    )
    parser.add_argument(
        "--human-3d",
        action="store_true",
        help="OpenGL 3D board; not used with --mappo-opponent or --human (2D only).",
    )
    args = parser.parse_args()

    flat = load_project_config(_ROOT)
    mappo_path = _resolve_ckpt(args.mappo_opponent or None)
    human_side = args.human or None

    if str(args.mappo_opponent).strip() and mappo_path is None:
        logger.error("MAPPO checkpoint not found: {}", args.mappo_opponent)
        sys.exit(1)

    if mappo_path is not None:
        if args.human == "red":
            logger.error("MAPPO opponent only supports playing as BLUE (team 1).")
            sys.exit(2)
        human_side = "blue"
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

    # --- Legacy AI demo or human vs MaskablePPO + PPO bots ---
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

    if human_side and args.human_3d:
        logger.warning("Human play uses 2D pygame; ignoring --human-3d.")
    render_mode = "human" if human_side or not args.human_3d else "human_3d"

    try:
        env = GameEnv(flat, mode="village", team=0, render_mode=render_mode)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Display render unavailable ({}); falling back to no window",
            e,
        )
        env = GameEnv(flat, mode="village", team=0, render_mode=None)

    env_obs_space = env.observation_space
    if red_model is not None and not _maskable_village_obs_matches_env(red_model, env_obs_space):
        logger.warning(
            "Red village checkpoint {} does not match game observation space (e.g. map.size); "
            "policy obs {} != env {} — using random red manager actions",
            village_path,
            red_model.observation_space,
            env_obs_space,
        )
        red_model = None
    if blue_model is not None and not _maskable_village_obs_matches_env(blue_model, env_obs_space):
        logger.warning(
            "Blue village checkpoint {} does not match game observation space; "
            "policy obs {} != env {} — using random blue manager actions",
            opp_path,
            blue_model.observation_space,
            env_obs_space,
        )
        blue_model = None

    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    use_trained_tick = red_model is not None or blue_model is not None or bot_policy is not None
    noop_v = int(env._village_space.offset_noop)  # noqa: SLF001

    if human_side and env.render_mode is None:
        logger.error("Human play needs a display (pygame 2D window).")
        sys.exit(1)

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
                obs, _ = env.reset(seed=args.seed)
            else:
                raise

    import pygame  # noqa: PLC0415

    def _render_v(overlay_lines: tuple[str, ...] | None = None) -> None:
        if env.render_mode is not None:
            env.render(overlay_lines=overlay_lines or ())

    def _render_v_cb(overlay_lines: tuple[str, ...] = ()) -> None:
        _render_v(overlay_lines)

    if env.render_mode is not None:
        logger.info(
            "Viewer render_mode={} | max_steps={} | close the window or Ctrl+C to stop early",
            env.render_mode,
            args.max_steps,
        )

    human_team: int | None = None
    if human_side == "red":
        human_team = 0
    elif human_side == "blue":
        human_team = 1

    for t in range(args.max_steps):
        st = env.game_state
        assert st is not None
        interval = int(flat["game"]["manager_interval"])
        is_mgr = st.tick % interval == 0

        if human_team is not None:
            h_bots = collect_team_bot_actions_for_tick(
                env,
                pygame,
                human_team,
                render=_render_v_cb,
            )
            if is_mgr:
                a_human = collect_village_action_for_tick(
                    env,
                    human_team,
                    pygame,
                    render=_render_v_cb,
                )
            else:
                a_human = noop_v
            if human_team == 0:
                a0 = a_human
                m1 = env.action_masks(team=1)
                if blue_model is not None:
                    o1 = env.get_village_observation(1)
                    p1, _ = blue_model.predict(
                        o1,
                        action_masks=m1,
                        deterministic=args.deterministic,
                    )
                    a1 = int(np.asarray(p1).reshape(-1)[0])
                else:
                    a1 = int(rng.choice(np.flatnonzero(m1)))
            else:
                a1 = a_human
                m0 = env.action_masks(team=0)
                if red_model is not None:
                    o0 = env.get_village_observation(0)
                    p0, _ = red_model.predict(
                        o0,
                        action_masks=m0,
                        deterministic=args.deterministic,
                    )
                    a0 = int(np.asarray(p0).reshape(-1)[0])
                else:
                    a0 = int(rng.choice(np.flatnonzero(m0)))
            obs, r, term, trunc, info = env.run_bots_then_village_decisions(
                bot_policy,
                a0,
                a1,
                human_team=human_team,
                human_bot_actions=h_bots,
            )
        elif use_trained_tick:
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
        if env.render_mode is not None:
            env.render()
        if term or trunc:
            logger.info("Done at t={} info={}", t, info)
            break
    env.close()


if __name__ == "__main__":
    main()
