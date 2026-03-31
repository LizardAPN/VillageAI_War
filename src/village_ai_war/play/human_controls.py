"""Pygame helpers for human bot / village actions in ``run_game``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from village_ai_war.agents.village_action_space import VillageActionSpace, decode_village_action
from village_ai_war.env.game_env import GameEnv
from village_ai_war.state import Role


def _action_from_key(pygame: object, key: int) -> int | None:
    """Map key to bot action 0..11, or None if unmapped."""
    if key in (pygame.K_w, pygame.K_UP):
        return 1
    if key in (pygame.K_d, pygame.K_RIGHT):
        return 2
    if key in (pygame.K_s, pygame.K_DOWN):
        return 3
    if key in (pygame.K_a, pygame.K_LEFT):
        return 4
    if key == pygame.K_i:
        return 5
    if key == pygame.K_l:
        return 6
    if key == pygame.K_k:
        return 7
    if key == pygame.K_j:
        return 8
    if key in (pygame.K_SPACE, pygame.K_0, pygame.K_KP0):
        return 0
    if key == pygame.K_g:
        return 9
    if key == pygame.K_f:
        return 10
    if key == pygame.K_r:
        return 11
    return None


def collect_team_bot_actions_for_tick(
    env: GameEnv,
    pygame: object,
    team: int,
    *,
    render: Callable[..., None],
) -> dict[int, int]:
    """Block until actions for every alive bot on ``team`` (0=red, 1=blue)."""
    state = env.game_state
    assert state is not None
    name = "RED" if team == 0 else "BLUE"
    bots = sorted(
        (b for b in state.villages[team].bots if b.is_alive),
        key=lambda b: int(b.bot_id),
    )
    out: dict[int, int] = {}
    for bi, bot in enumerate(bots):
        tentative: int | None = None
        confirmed = False
        while not confirmed:
            hint = (
                f"{name} bot {bi + 1}/{len(bots)} id={bot.bot_id} {bot.role.name} "
                f"@ {bot.position} HP={bot.hp}"
            )
            sel = "(press key, then Enter)" if tentative is None else f"action={tentative}"
            lines = (
                hint,
                sel,
                "WASD/arrows move  IJKL melee  Space noop  G/F/R if role fits  Enter confirm",
            )
            render(overlay_lines=lines)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit(0)
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    raise SystemExit(0)
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    out[int(bot.bot_id)] = 0 if tentative is None else int(tentative)
                    confirmed = True
                    break
                a = _action_from_key(pygame, event.key)
                if a is None:
                    continue
                if a == 9 and bot.role != Role.GATHERER:
                    continue
                if a == 10 and bot.role != Role.FARMER:
                    continue
                if a == 11 and bot.role != Role.BUILDER:
                    continue
                tentative = a
    return out


def collect_blue_bot_actions_for_tick(
    env: GameEnv,
    pygame: object,
    *,
    render: Callable[..., None],
) -> dict[int, int]:
    """Convenience: team 1 (vs MAPPO on red)."""
    return collect_team_bot_actions_for_tick(env, pygame, 1, render=render)


def collect_village_action_for_tick(
    env: GameEnv,
    human_team: int,
    pygame: object,
    *,
    render: Callable[..., None],
) -> int:
    """Pick one valid village action for ``human_team`` ([/] cycle, Enter confirm)."""
    space: VillageActionSpace = env._village_space  # noqa: SLF001
    m = env.action_masks(team=human_team)
    if int(np.sum(m)) == 1 and bool(m[int(space.offset_noop)]):
        return int(space.offset_noop)
    valid = [int(i) for i in range(m.size) if m[i]]
    if not valid:
        return int(space.offset_noop)
    idx_in_list = 0
    while True:
        a = valid[idx_in_list]
        dec = decode_village_action(space, a)
        lines = (
            f"Village team {human_team} — {idx_in_list + 1}/{len(valid)}",
            f"{dec}",
            "[ / ] cycle  Enter confirm  N = noop",
        )
        render(overlay_lines=lines)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise SystemExit(0)
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return a
            if event.key == pygame.K_n:
                return int(space.offset_noop)
            if event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                idx_in_list = (idx_in_list - 1) % len(valid)
            elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                idx_in_list = (idx_in_list + 1) % len(valid)
