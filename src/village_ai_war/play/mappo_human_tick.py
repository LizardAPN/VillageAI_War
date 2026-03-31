"""One simulation tick: MAPPO (red) vs human bot actions (blue), village-free."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, SupportsFloat

import numpy as np

from village_ai_war.env.game_env import GameEnv
from village_ai_war.play.mappo_obs import (
    build_mappo_global_state,
    build_mappo_locals_matrix,
    pack_mappo_observation_vector,
)


def play_mappo_human_tick(
    env: GameEnv,
    mappo_model: Any,
    human_blue_actions: Mapping[int, int],
    *,
    n_bot_slots: int,
    deterministic: bool = False,
) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
    """Mirror ``MAPPOBotEnv.step`` with team-0 MAPPO and explicit blue actions.

    Requires ``env.mode == \"bot\"`` and ``env.team == 0`` (MAPPO training layout).
    ``human_blue_actions`` must include every alive blue ``bot_id`` -> action in ``0..11``.
    """
    if env.mode != "bot":
        raise ValueError("play_mappo_human_tick requires GameEnv mode='bot'")
    if env.team != 0:
        raise ValueError("play_mappo_human_tick requires GameEnv team=0 (MAPPO trained side)")
    state = env.game_state
    assert state is not None

    blue_alive = sorted(
        (b for b in state.villages[1].bots if b.is_alive),
        key=lambda b: int(b.bot_id),
    )
    for b in blue_alive:
        if int(b.bot_id) not in human_blue_actions:
            raise ValueError(
                f"human_blue_actions missing bot_id={b.bot_id}; need all alive blue bots"
            )

    red_alive = sorted(
        (b for b in state.villages[0].bots if b.is_alive),
        key=lambda b: int(b.bot_id),
    )

    gs = build_mappo_global_state(state, env._vil_obs)
    mat = build_mappo_locals_matrix(
        state,
        env,
        mappo_team=0,
        n_bot_slots=n_bot_slots,
    )
    packed = pack_mappo_observation_vector(mat, gs)
    acts, _ = mappo_model.predict(packed, deterministic=deterministic)
    acts = np.asarray(acts, dtype=np.int64).reshape(-1)
    if acts.shape[0] != n_bot_slots:
        raise ValueError(f"MAPPO expected {n_bot_slots} actions, got {acts.shape[0]}")

    env.snapshot_bot_positions_for_tick()
    env.begin_mappo_tick()

    controlled: list[tuple[int, int]] = []
    for i, bot in enumerate(red_alive[:n_bot_slots]):
        a = int(acts[i])
        env.queue_bot_action(0, bot.bot_id, a)
        controlled.append((bot.bot_id, a))
    if controlled:
        env._controlled_bot_id = controlled[0][0]

    for bot in blue_alive:
        env.queue_bot_action(1, bot.bot_id, int(human_blue_actions[int(bot.bot_id)]))

    learner_bot_actions = {bid: ac for bid, ac in controlled}
    return env._simulation_tick(
        manager_action=None,
        learner_bot_action=None,
        learner_bot_actions=learner_bot_actions,
    )


def play_mappo_self_play_tick(
    env: GameEnv,
    mappo_model: Any,
    *,
    n_bot_slots: int,
    deterministic: bool = False,
) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
    """One tick: the same MAPPO policy controls RED and BLUE (no human).

    Uses the same packed observation layout as training: fixed team-0 map / village
    order in the global tail (:func:`build_mappo_global_state`), with local bot
    slots built separately for team 0 and team 1.

    Requires ``env.mode == \"bot\"`` and ``env.team == 0``.
    """
    if env.mode != "bot":
        raise ValueError("play_mappo_self_play_tick requires GameEnv mode='bot'")
    if env.team != 0:
        raise ValueError("play_mappo_self_play_tick requires GameEnv team=0")
    state = env.game_state
    assert state is not None

    gs = build_mappo_global_state(state, env._vil_obs)

    def _acts_for_team(mappo_team: int) -> np.ndarray:
        mat = build_mappo_locals_matrix(
            state,
            env,
            mappo_team=mappo_team,
            n_bot_slots=n_bot_slots,
        )
        packed = pack_mappo_observation_vector(mat, gs)
        acts, _ = mappo_model.predict(packed, deterministic=deterministic)
        return np.asarray(acts, dtype=np.int64).reshape(-1)

    acts_red = _acts_for_team(0)
    acts_blue = _acts_for_team(1)
    if acts_red.shape[0] != n_bot_slots:
        raise ValueError(
            f"MAPPO expected {n_bot_slots} red actions, got {acts_red.shape[0]}"
        )
    if acts_blue.shape[0] != n_bot_slots:
        raise ValueError(
            f"MAPPO expected {n_bot_slots} blue actions, got {acts_blue.shape[0]}"
        )

    red_alive = sorted(
        (b for b in state.villages[0].bots if b.is_alive),
        key=lambda b: int(b.bot_id),
    )
    blue_alive = sorted(
        (b for b in state.villages[1].bots if b.is_alive),
        key=lambda b: int(b.bot_id),
    )

    env.snapshot_bot_positions_for_tick()
    env.begin_mappo_tick()

    controlled: list[tuple[int, int]] = []
    for i, bot in enumerate(red_alive[:n_bot_slots]):
        a = int(acts_red[i])
        env.queue_bot_action(0, bot.bot_id, a)
        controlled.append((bot.bot_id, a))
    if controlled:
        env._controlled_bot_id = controlled[0][0]

    for i, bot in enumerate(blue_alive[:n_bot_slots]):
        env.queue_bot_action(1, bot.bot_id, int(acts_blue[i]))

    learner_bot_actions = {bid: ac for bid, ac in controlled}
    return env._simulation_tick(
        manager_action=None,
        learner_bot_action=None,
        learner_bot_actions=learner_bot_actions,
    )
