"""Smoke tests for GameEnv without display."""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from village_ai_war.env.game_env import GameEnv


def _tiny_config() -> dict[str, Any]:
    return {
        "map": {
            "size": 12,
            "seed": 0,
            "resource_density": 0.1,
            "mountain_density": 0.02,
            "resource_capacity": {"forest": 100, "stone": 50, "field": 999},
        },
        "economy": {
            "harvest_interval": 3,
            "harvest_amount": 5,
            "food_consumption": 1,
            "hunger_damage": 5,
            "bot_cost": {"wood": 50, "food": 100},
            "bot_spawn_delay": 2,
            "farm_food_bonus": 0.5,
        },
        "combat": {
            "stats": {
                "warrior": {"hp": 100, "damage": 10, "attack_range": 1},
                "gatherer": {"hp": 80, "damage": 8, "attack_range": 1},
                "farmer": {"hp": 70, "damage": 5, "attack_range": 1},
                "builder": {"hp": 80, "damage": 8, "attack_range": 1},
            },
            "tower_damage": 15,
            "tower_range": 3,
        },
        "buildings": {
            "townhall": {"hp": 500, "cost": {}},
            "barracks": {"hp": 100, "cost": {"wood": 100}},
            "storage": {"hp": 100, "cost": {"wood": 50}},
            "farm": {"hp": 100, "cost": {"wood": 80}},
            "tower": {"hp": 100, "cost": {"stone": 100}},
            "wall": {"hp": 100, "cost": {"stone": 30}},
            "citadel": {"hp": 100, "cost": {"stone": 200, "wood": 150}},
            "citadel_pop_bonus": 5,
        },
        "game": {
            "max_ticks": 50,
            "manager_interval": 5,
            "initial_resources": {"wood": 200, "stone": 100, "food": 500},
            "initial_bots": 4,
            "initial_buildings": ["barracks", "storage"],
            "blueprint_adjacent_to_townhall": True,
            "max_bots_for_role_change": 16,
        },
        "rewards": {
            "bot": {
                "alpha": 0.7,
                "warrior": {
                    "damage_dealt": 0.1,
                    "kill": 5.0,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "gatherer": {"resource_collected": 0.5, "damage_taken": -0.05, "death": -10.0, "noop": -0.01},
                "farmer": {"food_produced": 0.5, "damage_taken": -0.05, "death": -10.0, "noop": -0.01},
                "builder": {
                    "block_placed": 2.0,
                    "repair_pct": 0.1,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "global_modes": {"defend_coeff": -0.05, "attack_coeff": 0.05, "gather_coeff": 0.1},
            },
            "village": {
                "economy_coeff": 0.01,
                "kill_reward": 5.0,
                "loss_penalty": -3.0,
                "building_reward": 10.0,
                "stagnation_penalty": -0.05,
                "stagnation_threshold": 50,
                "win": 1000.0,
                "loss": -1000.0,
            },
        },
        "rendering": {"cell_size": 16, "fps": 60},
    }


def test_village_reset_step() -> None:
    env = GameEnv(_tiny_config(), mode="village", team=0, render_mode=None)
    obs, info = env.reset(seed=1)
    assert "map" in obs and "village" in obs
    masks = env.action_masks()
    assert masks.any()
    obs2, r, term, trunc, info2 = env.step(0)
    assert isinstance(r, float)
    assert "kills_this_tick" in info2


def test_bot_reset_step() -> None:
    env = GameEnv(_tiny_config(), mode="bot", team=0, render_mode=None)
    obs, _ = env.reset(seed=2)
    assert obs.shape == (181,)
    obs2, r, term, trunc, _ = env.step(0)
    assert len(obs2) == 181


def test_step_with_opponent_and_action_masks_team() -> None:
    cfg = _tiny_config()
    cfg["game"]["initial_bots"] = 1
    env = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step_with_opponent(0, 0)
    assert obs.shape == (181,)
    assert "kills_this_tick" in info
    assert "winner" in info

    venv = GameEnv(cfg, mode="village", team=0, render_mode=None)
    venv.reset(seed=0)
    m0 = venv.action_masks()
    m1 = venv.action_masks(team=1)
    assert m0.shape == m1.shape
    assert m0.dtype == bool


def test_mutual_extinction_terminates_episode() -> None:
    """Both teams with zero alive bots must end the episode (terminated), not run forever."""
    env = GameEnv(_tiny_config(), mode="village", team=0, render_mode=None)
    env.reset(seed=0)
    assert env._state is not None
    env._state.tick = 1  # not a manager tick; only noop village actions valid
    for v in env._state.villages:
        for b in v.bots:
            b.is_alive = False
    m0 = env.action_masks(team=0)
    m1 = env.action_masks(team=1)
    a0 = int(np.argmax(m0))
    a1 = int(np.argmax(m1))
    _obs, _r, term, trunc, info = env.run_bots_then_village_decisions(None, a0, a1)
    assert term is True
    assert trunc is False
    assert info.get("winner") is None


def test_get_village_observation_and_run_bots_then_village() -> None:
    env = GameEnv(_tiny_config(), mode="village", team=0, render_mode=None)
    env.reset(seed=3)
    o0 = env.get_village_observation(0)
    o1 = env.get_village_observation(1)
    assert "map" in o0 and "village" in o0
    assert o0["map"].shape == o1["map"].shape
    m0 = env.action_masks(team=0)
    m1 = env.action_masks(team=1)
    a0 = int(np.flatnonzero(m0)[0])
    a1 = int(np.flatnonzero(m1)[0])
    obs, r, term, trunc, info = env.run_bots_then_village_decisions(None, a0, a1)
    assert "map" in obs and "village" in obs
    assert isinstance(r, float)
    assert "kills_this_tick" in info
