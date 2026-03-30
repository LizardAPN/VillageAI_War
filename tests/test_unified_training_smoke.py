"""Smoke tests for the unified training env and village self-play with empty pools."""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from village_ai_war.training.self_play_env import SelfPlayVillageEnv, UnifiedBotSelfPlayEnv


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
            "initial_bots": 1,
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


def test_unified_bot_env_reset_step(tmp_path: Any) -> None:
    """UnifiedBotSelfPlayEnv should reset and step without errors using empty pools."""
    cfg = _tiny_config()
    env = UnifiedBotSelfPlayEnv(
        cfg,
        bot_policy_holder=None,
        village_checkpoint_path=str(tmp_path / "no_village"),
        opponent_bot_pool_dir=str(tmp_path / "pool_bots"),
        opponent_village_pool_dir=str(tmp_path / "pool_village"),
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (181,), f"Expected bot obs shape (181,), got {obs.shape}"
    for _ in range(10):
        action = int(np.random.randint(0, env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (181,)
        assert isinstance(reward, float)
        if terminated or truncated:
            obs, info = env.reset(seed=43)
    env.close()


def test_unified_bot_env_multi_bots(tmp_path: Any) -> None:
    """With initial_bots > 1, friendly bots fall back to random actions when holder is empty."""
    cfg = _tiny_config()
    cfg["game"]["initial_bots"] = 4
    env = UnifiedBotSelfPlayEnv(
        cfg,
        bot_policy_holder={"model": None},
        village_checkpoint_path=str(tmp_path / "no_village"),
        opponent_bot_pool_dir=str(tmp_path / "pool_bots"),
        opponent_village_pool_dir=str(tmp_path / "pool_village"),
    )
    obs, _ = env.reset(seed=7)
    assert obs.shape == (181,)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(0)
        if term or trunc:
            obs, _ = env.reset(seed=8)
    env.close()


def test_selfplay_village_env_empty_pools(tmp_path: Any) -> None:
    """SelfPlayVillageEnv should work with no bot checkpoint and empty opponent pool."""
    cfg = _tiny_config()
    env = SelfPlayVillageEnv(
        cfg,
        bot_checkpoint_dir=str(tmp_path / "no_bots"),
        opponent_pool_dir=str(tmp_path / "pool_village"),
    )
    obs, _ = env.reset(seed=10)
    assert "map" in obs and "village" in obs
    masks = env.action_masks()
    assert masks.any()
    for _ in range(5):
        action = int(np.flatnonzero(env.action_masks())[0])
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=11)
    env.close()
