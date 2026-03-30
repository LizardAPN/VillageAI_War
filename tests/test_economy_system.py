"""Tests for economy harvesting and food consumption."""

from typing import Any

import numpy as np

from village_ai_war.env.economy_system import EconomySystem
from village_ai_war.env.map_generator import generate_initial_state
from village_ai_war.state import (
    BotState,
    BuildingState,
    BuildingType,
    GameState,
    ResourceLayer,
    ResourceStock,
    Role,
    VillageState,
)


def _minimal_config() -> dict[str, Any]:
    return {
        "map": {
            "size": 8,
            "resource_density": 0.1,
            "mountain_density": 0.02,
            "resource_capacity": {"forest": 100, "stone": 50, "field": 999},
        },
        "economy": {
            "harvest_interval": 1,
            "harvest_amount": 5,
            "food_consumption": 1,
            "hunger_damage": 5,
            "bot_cost": {"wood": 10, "food": 10},
            "bot_spawn_delay": 2,
            "farm_food_bonus": 0.5,
        },
        "combat": {
            "stats": {
                "warrior": {"hp": 100, "damage": 1, "attack_range": 1},
                "gatherer": {"hp": 100, "damage": 1, "attack_range": 1},
                "farmer": {"hp": 100, "damage": 1, "attack_range": 1},
                "builder": {"hp": 100, "damage": 1, "attack_range": 1},
            }
        },
        "buildings": {
            "townhall": {"hp": 100, "cost": {}},
            "barracks": {"hp": 50, "cost": {"wood": 1}},
            "storage": {"hp": 50, "cost": {"wood": 1}},
            "farm": {"hp": 50, "cost": {"wood": 1}},
            "tower": {"hp": 50, "cost": {"stone": 1}},
            "wall": {"hp": 50, "cost": {"stone": 1}},
            "citadel": {"hp": 50, "cost": {"stone": 1}},
        },
        "game": {
            "max_ticks": 100,
            "initial_resources": {"wood": 0, "stone": 0, "food": 100},
            "initial_bots": 0,
            "initial_buildings": [],
        },
    }


def test_gatherer_collects_wood() -> None:
    cfg = _minimal_config()
    g = GameState(
        map_size=8,
        max_ticks=100,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(
                team=0,
                resources=ResourceStock(wood=0, stone=0, food=50),
                bots=[
                    BotState(bot_id=0, team=0, role=Role.GATHERER, position=(1, 1)),
                ],
            ),
            VillageState(team=1),
        ],
        next_bot_id=1,
    )
    g.resources[1][1] = int(ResourceLayer.FOREST)
    g.resource_amounts[1][1] = 20
    ev = EconomySystem.step(g, cfg)
    assert g.villages[0].resources.wood > 0
    assert ev["wood_delta"][0] > 0


def test_hunger_when_no_food() -> None:
    cfg = _minimal_config()
    g = GameState(
        map_size=8,
        max_ticks=100,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(
                team=0,
                resources=ResourceStock(food=0),
                bots=[BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0), hp=20)],
            ),
            VillageState(team=1),
        ],
        next_bot_id=1,
    )
    EconomySystem.step(g, cfg)
    assert g.villages[0].bots[0].hp < 20


def test_map_generator_feeds_economy() -> None:
    cfg = _minimal_config()
    cfg["map"]["size"] = 16
    cfg["game"]["initial_bots"] = 2
    cfg["game"]["initial_buildings"] = ["barracks"]
    rng = np.random.default_rng(0)
    state = generate_initial_state(cfg, rng)
    assert len(state.villages) == 2
    EconomySystem.step(state, cfg)
    assert state.tick == 0


def test_queue_recruit() -> None:
    cfg = _minimal_config()
    th = BuildingState(
        building_id=0,
        team=0,
        building_type=BuildingType.TOWNHALL,
        position=(3, 3),
        hp=100,
        max_hp=100,
    )
    g = GameState(
        map_size=8,
        max_ticks=100,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(
                team=0,
                resources=ResourceStock(wood=100, stone=0, food=100),
                pop_cap=5,
                bots=[
                    BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0)),
                ],
                buildings=[th],
            ),
            VillageState(team=1),
        ],
        next_bot_id=1,
    )
    ok = EconomySystem.queue_recruit(g, 0, Role.GATHERER, cfg)
    assert ok
    assert g.villages[0].spawn_queue_ticks_remaining == 2
    EconomySystem.step(g, cfg)
    EconomySystem.step(g, cfg)
    assert any(b.bot_id != 0 for b in g.villages[0].bots)
