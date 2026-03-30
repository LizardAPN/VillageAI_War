"""Tests for combat intents and towers."""

from typing import Any

from village_ai_war.env.combat_system import CombatSystem
from village_ai_war.state import BotState, BuildingState, BuildingType, GameState, ResourceStock, Role, VillageState


def _ccfg() -> dict[str, Any]:
    return {
        "combat": {
            "stats": {
                "warrior": {"hp": 100, "damage": 30, "attack_range": 1},
                "gatherer": {"hp": 50, "damage": 10, "attack_range": 1},
                "farmer": {"hp": 50, "damage": 10, "attack_range": 1},
                "builder": {"hp": 50, "damage": 10, "attack_range": 1},
            },
            "tower_damage": 40,
            "tower_range": 3,
        }
    }


def test_melee_kills_adjacent() -> None:
    cfg = _ccfg()
    red = BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(1, 1), hp=100, max_hp=100)
    blue = BotState(bot_id=1, team=1, role=Role.WARRIOR, position=(2, 1), hp=20, max_hp=20)
    g = GameState(
        map_size=8,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(team=0, resources=ResourceStock(), bots=[red]),
            VillageState(team=1, resources=ResourceStock(), bots=[blue]),
        ],
        next_bot_id=2,
    )
    ev = CombatSystem.apply_melee_intents(g, cfg, [(0, 0, (1, 0))])
    assert ev["kills"][0] >= 1
    assert not g.villages[1].bots[0].is_alive


def test_tower_shoots() -> None:
    cfg = _ccfg()
    tower = BuildingState(
        building_id=0,
        team=0,
        building_type=BuildingType.TOWER,
        position=(0, 0),
        hp=100,
        max_hp=100,
    )
    enemy = BotState(bot_id=1, team=1, role=Role.WARRIOR, position=(1, 0), hp=30, max_hp=30)
    g = GameState(
        map_size=8,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(team=0, resources=ResourceStock(), bots=[], buildings=[tower]),
            VillageState(team=1, resources=ResourceStock(), bots=[enemy]),
        ],
        next_bot_id=2,
    )
    ev = CombatSystem.apply_tower_fire(g, cfg)
    assert ev["damage_taken"][1] == 40
    assert g.villages[1].bots[0].hp == 0
