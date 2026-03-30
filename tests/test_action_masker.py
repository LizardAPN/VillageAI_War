"""Tests for village action masks."""

from typing import Any

from village_ai_war.agents.action_masker import ActionMasker
from village_ai_war.agents.village_action_space import VillageActionSpace
from village_ai_war.state import (
    BotState,
    BuildingState,
    BuildingType,
    GameState,
    ResourceStock,
    Role,
    VillageState,
)


def _config(n: int = 8) -> dict[str, Any]:
    return {
        "game": {
            "max_bots_for_role_change": 8,
            "blueprint_adjacent_to_townhall": True,
        },
        "economy": {
            "bot_cost": {"wood": 50, "food": 100},
        },
        "buildings": {
            "barracks": {"hp": 100, "cost": {"wood": 1000}},
            "storage": {"hp": 100, "cost": {"wood": 50}},
            "farm": {"hp": 100, "cost": {"wood": 80}},
            "tower": {"hp": 100, "cost": {"stone": 100}},
            "wall": {"hp": 100, "cost": {"stone": 30}},
            "citadel": {"hp": 100, "cost": {"stone": 200, "wood": 150}},
        },
    }


def test_mask_shape_and_noop() -> None:
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
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(
                team=0,
                resources=ResourceStock(wood=500, stone=500, food=500),
                pop_cap=10,
                bots=[BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0))],
                buildings=[th],
            ),
            VillageState(team=1),
        ],
    )
    sp = VillageActionSpace(8, max_bots=8)
    m = ActionMasker.compute_masks(g, 0, _config(8))
    assert m.shape == (sp.n_actions,)
    assert bool(m[0])
