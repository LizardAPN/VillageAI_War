"""Blueprint progress requires adjacent builder and scales with construction_ticks."""

from __future__ import annotations

from typing import Any

from village_ai_war.env.building_system import BuildingSystem
from village_ai_war.state import BotState, BuildingType, GameState, ResourceStock, Role, VillageState


def _bcfg() -> dict[str, Any]:
    return {
        "buildings": {
            "townhall": {"hp": 100, "cost": {}},
            "barracks": {"hp": 50, "cost": {"wood": 1}, "construction_ticks": 10},
            "storage": {"hp": 50, "cost": {"wood": 1}},
            "farm": {"hp": 50, "cost": {"wood": 1}},
            "tower": {"hp": 50, "cost": {"stone": 1}},
            "wall": {"hp": 50, "cost": {"stone": 1}},
            "citadel": {"hp": 50, "cost": {"stone": 1}},
        }
    }


def _state_with_blueprint(bp_pos: tuple[int, int], builder_pos: tuple[int, int] | None) -> GameState:
    bots: list[BotState] = []
    if builder_pos is not None:
        bots.append(
            BotState(
                bot_id=1,
                team=0,
                role=Role.BUILDER,
                position=builder_pos,
                hp=10,
                max_hp=10,
            )
        )
    return GameState(
        map_size=8,
        max_ticks=100,
        terrain=[[0] * 8 for _ in range(8)],
        resources=[[0] * 8 for _ in range(8)],
        resource_amounts=[[0] * 8 for _ in range(8)],
        villages=[
            VillageState(team=0, resources=ResourceStock(), bots=bots),
            VillageState(team=1),
        ],
        blueprints=[
            {
                "team": 0,
                "building_type": int(BuildingType.BARRACKS),
                "position": [bp_pos[0], bp_pos[1]],
                "progress": 0.0,
            }
        ],
        next_bot_id=2,
    )


def test_no_progress_without_adjacent_builder() -> None:
    cfg = _bcfg()
    g = _state_with_blueprint((3, 3), builder_pos=(0, 0))
    ev = BuildingSystem.construction_tick(g, cfg)
    assert not ev["buildings_completed"]
    assert g.blueprints and float(g.blueprints[0]["progress"]) == 0.0


def test_progress_and_block_placed_with_adjacent_builder() -> None:
    cfg = _bcfg()
    g = _state_with_blueprint((3, 3), builder_pos=(3, 4))
    ev = BuildingSystem.construction_tick(g, cfg)
    assert float(g.blueprints[0]["progress"]) > 0.0
    assert ev["block_placed_by_bot"].get(1, 0.0) > 0.0
