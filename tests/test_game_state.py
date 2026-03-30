"""Tests for Pydantic game state models."""

import pytest
from pydantic import ValidationError

from village_ai_war.state import (
    BotState,
    BuildingState,
    BuildingType,
    GameState,
    GlobalRewardMode,
    ResourceStock,
    Role,
    VillageState,
)


def test_bot_state_defaults() -> None:
    b = BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(1, 2))
    assert b.hp == 100 and b.is_alive is True


def test_village_state_nested() -> None:
    v = VillageState(
        team=0,
        resources=ResourceStock(wood=1, stone=2, food=3),
        bots=[
            BotState(bot_id=0, team=0, role=Role.GATHERER, position=(0, 0)),
        ],
    )
    assert v.resources.wood == 1
    assert len(v.bots) == 1


def test_building_progress_bounds() -> None:
    BuildingState(
        building_id=0,
        team=0,
        building_type=BuildingType.TOWNHALL,
        position=(0, 0),
        hp=100,
        max_hp=100,
        construction_progress=1.0,
    )
    with pytest.raises(ValidationError):
        BuildingState(
            building_id=0,
            team=0,
            building_type=BuildingType.TOWNHALL,
            position=(0, 0),
            hp=100,
            max_hp=100,
            construction_progress=1.5,
        )


def test_game_state_roundtrip_lists() -> None:
    n = 4
    terrain = [[0] * n for _ in range(n)]
    resources = [[0] * n for _ in range(n)]
    amounts = [[0] * n for _ in range(n)]
    gs = GameState(
        map_size=n,
        terrain=terrain,
        resources=resources,
        resource_amounts=amounts,
        villages=[
            VillageState(team=0),
            VillageState(team=1),
        ],
    )
    d = gs.model_dump()
    gs2 = GameState.model_validate(d)
    assert gs2.map_size == n
    assert len(gs2.villages) == 2


def test_global_reward_mode_intenum() -> None:
    v = VillageState(team=0, global_reward_mode=GlobalRewardMode.ATTACK)
    assert v.global_reward_mode == GlobalRewardMode.ATTACK
