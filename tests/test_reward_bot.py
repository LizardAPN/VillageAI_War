"""Tests for bot reward shaping."""

from typing import Any

import pytest

from village_ai_war.rewards.bot_reward import BotRewardCalculator
from village_ai_war.state import BotState, GlobalRewardMode, ResourceStock, Role, VillageState


def _cfg() -> dict[str, Any]:
    return {
        "rewards": {
            "bot": {
                "alpha": 0.5,
                "warrior": {
                    "damage_dealt": 0.1,
                    "kill": 5.0,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "gatherer": {"resource_collected": 0.5, "noop": -0.01},
                "farmer": {"food_produced": 0.5, "noop": -0.01},
                "builder": {"block_placed": 2.0, "noop": -0.01},
                "global_modes": {
                    "defend_coeff": -0.05,
                    "attack_coeff": 0.05,
                    "gather_coeff": 0.1,
                },
            }
        }
    }


def _cfg_team_terminal() -> dict[str, Any]:
    c = _cfg()
    c["rewards"]["bot"]["team"] = {
        "hunger_damage_penalty": -0.1,
        "fed_no_hunger_bonus": 0.5,
        "food_security_coeff": 1.0,
        "food_security_threshold": 100,
        "food_delta_positive_coeff": 1.0,
    }
    c["rewards"]["bot"]["terminal"] = {"win": 10.0, "loss": -10.0, "draw": -1.0}
    return c


def test_team_addon_hunger_penalty() -> None:
    vil = VillageState(
        team=0,
        bots=[BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0))],
    )
    merged: dict[str, Any] = {"hunger_damage": {0: 10}, "food_delta": {0: 0}}
    r = BotRewardCalculator.team_addon(merged, vil, _cfg_team_terminal())
    assert r == -0.1 * 10.0


def test_team_addon_fed_and_food_delta() -> None:
    vil = VillageState(
        team=0,
        resources=ResourceStock(food=50),
        bots=[BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0))],
    )
    merged: dict[str, Any] = {"hunger_damage": {0: 0}, "food_delta": {0: 50}}
    r = BotRewardCalculator.team_addon(merged, vil, _cfg_team_terminal())
    assert r == pytest.approx(0.5 + 1.0 * 50.0 / 100.0)


def test_team_addon_no_team_section() -> None:
    vil = VillageState(team=0, bots=[])
    assert BotRewardCalculator.team_addon({"hunger_damage": {0: 99}}, vil, _cfg()) == 0.0


def test_terminal_addon() -> None:
    cfg = _cfg_team_terminal()
    assert BotRewardCalculator.terminal_addon(cfg, False, 0, 0) == 0.0
    assert BotRewardCalculator.terminal_addon(cfg, True, 0, 0) == 10.0
    assert BotRewardCalculator.terminal_addon(cfg, True, 0, 1) == -10.0
    assert BotRewardCalculator.terminal_addon(cfg, True, 0, None) == -1.0
    assert BotRewardCalculator.terminal_addon(_cfg(), True, 0, 0) == 0.0


def test_warrior_damage_local() -> None:
    bot = BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0))
    r = BotRewardCalculator.compute(
        {"damage_dealt": 2.0},
        bot,
        GlobalRewardMode.NEUTRAL,
        _cfg(),
    )
    assert r > 0


def test_noop_negative() -> None:
    bot = BotState(bot_id=0, team=0, role=Role.WARRIOR, position=(0, 0))
    r = BotRewardCalculator.compute(
        {"noop": 1.0},
        bot,
        GlobalRewardMode.NEUTRAL,
        _cfg(),
    )
    assert r < 0


def test_gather_mode_global() -> None:
    bot = BotState(bot_id=0, team=0, role=Role.GATHERER, position=(0, 0))
    r = BotRewardCalculator.compute(
        {"global_scale": 10.0},
        bot,
        GlobalRewardMode.GATHER,
        _cfg(),
    )
    assert r > 0
