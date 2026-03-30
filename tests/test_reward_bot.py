"""Tests for bot reward shaping."""

from typing import Any

from village_ai_war.rewards.bot_reward import BotRewardCalculator
from village_ai_war.state import BotState, GlobalRewardMode, Role


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
