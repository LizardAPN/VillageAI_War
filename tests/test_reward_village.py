"""Tests for village reward."""

from typing import Any

from village_ai_war.rewards.village_reward import VillageRewardCalculator
from village_ai_war.state import ResourceStock, VillageState


def _cfg() -> dict[str, Any]:
    return {
        "rewards": {
            "village": {
                "economy_coeff": 0.01,
                "kill_reward": 5.0,
                "loss_penalty": -3.0,
                "building_reward": 10.0,
                "stagnation_penalty": -0.05,
                "stagnation_threshold": 50,
                "win": 1000.0,
                "loss": -1000.0,
            }
        }
    }


def test_kill_bonus() -> None:
    v = VillageState(team=0, resources=ResourceStock())
    r = VillageRewardCalculator.compute(
        {"kills": {0: 1}},
        v,
        _cfg(),
        terminated=False,
        won=None,
    )
    assert r >= 5.0


def test_win_terminal() -> None:
    v = VillageState(team=0, resources=ResourceStock())
    r = VillageRewardCalculator.compute(
        {},
        v,
        _cfg(),
        terminated=True,
        won=True,
    )
    assert r >= 1000.0


def test_food_security_bonus() -> None:
    cfg = {
        "rewards": {
            "village": {
                **_cfg()["rewards"]["village"],
                "food_security_bonus": 0.02,
                "food_security_threshold": 100,
            }
        }
    }
    v = VillageState(team=0, resources=ResourceStock(food=150))
    r = VillageRewardCalculator.compute({}, v, cfg, terminated=False, won=None)
    assert r >= (150 - 100) * 0.02
