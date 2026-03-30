"""Simulation subsystems and Gymnasium environment."""

from typing import Any

__all__ = ["GameEnv"]


def __getattr__(name: str) -> Any:
    if name == "GameEnv":
        from village_ai_war.env.game_env import GameEnv as _GameEnv

        return _GameEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
