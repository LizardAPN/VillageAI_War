"""Bot entity state and role enumeration."""

from enum import IntEnum

from pydantic import BaseModel, Field


class Role(IntEnum):
    """Low-level bot specialization."""

    WARRIOR = 0
    GATHERER = 1
    FARMER = 2
    BUILDER = 3


class BotState(BaseModel):
    """Mutable bot unit state (serialized via Pydantic; sim mutates fields)."""

    bot_id: int
    team: int
    role: Role
    position: tuple[int, int]
    hp: int = 100
    max_hp: int = 100
    cooldown: int = 0
    is_alive: bool = True
    harvest_cooldown: int = Field(
        default=0,
        description="Ticks until next harvest tick can apply for gatherers.",
    )

    model_config = {"frozen": False}
