"""Village-level aggregates: resources, buildings, manager state."""

from enum import IntEnum

from pydantic import BaseModel, Field

from village_ai_war.state.bot_state import BotState


class GlobalRewardMode(IntEnum):
    """High-level strategic mode set by the village agent."""

    NEUTRAL = 0
    DEFEND = 1
    ATTACK = 2
    GATHER = 3


class BuildingType(IntEnum):
    """Constructible and core building types."""

    TOWNHALL = 0
    BARRACKS = 1
    STORAGE = 2
    FARM = 3
    TOWER = 4
    WALL = 5
    CITADEL = 6


class BuildingState(BaseModel):
    """A building instance on the map."""

    building_id: int
    team: int
    building_type: BuildingType
    position: tuple[int, int]
    hp: int
    max_hp: int
    is_under_construction: bool = False
    construction_progress: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Construction progress in [0, 1].",
    )

    model_config = {"frozen": False}


class ResourceStock(BaseModel):
    """Village resource inventory."""

    wood: int = 0
    stone: int = 0
    food: int = 0

    model_config = {"frozen": False}


class VillageState(BaseModel):
    """All state for one team's village."""

    team: int
    resources: ResourceStock = Field(default_factory=ResourceStock)
    pop_cap: int = 10
    global_reward_mode: GlobalRewardMode = GlobalRewardMode.NEUTRAL
    rally_point: tuple[int, int] | None = None
    bots: list[BotState] = Field(default_factory=list)
    buildings: list[BuildingState] = Field(default_factory=list)
    total_kills: int = 0
    total_losses: int = 0
    ticks_without_progress: int = 0
    spawn_queue_ticks_remaining: int = 0
    pending_recruit_role: int | None = None

    model_config = {"frozen": False}
