"""Full world state: map, villages, blueprints, termination."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from village_ai_war.state.village_state import VillageState


class GameState(BaseModel):
    """Complete game state; terrain/resources as nested lists for Pydantic I/O."""

    tick: int = 0
    max_ticks: int = 2000
    map_size: int = 24
    terrain: list[list[int]] = Field(
        default_factory=list,
        description="(N, N) terrain type integers (TerrainType).",
    )
    resources: list[list[int]] = Field(
        default_factory=list,
        description="(N, N) resource layer integers (ResourceLayer).",
    )
    resource_amounts: list[list[int]] = Field(
        default_factory=list,
        description="(N, N) remaining harvest amount per cell.",
    )
    blueprints: list[dict[str, Any]] = Field(default_factory=list)
    villages: list[VillageState] = Field(default_factory=list)
    is_done: bool = False
    winner: Optional[int] = Field(
        default=None,
        description="0=red, 1=blue, None=ongoing or draw.",
    )
    next_bot_id: int = 0
    next_building_id: int = 0

    model_config = {"frozen": False}
