"""Flattened discrete action layout for the village manager (MaskablePPO)."""

from __future__ import annotations

from dataclasses import dataclass

from village_ai_war.state import BuildingType, GlobalRewardMode, Role


@dataclass(frozen=True)
class VillageActionSpace:
    """Indices and sizes for village ``Discrete`` actions."""

    map_size: int
    max_bots: int
    n_neighbor_slots: int = 8

    @property
    def n_noop(self) -> int:
        return 1

    @property
    def n_mode(self) -> int:
        return len(GlobalRewardMode)

    @property
    def n_rally(self) -> int:
        return self.map_size * self.map_size + 1

    @property
    def n_recruit(self) -> int:
        return len(Role)

    @property
    def placable_building_types(self) -> list[BuildingType]:
        return [
            BuildingType.BARRACKS,
            BuildingType.STORAGE,
            BuildingType.FARM,
            BuildingType.TOWER,
            BuildingType.WALL,
            BuildingType.CITADEL,
        ]

    @property
    def n_blueprint(self) -> int:
        return len(self.placable_building_types) * self.n_neighbor_slots

    @property
    def n_role_change(self) -> int:
        return self.max_bots * len(Role)

    @property
    def offset_noop(self) -> int:
        return 0

    @property
    def offset_mode(self) -> int:
        return self.offset_noop + self.n_noop

    @property
    def offset_rally(self) -> int:
        return self.offset_mode + self.n_mode

    @property
    def offset_recruit(self) -> int:
        return self.offset_rally + self.n_rally

    @property
    def offset_blueprint(self) -> int:
        return self.offset_recruit + self.n_recruit

    @property
    def offset_role_change(self) -> int:
        return self.offset_blueprint + self.n_blueprint

    @property
    def n_actions(self) -> int:
        return self.offset_role_change + self.n_role_change

    def neighbor_delta(self, slot: int) -> tuple[int, int]:
        """Eight-neighborhood ordering: N, NE, E, SE, S, SW, W, NW."""
        dirs = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]
        return dirs[slot % len(dirs)]


def decode_village_action(
    space: VillageActionSpace,
    action: int,
) -> dict[str, int | tuple[int, int] | None]:
    """Decode flat action index into structured fields."""
    a = int(action)
    if a == space.offset_noop:
        return {"kind": "noop"}
    if space.offset_mode <= a < space.offset_rally:
        mode = a - space.offset_mode
        return {"kind": "set_mode", "mode": int(mode)}
    if space.offset_rally <= a < space.offset_recruit:
        r = a - space.offset_rally
        if r == space.map_size * space.map_size:
            return {"kind": "clear_rally"}
        x = r % space.map_size
        y = r // space.map_size
        return {"kind": "set_rally", "position": (x, y)}
    if space.offset_recruit <= a < space.offset_blueprint:
        role = a - space.offset_recruit
        return {"kind": "recruit", "role": int(role)}
    if space.offset_blueprint <= a < space.offset_role_change:
        local = a - space.offset_blueprint
        nslots = space.n_neighbor_slots
        b_idx = local // nslots
        s_idx = local % nslots
        btype = space.placable_building_types[b_idx]
        return {"kind": "blueprint", "building_type": int(btype), "neighbor_slot": int(s_idx)}
    rc = a - space.offset_role_change
    role = rc % len(Role)
    bot_slot = rc // len(Role)
    return {"kind": "change_role", "bot_slot": int(bot_slot), "role": int(role)}
