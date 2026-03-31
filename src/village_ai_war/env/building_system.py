"""Blueprints, construction progress, and passive building effects."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.exceptions import InsufficientResourcesError, InvalidActionError
from village_ai_war.state import BuildingState, BuildingType, GameState, Role, TerrainType


class BuildingSystem:
    """Construction and pop-cap modifiers; mutates ``GameState``."""

    @staticmethod
    def construction_tick(state: GameState, config: Mapping[str, Any]) -> dict[str, Any]:
        """Advance blueprint progress when an ally builder is adjacent; complete buildings.

        Returns:
            Events with ``buildings_completed`` and ``block_placed_by_bot`` (progress share).
        """
        bcfg = config["buildings"]
        events: dict[str, Any] = {"buildings_completed": [], "block_placed_by_bot": {}}
        block_by_bot: dict[int, float] = events["block_placed_by_bot"]

        still: list[dict[str, Any]] = []
        for bp in state.blueprints:
            team = int(bp["team"])
            btype = BuildingType(int(bp["building_type"]))
            pos = tuple(bp["position"])
            px, py = int(pos[0]), int(pos[1])
            key = btype.name.lower()
            bdef = bcfg.get(key)
            if not isinstance(bdef, Mapping):
                bdef = {}
            ticks = int(bdef.get("construction_ticks", 20))
            step = 1.0 / max(ticks, 1)

            adjacent_builders: list[int] = []
            for v in state.villages:
                if v.team != team:
                    continue
                for bot in v.bots:
                    if not bot.is_alive or bot.role != Role.BUILDER:
                        continue
                    bx, by = bot.position
                    if abs(bx - px) + abs(by - py) == 1:
                        adjacent_builders.append(bot.bot_id)

            prog = float(bp.get("progress", 0.0))
            if adjacent_builders:
                room = 1.0 - prog
                delta = min(step, room)
                prog = prog + delta
                share = delta / len(adjacent_builders)
                for bid in adjacent_builders:
                    block_by_bot[bid] = block_by_bot.get(bid, 0.0) + share

            bp["progress"] = prog
            if prog >= 1.0:
                hp = BuildingSystem._max_hp(btype, bcfg)
                bid = state.next_building_id
                state.next_building_id += 1
                bstate = BuildingState(
                    building_id=bid,
                    team=team,
                    building_type=btype,
                    position=pos,
                    hp=hp,
                    max_hp=hp,
                    is_under_construction=False,
                    construction_progress=1.0,
                )
                state.villages[team].buildings.append(bstate)
                events["buildings_completed"].append((team, bid))
                BuildingSystem._apply_pop_cap(state, team, config)
            else:
                still.append(bp)
        state.blueprints = still
        return events

    @staticmethod
    def _max_hp(btype: BuildingType, bcfg: Mapping[str, Any]) -> int:
        key = btype.name.lower()
        return int(bcfg[key]["hp"])

    @staticmethod
    def _apply_pop_cap(state: GameState, team: int, config: Mapping[str, Any]) -> None:
        """Recompute pop cap from citadels."""
        bonus = int(config["buildings"].get("citadel_pop_bonus", 5))
        village = state.villages[team]
        base = 10
        extras = sum(
            bonus
            for b in village.buildings
            if b.building_type == BuildingType.CITADEL and not b.is_under_construction
        )
        village.pop_cap = base + extras

    @staticmethod
    def try_place_blueprint(
        state: GameState,
        team: int,
        building_type: BuildingType,
        position: tuple[int, int],
        config: Mapping[str, Any],
        adjacent_to_townhall: bool,
    ) -> None:
        """Spend resources and enqueue blueprint if valid.

        Raises:
            InsufficientResourcesError: If village cannot pay.
            InvalidActionError: If tile blocked or rules violated.
        """
        if building_type == BuildingType.TOWNHALL:
            raise InvalidActionError("Cannot blueprint town hall")
        n = state.map_size
        x, y = position
        if not (0 <= x < n and 0 <= y < n):
            raise InvalidActionError("Out of bounds")
        terrain = np.asarray(state.terrain, dtype=np.int32)
        if terrain[y, x] == int(TerrainType.MOUNTAIN):
            raise InvalidActionError("Mountain cell")

        village = state.villages[team]
        for b in village.buildings:
            if b.position == (x, y) and b.hp > 0:
                raise InvalidActionError("Cell occupied by building")
        for v in state.villages:
            for b in v.bots:
                if b.is_alive and b.position == (x, y):
                    raise InvalidActionError("Cell occupied by unit")
        for bp in state.blueprints:
            if tuple(bp["position"]) == (x, y):
                raise InvalidActionError("Cell has blueprint")

        if adjacent_to_townhall:
            th_positions = [
                b.position
                for b in village.buildings
                if b.building_type == BuildingType.TOWNHALL and b.hp > 0
            ]
            if not any(abs(x - tx) + abs(y - ty) == 1 for tx, ty in th_positions):
                raise InvalidActionError("Not adjacent to town hall")

        cost = BuildingSystem._cost_dict(building_type, config["buildings"])
        for k, v in cost.items():
            cur = getattr(village.resources, k)
            if cur < v:
                raise InsufficientResourcesError(f"Need {k} {v}, have {cur}")
        for k, v in cost.items():
            setattr(village.resources, k, getattr(village.resources, k) - v)

        state.blueprints.append(
            {
                "team": team,
                "building_type": int(building_type),
                "position": [x, y],
                "progress": 0.0,
            }
        )

    @staticmethod
    def _cost_dict(btype: BuildingType, bcfg: Mapping[str, Any]) -> dict[str, int]:
        key = btype.name.lower()
        raw = dict(bcfg[key]["cost"])
        out: dict[str, int] = {}
        for k, v in raw.items():
            out[str(k)] = int(v)
        return out
