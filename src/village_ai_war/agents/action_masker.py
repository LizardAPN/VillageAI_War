"""Valid-action masks for village manager (MaskablePPO)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.agents.village_action_space import VillageActionSpace
from village_ai_war.env.building_system import BuildingSystem
from village_ai_war.state import BuildingType, GameState, Role, TerrainType


class ActionMasker:
    """Build boolean masks over the flattened village action space."""

    @staticmethod
    def compute_masks(
        state: GameState,
        team: int,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        """Return shape ``(n_actions,)`` mask (True = valid)."""
        max_bots = int(config["game"].get("max_bots_for_role_change", 32))
        space = VillageActionSpace(state.map_size, max_bots=max_bots)
        mask = np.zeros((space.n_actions,), dtype=bool)
        mask[space.offset_noop] = True

        village = state.villages[team]
        n = state.map_size
        terrain = np.asarray(state.terrain, dtype=np.int32)
        adjacent_only = bool(config["game"].get("blueprint_adjacent_to_townhall", True))

        # Modes always valid
        for i in range(space.n_mode):
            mask[space.offset_mode + i] = True

        # Rally: any cell + clear
        for i in range(space.n_rally):
            mask[space.offset_rally + i] = True

        # Recruit: need resources, pop cap, no active queue
        ecfg = config["economy"]
        wood = int(ecfg["bot_cost"]["wood"])
        food = int(ecfg["bot_cost"]["food"])
        alive = sum(1 for b in village.bots if b.is_alive)
        can_recruit = (
            alive < village.pop_cap
            and village.spawn_queue_ticks_remaining == 0
            and village.resources.wood >= wood
            and village.resources.food >= food
        )
        for i in range(space.n_recruit):
            mask[space.offset_recruit + i] = can_recruit

        # Blueprint: per (type, neighbor slot)
        th_positions = [
            b.position
            for b in village.buildings
            if b.building_type == BuildingType.TOWNHALL and b.hp > 0
        ]
        for bi, _bt in enumerate(space.placable_building_types):
            for si in range(space.n_neighbor_slots):
                idx = space.offset_blueprint + bi * space.n_neighbor_slots + si
                if not th_positions:
                    mask[idx] = False
                    continue
                dx, dy = space.neighbor_delta(si)
                ok = False
                for tx, ty in th_positions:
                    cx, cy = tx + dx, ty + dy
                    if not (0 <= cx < n and 0 <= cy < n):
                        continue
                    if terrain[cy, cx] == int(TerrainType.MOUNTAIN):
                        continue
                    if ActionMasker._cell_blocked(state, team, cx, cy):
                        continue
                    try:
                        cost = BuildingSystem._cost_dict(
                            space.placable_building_types[bi],
                            config["buildings"],
                        )
                    except Exception:
                        ok = False
                        break
                    affordable = all(
                        getattr(village.resources, k) >= v for k, v in cost.items()
                    )
                    if not affordable:
                        continue
                    if adjacent_only:
                        ok = True
                        break
                    ok = True
                mask[idx] = ok

        # Role change: bot_slot must reference alive bot
        bots_alive = [b for b in village.bots if b.is_alive][: max_bots]
        for slot in range(max_bots):
            for ri in range(len(Role)):
                idx = space.offset_role_change + slot * len(Role) + ri
                if slot >= len(bots_alive):
                    mask[idx] = False
                else:
                    mask[idx] = True

        return mask


    @staticmethod
    def _cell_blocked(state: GameState, team: int, x: int, y: int) -> bool:
        for v in state.villages:
            for b in v.buildings:
                if b.position == (x, y) and b.hp > 0:
                    return True
            for u in v.bots:
                if u.is_alive and u.position == (x, y):
                    return True
        for bp in state.blueprints:
            if tuple(bp["position"]) == (x, y):
                return True
        return False
