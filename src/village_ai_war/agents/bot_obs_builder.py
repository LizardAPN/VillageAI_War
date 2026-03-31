"""Build normalized per-bot observation vectors (MAPPO local slots, ``GameEnv`` bot mode)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from village_ai_war.state import GameState, ResourceLayer, Role, TerrainType


class BotObsBuilder:
    """Observation layout for a single controllable bot.

    Vector length is ``OBS_DIM`` (all values in ``[0, 1]``):

    - ``0:49`` — 7×7 local terrain patch (centered on bot), values are
      ``TerrainType / max_terrain``.
    - ``49:98`` — 7×7 local resource layer patch, normalized by max enum value.
    - ``98:102`` — one-hot ``Role`` (4).
    - ``102`` — HP fraction.
    - ``103`` — cooldown fraction (``cooldown / 10`` clipped).
    - ``104:106`` — bot position ``(x, y) / (map_size - 1)``.
    - ``106:110`` — one-hot ``GlobalRewardMode`` for own village (4).
    - ``110:114`` — ally alive count / ``pop_cap``, enemy alive / ``pop_cap``,
      ally TH HP fraction, enemy TH HP fraction (4 scalars).
    - ``114:116`` — role-specific hints (dist / map_size, secondary norm).
    - ``116:OBS_DIM`` — reserved (zeros).

    Args:
        map_size: Side length of the square map.
        max_terrain: Normalizer for terrain enum (default ``max(TerrainType)``).
        config: Optional merged game config (for ``map.resource_capacity``).
    """

    PATCH = 7
    _CORE_END = 116
    _RESERVED_TAIL = 65
    OBS_DIM = _CORE_END + _RESERVED_TAIL

    def __init__(
        self,
        map_size: int,
        max_terrain: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.map_size = map_size
        self.max_terrain = float(max(TerrainType)) if max_terrain is None else float(max_terrain)
        self.config = config

    def build(
        self,
        state: GameState,
        bot_id: int,
        config: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Return observation vector for ``bot_id``."""
        cfg = config if config is not None else self.config
        bot = next(
            (b for v in state.villages for b in v.bots if b.bot_id == bot_id),
            None,
        )
        if bot is None:
            return np.zeros((self.OBS_DIM,), dtype=np.float32)

        n = state.map_size
        terrain = np.asarray(state.terrain, dtype=np.float32) / self.max_terrain
        resources = np.asarray(state.resources, dtype=np.float32) / 4.0
        amounts = np.asarray(state.resource_amounts, dtype=np.int32)
        res_layer = np.asarray(state.resources, dtype=np.int32)

        cx, cy = bot.position
        r = self.PATCH // 2
        terr_patch = np.zeros((self.PATCH, self.PATCH), dtype=np.float32)
        res_patch = np.zeros((self.PATCH, self.PATCH), dtype=np.float32)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < n and 0 <= py < n:
                    terr_patch[dy + r, dx + r] = terrain[py, px]
                    res_patch[dy + r, dx + r] = resources[py, px]

        out = np.zeros((self.OBS_DIM,), dtype=np.float32)
        out[0:49] = terr_patch.reshape(-1)
        out[49:98] = res_patch.reshape(-1)

        role_oh = np.zeros(4, dtype=np.float32)
        role_oh[int(bot.role)] = 1.0
        out[98:102] = role_oh

        out[102] = float(np.clip(bot.hp / max(bot.max_hp, 1), 0.0, 1.0))
        out[103] = float(np.clip(bot.cooldown / 10.0, 0.0, 1.0))
        out[104] = float(cx / max(n - 1, 1))
        out[105] = float(cy / max(n - 1, 1))

        village = state.villages[bot.team]
        mode_oh = np.zeros(4, dtype=np.float32)
        mode_oh[int(village.global_reward_mode)] = 1.0
        out[106:110] = mode_oh

        enemy_team = 1 - bot.team
        ally_alive = sum(1 for b in village.bots if b.is_alive)
        enemy_alive = sum(1 for b in state.villages[enemy_team].bots if b.is_alive)
        pop_cap = max(village.pop_cap, 1)

        def th_hp(team: int) -> float:
            ths = [
                b
                for b in state.villages[team].buildings
                if b.building_type.name == "TOWNHALL"
            ]
            if not ths:
                return 0.0
            b = ths[0]
            return float(np.clip(b.hp / max(b.max_hp, 1), 0.0, 1.0))

        out[110] = float(np.clip(ally_alive / pop_cap, 0.0, 1.0))
        out[111] = float(np.clip(enemy_alive / pop_cap, 0.0, 1.0))
        out[112] = th_hp(bot.team)
        out[113] = th_hp(enemy_team)

        cap_forest = 800
        cap_stone = 500
        cap_field = 999999
        if cfg is not None:
            rcap = cfg.get("map", {}).get("resource_capacity", {})
            if isinstance(rcap, Mapping):
                cap_forest = int(rcap.get("forest", cap_forest))
                cap_stone = int(rcap.get("stone", cap_stone))
                cap_field = int(rcap.get("field", cap_field))

        def nearest_dist(pos: tuple[int, int], cells: list[tuple[int, int]]) -> float:
            if not cells:
                return float(n + n)
            return float(min(abs(pos[0] - x) + abs(pos[1] - y) for x, y in cells))

        enemy_cells = [
            tuple(b.position)
            for b in state.villages[enemy_team].bots
            if b.is_alive
        ]
        res_cells: list[tuple[int, int]] = []
        field_cells: list[tuple[int, int]] = []
        for y in range(n):
            for x in range(n):
                if amounts[y, x] <= 0:
                    continue
                layer = int(res_layer[y, x])
                if layer == int(ResourceLayer.NONE):
                    continue
                res_cells.append((x, y))
                if layer == int(ResourceLayer.FIELD):
                    field_cells.append((x, y))

        bp_team: list[tuple[tuple[int, int], float]] = [
            (
                (int(bp["position"][0]), int(bp["position"][1])),
                float(bp.get("progress", 0.0)),
            )
            for bp in state.blueprints
            if int(bp["team"]) == bot.team
        ]

        ms = float(max(n, 1))
        if bot.role == Role.WARRIOR:
            d = nearest_dist((cx, cy), enemy_cells)
            out[114] = float(np.clip(d / ms, 0.0, 1.0))
            near = sum(
                1
                for ex, ey in enemy_cells
                if abs(ex - cx) + abs(ey - cy) <= 1
            )
            out[115] = float(np.clip(near / 5.0, 0.0, 1.0))
        elif bot.role == Role.GATHERER:
            d = nearest_dist((cx, cy), res_cells)
            out[114] = float(np.clip(d / ms, 0.0, 1.0))
            layer_here = int(res_layer[cy, cx]) if 0 <= cy < n and 0 <= cx < n else int(ResourceLayer.NONE)
            cap_here = cap_forest
            if layer_here == int(ResourceLayer.STONE):
                cap_here = cap_stone
            elif layer_here == int(ResourceLayer.FIELD):
                cap_here = cap_field
            amt_here = int(amounts[cy, cx]) if 0 <= cy < n and 0 <= cx < n else 0
            out[115] = float(np.clip(amt_here / max(cap_here, 1), 0.0, 1.0))
        elif bot.role == Role.FARMER:
            d = nearest_dist((cx, cy), field_cells)
            out[114] = float(np.clip(d / ms, 0.0, 1.0))
            out[115] = float(np.clip(village.resources.food / 1000.0, 0.0, 1.0))
        elif bot.role == Role.BUILDER:
            if not bp_team:
                out[114] = 1.0
                out[115] = 0.0
            else:
                best_pos, best_prog = min(
                    bp_team,
                    key=lambda t: abs(cx - t[0][0]) + abs(cy - t[0][1]),
                )
                best_d = abs(cx - best_pos[0]) + abs(cy - best_pos[1])
                out[114] = float(np.clip(best_d / ms, 0.0, 1.0))
                out[115] = float(np.clip(best_prog, 0.0, 1.0))

        return out
