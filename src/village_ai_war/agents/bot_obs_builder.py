"""Build normalized low-level bot observations (fixed length 181)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.state import GameState, GlobalRewardMode, Role, TerrainType


class BotObsBuilder:
    """Observation layout for a single controllable bot.

    Layout (``181`` floats in ``[0, 1]``):

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
    - ``114:181`` — padded zeros (reserved for future local patches / path hints).

    Args:
        map_size: Side length of the square map.
        max_terrain: Normalizer for terrain enum (default ``max(TerrainType)``).
    """

    OBS_DIM = 181
    PATCH = 7

    def __init__(self, map_size: int, max_terrain: float | None = None) -> None:
        self.map_size = map_size
        self.max_terrain = float(max(TerrainType)) if max_terrain is None else float(max_terrain)

    def build(self, state: GameState, bot_id: int) -> np.ndarray:
        """Return observation vector for ``bot_id``."""
        bot = next(
            (b for v in state.villages for b in v.bots if b.bot_id == bot_id),
            None,
        )
        if bot is None:
            return np.zeros((self.OBS_DIM,), dtype=np.float32)

        n = state.map_size
        terrain = np.asarray(state.terrain, dtype=np.float32) / self.max_terrain
        resources = np.asarray(state.resources, dtype=np.float32) / 4.0

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

        return out
