"""Build normalized high-level village observations (map tensor + vector)."""

from __future__ import annotations

import numpy as np

from village_ai_war.state import BuildingType, GameState, TerrainType


class VillageObsBuilder:
    """Six map channels and a 20-dim village vector, all in ``[0, 1]``.

    Map channels (``H × W × 6``):

    #. Terrain normalized.
    #. Resource layer normalized.
    #. Ally buildings occupancy (HP fraction at cell).
    #. Enemy buildings occupancy.
    #. Ally units (count / ``pop_cap`` clipped).
    #. Enemy units.

    Village vector (20):

    #. Wood / 500, stone / 500, food / 1000 (3)
    #. Pop used / ``pop_cap``, ``pop_cap`` / 32 (2)
    #. Mode one-hot (4)
    #. Rally set indicator + rally x,y normalized (3)
    #. Ally kills / 50, losses / 50 (2)
    #. Enemy alive / ``pop_cap``, ally alive / ``pop_cap`` (2)
    #. Pending spawn progress (1)
    #. Padding zeros (4) — reserved
    """

    VEC_DIM = 20
    N_CHANNELS = 6

    def __init__(self, map_size: int) -> None:
        self.map_size = map_size

    def build_map(self, state: GameState, team: int) -> np.ndarray:
        """Map tensor ``(N, N, C)`` in ``[0, 1]``; ally/enemy are relative to ``team``."""
        n = state.map_size
        terr = np.asarray(state.terrain, dtype=np.float32) / float(max(TerrainType))
        res = np.asarray(state.resources, dtype=np.float32) / 4.0

        ally_b = np.zeros((n, n), dtype=np.float32)
        en_b = np.zeros((n, n), dtype=np.float32)
        ally_u = np.zeros((n, n), dtype=np.float32)
        en_u = np.zeros((n, n), dtype=np.float32)

        village = state.villages[team]
        enemy = state.villages[1 - team]
        pop_cap = max(village.pop_cap, 1)

        for b in village.buildings:
            x, y = b.position
            if b.building_type == BuildingType.TOWNHALL or b.hp > 0:
                ally_b[y, x] = max(ally_b[y, x], b.hp / max(b.max_hp, 1))
        for b in enemy.buildings:
            x, y = b.position
            if b.hp > 0:
                en_b[y, x] = max(en_b[y, x], b.hp / max(b.max_hp, 1))

        for b in village.bots:
            if b.is_alive:
                x, y = b.position
                ally_u[y, x] += 1.0
        for b in enemy.bots:
            if b.is_alive:
                x, y = b.position
                en_u[y, x] += 1.0

        ally_u = np.clip(ally_u / pop_cap, 0.0, 1.0)
        en_u = np.clip(en_u / pop_cap, 0.0, 1.0)

        mp = np.stack([terr, res, ally_b, en_b, ally_u, en_u], axis=-1)
        return mp.astype(np.float32)

    def build_village_vec(self, state: GameState, team: int) -> np.ndarray:
        """Village state vector (``VEC_DIM``,) in ``[0, 1]`` for ``team``."""
        n = state.map_size
        village = state.villages[team]
        enemy = state.villages[1 - team]
        pop_cap = max(village.pop_cap, 1)

        vec = np.zeros((self.VEC_DIM,), dtype=np.float32)
        vec[0] = float(np.clip(village.resources.wood / 500.0, 0.0, 1.0))
        vec[1] = float(np.clip(village.resources.stone / 500.0, 0.0, 1.0))
        vec[2] = float(np.clip(village.resources.food / 1000.0, 0.0, 1.0))
        alive = sum(1 for b in village.bots if b.is_alive)
        vec[3] = float(np.clip(alive / pop_cap, 0.0, 1.0))
        vec[4] = float(np.clip(pop_cap / 32.0, 0.0, 1.0))
        vec[5 + int(village.global_reward_mode)] = 1.0
        if village.rally_point is not None:
            vec[9] = 1.0
            vec[10] = float(village.rally_point[0] / max(n - 1, 1))
            vec[11] = float(village.rally_point[1] / max(n - 1, 1))
        vec[12] = float(np.clip(village.total_kills / 50.0, 0.0, 1.0))
        vec[13] = float(np.clip(village.total_losses / 50.0, 0.0, 1.0))
        e_alive = sum(1 for b in enemy.bots if b.is_alive)
        vec[14] = float(np.clip(e_alive / max(enemy.pop_cap, 1), 0.0, 1.0))
        vec[15] = float(np.clip(alive / pop_cap, 0.0, 1.0))
        if village.spawn_queue_ticks_remaining > 0:
            vec[16] = 1.0 - float(village.spawn_queue_ticks_remaining) / 10.0
            vec[16] = float(np.clip(vec[16], 0.0, 1.0))

        return vec

    def build(self, state: GameState, team: int) -> dict[str, np.ndarray]:
        """Return dict with ``"map"`` and ``"village"`` arrays."""
        return {
            "map": self.build_map(state, team),
            "village": self.build_village_vec(state, team),
        }
