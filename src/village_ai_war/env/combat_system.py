"""Combat resolution: directed melee, towers, and attacks on buildings."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.state import BotState, BuildingState, BuildingType, GameState, TerrainType


class CombatSystem:
    """Applies combat; mutates ``GameState``."""

    @staticmethod
    def apply_melee_intents(
        state: GameState,
        config: Mapping[str, Any],
        intents: list[tuple[int, int, tuple[int, int]]],
    ) -> dict[str, Any]:
        """Apply one melee strike per intent: ``(attacker_team, bot_id, (dx,dy))``.

        Args:
            state: Game state.
            config: Merged config with ``combat``.
            intents: Attack directions for bots that chose attack this tick.

        Returns:
            Event fragment merged into global combat events.
        """
        ccfg = config["combat"]
        stats = ccfg["stats"]
        n = state.map_size
        terrain = np.asarray(state.terrain, dtype=np.int32)

        events: dict[str, Any] = {
            "damage_dealt": {0: 0, 1: 0},
            "damage_taken": {0: 0, 1: 0},
            "kills": {0: 0, 1: 0},
            "building_damage": [],
        }

        pos_to_bots: dict[tuple[int, int], list[tuple[int, BotState]]] = {}
        for v in state.villages:
            for b in v.bots:
                if b.is_alive:
                    pos_to_bots.setdefault(b.position, []).append((v.team, b))

        for atk_team, bot_id, (dx, dy) in intents:
            bot = next(
                (b for v in state.villages for b in v.bots if b.bot_id == bot_id),
                None,
            )
            if bot is None or not bot.is_alive:
                continue
            role_key = bot.role.name.lower()
            dmg = int(stats[role_key]["damage"])
            bx, by = bot.position
            tx, ty = bx + dx, by + dy
            if not (0 <= tx < n and 0 <= ty < n):
                continue
            if terrain[ty, tx] == int(TerrainType.MOUNTAIN):
                continue
            target = CombatSystem._find_enemy_unit_at(pos_to_bots, atk_team, (tx, ty))
            if target is not None:
                tteam, tb = target
                tb.hp -= dmg
                events["damage_dealt"][atk_team] += dmg
                events["damage_taken"][tteam] += dmg
                if tb.hp <= 0:
                    tb.hp = 0
                    tb.is_alive = False
                    events["kills"][atk_team] += 1
                    state.villages[atk_team].total_kills += 1
                    state.villages[tteam].total_losses += 1
                CombatSystem.register_melee_attack(bot, config)
                continue
            bld = CombatSystem._enemy_building_at(state, atk_team, (tx, ty))
            if bld is not None:
                bld.hp -= dmg
                events["building_damage"].append((bld.building_id, dmg))
                if bld.hp <= 0:
                    bld.hp = 0
                CombatSystem.register_melee_attack(bot, config)

        return events

    @staticmethod
    def apply_tower_fire(state: GameState, config: Mapping[str, Any]) -> dict[str, Any]:
        """Tower auto-attack nearest enemy unit in range."""
        ccfg = config["combat"]
        tower_damage = int(ccfg["tower_damage"])
        tower_range = int(ccfg["tower_range"])

        events: dict[str, Any] = {
            "damage_dealt": {0: 0, 1: 0},
            "damage_taken": {0: 0, 1: 0},
            "kills": {0: 0, 1: 0},
            "building_damage": [],
        }

        for village in state.villages:
            for bld in village.buildings:
                if bld.is_under_construction:
                    continue
                if bld.building_type != BuildingType.TOWER:
                    continue
                if bld.hp <= 0:
                    continue
                tx, ty = bld.position
                enemy_team = 1 - village.team
                enemy_v = state.villages[enemy_team]
                best: BotState | None = None
                best_d = tower_range + 1
                for eb in enemy_v.bots:
                    if not eb.is_alive:
                        continue
                    ex, ey = eb.position
                    d = abs(ex - tx) + abs(ey - ty)
                    if d <= tower_range and d < best_d:
                        best_d = d
                        best = eb
                if best is not None:
                    best.hp -= tower_damage
                    events["damage_dealt"][village.team] += tower_damage
                    events["damage_taken"][enemy_team] += tower_damage
                    if best.hp <= 0:
                        best.hp = 0
                        best.is_alive = False
                        village.total_kills += 1
                        enemy_v.total_losses += 1
                        events["kills"][village.team] += 1

        return events

    @staticmethod
    def tick_cooldowns(state: GameState) -> None:
        """Decrement per-bot attack cooldowns."""
        for village in state.villages:
            for bot in village.bots:
                if bot.cooldown > 0:
                    bot.cooldown -= 1

    @staticmethod
    def _find_enemy_unit_at(
        pos_to_bots: Mapping[tuple[int, int], list[tuple[int, BotState]]],
        my_team: int,
        pos: tuple[int, int],
    ) -> tuple[int, BotState] | None:
        for team, b in pos_to_bots.get(pos, []):
            if team != my_team and b.is_alive:
                return team, b
        return None

    @staticmethod
    def _enemy_building_at(
        state: GameState,
        my_team: int,
        pos: tuple[int, int],
    ) -> BuildingState | None:
        px, py = pos
        for v in state.villages:
            if v.team == my_team:
                continue
            for b in v.buildings:
                if b.position == (px, py) and b.hp > 0:
                    return b
        return None

    @staticmethod
    def register_melee_attack(bot: BotState, config: Mapping[str, Any]) -> None:
        """Set brief cooldown after melee (extensible via config)."""
        _ = config
        bot.cooldown = 1
