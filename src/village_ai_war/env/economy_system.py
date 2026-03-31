"""Harvesting, food consumption, hunger, and unit production."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from village_ai_war.state import BotState, GameState, ResourceLayer, Role, TerrainType


class EconomySystem:
    """Vector-friendly economy updates; mutates ``GameState`` in place."""

    @staticmethod
    def step(state: GameState, config: Mapping[str, Any]) -> dict[str, Any]:
        """Run economy phase for one tick.

        Args:
            state: Full game state (mutated).
            config: Merged config with ``economy`` and ``map`` sections.

        Returns:
            Event dict for reward calculators:
            ``resource_collected``, ``food_produced``, ``bots_spawned``, etc.
        """
        ecfg = config["economy"]
        n = state.map_size
        terrain = np.asarray(state.terrain, dtype=np.int32)
        res_layer = np.asarray(state.resources, dtype=np.int32)
        amounts = np.asarray(state.resource_amounts, dtype=np.int32)

        harvest_interval = int(ecfg["harvest_interval"])
        harvest_amount = int(ecfg["harvest_amount"])
        food_per_bot = float(ecfg["food_consumption"])
        hunger_damage = int(ecfg["hunger_damage"])
        spawn_delay = int(ecfg["bot_spawn_delay"])
        bot_cost_wood = int(ecfg["bot_cost"]["wood"])
        bot_cost_food = int(ecfg["bot_cost"]["food"])
        farm_bonus = float(ecfg.get("farm_food_bonus", 0.5))

        resource_by_bot: dict[int, int] = {}

        def add_resource_bot(bot_id: int, amt: int) -> None:
            if amt <= 0:
                return
            resource_by_bot[bot_id] = resource_by_bot.get(bot_id, 0) + amt

        events: dict[str, Any] = {
            "resource_collected": {0: 0, 1: 0},
            "resource_collected_by_bot": resource_by_bot,
            "wood_delta": {0: 0, 1: 0},
            "stone_delta": {0: 0, 1: 0},
            "food_delta": {0: 0, 1: 0},
            "food_produced": {0: 0, 1: 0},
            "bots_spawned": [],
            "hunger_damage": {0: 0, 1: 0},
        }

        for village in state.villages:
            team = village.team
            farm_count = sum(
                1
                for b in village.buildings
                if b.building_type.name == "FARM" and not b.is_under_construction
            )
            food_mult = 1.0 + farm_bonus * farm_count

            for bot in village.bots:
                if not bot.is_alive:
                    continue
                if bot.role != Role.GATHERER:
                    continue
                x, y = bot.position
                if not (0 <= x < n and 0 <= y < n):
                    continue
                if terrain[y, x] == int(TerrainType.MOUNTAIN):
                    continue
                layer = int(res_layer[y, x])
                if layer == int(ResourceLayer.NONE) or amounts[y, x] <= 0:
                    continue
                bot.harvest_cooldown += 1
                if bot.harvest_cooldown < harvest_interval:
                    continue
                bot.harvest_cooldown = 0
                take = min(harvest_amount, int(amounts[y, x]))
                if take <= 0:
                    continue
                amounts[y, x] -= take
                if layer == int(ResourceLayer.FOREST):
                    village.resources.wood += take
                    events["resource_collected"][team] += take
                    events["wood_delta"][team] += take
                    add_resource_bot(bot.bot_id, take)
                elif layer == int(ResourceLayer.STONE):
                    village.resources.stone += take
                    events["resource_collected"][team] += take
                    events["stone_delta"][team] += take
                    add_resource_bot(bot.bot_id, take)
                elif layer == int(ResourceLayer.FIELD):
                    prod = int(take * food_mult)
                    village.resources.food += prod
                    events["food_produced"][team] += prod
                    events["food_delta"][team] += prod
                    add_resource_bot(bot.bot_id, prod)

            alive = [b for b in village.bots if b.is_alive]
            need = max(0, int(math.ceil(food_per_bot * len(alive))))
            if village.resources.food >= need:
                village.resources.food -= need
                events["food_delta"][team] -= need
            else:
                short = need - village.resources.food
                village.resources.food = 0
                events["food_delta"][team] -= need
                for _b in alive:
                    _b.hp -= hunger_damage
                    events["hunger_damage"][team] += hunger_damage
                    if _b.hp <= 0:
                        _b.hp = 0
                        _b.is_alive = False

            if village.spawn_queue_ticks_remaining > 0:
                village.spawn_queue_ticks_remaining -= 1
                if village.spawn_queue_ticks_remaining == 0 and village.pending_recruit_role is not None:
                    role_int = int(village.pending_recruit_role)
                    village.pending_recruit_role = None
                    alive_n = sum(1 for b in village.bots if b.is_alive)
                    if alive_n < village.pop_cap:
                        th = next(
                            (b for b in village.buildings if b.building_type.name == "TOWNHALL"),
                            None,
                        )
                        if th is not None:
                            bx, by = th.position
                            role = Role(role_int)
                            stats = config["combat"]["stats"][role.name.lower()]
                            hp = int(stats["hp"])
                            bid = state.next_bot_id
                            state.next_bot_id += 1
                            village.bots.append(
                                BotState(
                                    bot_id=bid,
                                    team=team,
                                    role=role,
                                    position=(bx, by),
                                    hp=hp,
                                    max_hp=hp,
                                )
                            )
                            events["bots_spawned"].append((team, bid))

        state.resource_amounts = amounts.tolist()
        return events

    @staticmethod
    def queue_recruit(
        state: GameState,
        team: int,
        role: Role,
        config: Mapping[str, Any],
    ) -> bool:
        """Deduct recruit cost and arm spawn delay if affordable and capacity allows.

        Returns:
            True if recruitment was queued.
        """
        ecfg = config["economy"]
        village = state.villages[team]
        alive = sum(1 for b in village.bots if b.is_alive)
        if alive >= village.pop_cap:
            return False
        if village.spawn_queue_ticks_remaining > 0:
            return False
        wood = int(ecfg["bot_cost"]["wood"])
        food = int(ecfg["bot_cost"]["food"])
        if village.resources.wood < wood or village.resources.food < food:
            return False
        village.resources.wood -= wood
        village.resources.food -= food
        village.pending_recruit_role = int(role)
        village.spawn_queue_ticks_remaining = int(ecfg["bot_spawn_delay"])
        return True
