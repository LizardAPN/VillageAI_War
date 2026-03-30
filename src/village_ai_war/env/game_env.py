"""Gymnasium environment for Village AI War."""

from __future__ import annotations

from typing import Any, Mapping, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from village_ai_war.agents.action_masker import ActionMasker
from village_ai_war.agents.bot_obs_builder import BotObsBuilder
from village_ai_war.agents.village_action_space import VillageActionSpace, decode_village_action
from village_ai_war.agents.village_obs_builder import VillageObsBuilder
from village_ai_war.env.building_system import BuildingSystem
from village_ai_war.env.combat_system import CombatSystem
from village_ai_war.env.economy_system import EconomySystem
from village_ai_war.env.map_generator import generate_initial_state
from village_ai_war.exceptions import InsufficientResourcesError, InvalidActionError
from village_ai_war.rewards.bot_reward import BotRewardCalculator
from village_ai_war.rewards.village_reward import VillageRewardCalculator
from village_ai_war.state import (
    BotState,
    BuildingType,
    GameState,
    GlobalRewardMode,
    ResourceLayer,
    Role,
    TerrainType,
)


class GameEnv(gym.Env):
    """Village AI War — two-level hierarchical RL environment.

    Args:
        config: Hydra/OmegaConf merged config (dict-like).
        mode: ``"bot"`` | ``"village"`` | ``"full"``.
        team: Learning team index (``0`` red, ``1`` blue).
        render_mode: ``"human"`` | ``"rgb_array"`` | ``None``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    BOT_ACTIONS = 12
    _DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def __init__(
        self,
        config: Mapping[str, Any],
        mode: str = "village",
        team: int = 0,
        render_mode: str | None = None,
        bot_role: Role | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.team = team
        self.render_mode = render_mode
        self.bot_role = bot_role
        self._state: GameState | None = None
        self._rng: np.random.Generator | None = None
        self._controlled_bot_id: int = 0
        self._renderer: Any = None

        n = int(config["map"]["size"])
        max_bots = int(config["game"].get("max_bots_for_role_change", 32))
        self._village_space = VillageActionSpace(n, max_bots=max_bots)
        self._bot_obs = BotObsBuilder(n)
        self._vil_obs = VillageObsBuilder(n)

        if mode == "bot":
            self.action_space = spaces.Discrete(self.BOT_ACTIONS)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(BotObsBuilder.OBS_DIM,),
                dtype=np.float32,
            )
        elif mode in ("village", "full"):
            self.action_space = spaces.Discrete(self._village_space.n_actions)
            self.observation_space = spaces.Dict(
                {
                    "map": spaces.Box(0.0, 1.0, (n, n, VillageObsBuilder.N_CHANNELS), np.float32),
                    "village": spaces.Box(
                        0.0, 1.0, (VillageObsBuilder.VEC_DIM,), np.float32
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        map_seed = int(self.config["map"].get("seed", 0))
        if seed is not None:
            map_seed = seed
        cfg = dict(self.config)
        cfg["map"] = dict(cfg["map"])
        cfg["map"]["seed"] = map_seed
        self._state = generate_initial_state(cfg, self._rng)

        # Pick controlled bot for bot mode (optional role filter for stage-1 training)
        if self.mode == "bot":
            vb = self._state.villages[self.team].bots
            role_filter: int | None = int(self.bot_role) if self.bot_role is not None else None
            if options and "role" in options:
                role_filter = int(options["role"])
            candidates = [
                b
                for b in vb
                if b.is_alive and (role_filter is None or int(b.role) == role_filter)
            ]
            pick = candidates[0] if candidates else (vb[0] if vb else None)
            self._controlled_bot_id = pick.bot_id if pick is not None else 0

        obs = self._build_obs()
        info = self._info_dict()
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._state is not None and self._rng is not None
        state = self._state
        terminated = False
        truncated = False
        reward = 0.0

        manager_action: dict[str, Any] | None = None
        interval = int(self.config["game"]["manager_interval"])
        if self.mode in ("village", "full") and state.tick % interval == 0:
            dec = decode_village_action(self._village_space, int(action))
            manager_action = dec
            self._apply_village_decision(self.team, dec)
        if self.mode == "full" and state.tick % interval == 0:
            self._apply_village_decision(1 - self.team, {"kind": "noop"})

        melee_intents: list[tuple[int, int, tuple[int, int]]] = []

        if self.mode == "bot":
            self._apply_bot_action(self.team, self._controlled_bot_id, int(action), melee_intents)
            self._heuristic_team(self.team, exclude_id=self._controlled_bot_id, melee_intents=melee_intents)
            self._heuristic_team(1 - self.team, exclude_id=None, melee_intents=melee_intents)
        else:
            self._heuristic_team(0, exclude_id=None, melee_intents=melee_intents)
            self._heuristic_team(1, exclude_id=None, melee_intents=melee_intents)

        cmb = CombatSystem.apply_melee_intents(state, self.config, melee_intents)
        eco = EconomySystem.step(state, self.config)
        bld = BuildingSystem.construction_tick(state, self.config)
        tw = CombatSystem.apply_tower_fire(state, self.config)
        CombatSystem.tick_cooldowns(state)

        merged = GameEnv._merge_combat_events(cmb, tw)
        merged["losses"] = {0: cmb["kills"].get(1, 0), 1: cmb["kills"].get(0, 0)}
        merged["resource_collected"] = eco.get("resource_collected", {})
        merged["food_produced"] = eco.get("food_produced", {})
        merged["building_completed"] = bld.get("buildings_completed", [])

        kills_this_tick = int(cmb["kills"].get(self.team, 0)) + int(tw["kills"].get(self.team, 0))
        resources_delta = {
            "wood": eco["wood_delta"].get(self.team, 0),
            "stone": eco["stone_delta"].get(self.team, 0),
            "food": eco["food_delta"].get(self.team, 0),
        }
        bots_alive = sum(1 for b in state.villages[self.team].bots if b.is_alive)

        kills_any = sum(int(merged["kills"].get(t, 0)) for t in (0, 1))
        res_any = any(
            abs(eco["wood_delta"].get(t, 0))
            + abs(eco["stone_delta"].get(t, 0))
            + abs(eco["food_delta"].get(t, 0))
            for t in (0, 1)
        )
        progress = bool(kills_any or res_any or merged.get("building_completed"))
        for vil in state.villages:
            if progress:
                vil.ticks_without_progress = 0
            else:
                vil.ticks_without_progress += 1

        won = GameEnv._terminal_update(state, self.config)
        if won is not None:
            terminated = True
        if state.tick >= state.max_ticks:
            truncated = True
            state.is_done = True

        state.tick += 1

        if self.mode == "bot":
            bot = next(
                (
                    b
                    for v in state.villages
                    for b in v.bots
                    if b.bot_id == self._controlled_bot_id
                ),
                None,
            )
            if bot is not None:
                mode = state.villages[self.team].global_reward_mode
                bev = self._bot_events_for(bot, merged, int(action))
                reward = BotRewardCalculator.compute(bev, bot, mode, self.config)
        else:
            reward = VillageRewardCalculator.compute(
                merged,
                state.villages[self.team],
                self.config,
                terminated or truncated,
                True
                if won == self.team
                else (False if won is not None and won != self.team else None),
            )

        obs = self._build_obs()
        info = {
            **self._info_dict(),
            "kills_this_tick": kills_this_tick,
            "resources_delta": resources_delta,
            "manager_action": manager_action,
            "bots_alive": bots_alive,
        }
        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid village actions (MaskablePPO)."""
        assert self._state is not None
        if self.mode not in ("village", "full"):
            return np.ones((self._village_space.n_actions,), dtype=bool)
        m = ActionMasker.compute_masks(self._state, self.team, self.config)
        interval = int(self.config["game"]["manager_interval"])
        if self._state.tick % interval != 0:
            m = np.zeros_like(m)
            m[self._village_space.offset_noop] = True
        return m

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        from village_ai_war.rendering.pygame_renderer import PygameRenderer

        if self._renderer is None:
            self._renderer = PygameRenderer(self.config, self._state)
        return self._renderer.render(self._state, mode=cast(str, self.render_mode))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _build_obs(self) -> Any:
        assert self._state is not None
        if self.mode == "bot":
            return self._bot_obs.build(self._state, self._controlled_bot_id)
        return self._vil_obs.build(self._state, self.team)

    def _info_dict(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "tick": self._state.tick,
            "team": self.team,
            "mode": self.mode,
        }

    def _apply_village_decision(self, team: int, dec: Mapping[str, Any]) -> None:
        assert self._state is not None
        village = self._state.villages[team]
        kind = dec.get("kind")
        if kind == "noop" or kind is None:
            return
        if kind == "set_mode":
            village.global_reward_mode = GlobalRewardMode(int(dec["mode"]))
            village.ticks_without_progress = 0
        elif kind == "set_rally":
            village.rally_point = cast(tuple[int, int], dec["position"])
            village.ticks_without_progress = 0
        elif kind == "clear_rally":
            village.rally_point = None
        elif kind == "recruit":
            EconomySystem.queue_recruit(self._state, team, Role(int(dec["role"])), self.config)
            village.ticks_without_progress = 0
        elif kind == "blueprint":
            bt = BuildingType(int(dec["building_type"]))
            slot = int(dec["neighbor_slot"])
            dx, dy = self._village_space.neighbor_delta(slot)
            th = next(
                (b for b in village.buildings if b.building_type == BuildingType.TOWNHALL),
                None,
            )
            if th is None:
                return
            x, y = th.position[0] + dx, th.position[1] + dy
            try:
                BuildingSystem.try_place_blueprint(
                    self._state,
                    team,
                    bt,
                    (x, y),
                    self.config,
                    bool(self.config["game"].get("blueprint_adjacent_to_townhall", True)),
                )
                village.ticks_without_progress = 0
            except (InsufficientResourcesError, InvalidActionError):
                pass
        elif kind == "change_role":
            slot = int(dec["bot_slot"])
            new_role = Role(int(dec["role"]))
            alive = [b for b in village.bots if b.is_alive]
            if slot >= len(alive):
                return
            alive[slot].role = new_role
            village.ticks_without_progress = 0

    def _apply_bot_action(
        self,
        team: int,
        bot_id: int,
        action: int,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
    ) -> None:
        assert self._state is not None
        bot = next(
            (b for b in self._state.villages[team].bots if b.bot_id == bot_id),
            None,
        )
        if bot is None or not bot.is_alive:
            return
        n = self._state.map_size
        terrain = np.asarray(self._state.terrain, dtype=np.int32)
        ax, ay = bot.position

        if action == 0:
            return
        if 1 <= action <= 4:
            dx, dy = self._DIRS[action - 1]
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < n and 0 <= ny < n and terrain[ny, nx] != int(TerrainType.MOUNTAIN):
                if not GameEnv._unit_at(self._state, nx, ny, exclude=bot):
                    bot.position = (nx, ny)
        elif 5 <= action <= 8:
            dx, dy = self._DIRS[action - 5]
            melee_intents.append((team, bot_id, (dx, dy)))
        elif action == 9 and bot.role == Role.GATHERER:
            x, y = bot.position
            layer = int(np.asarray(self._state.resources, dtype=np.int32)[y, x])
            if layer != int(ResourceLayer.NONE):
                bot.harvest_cooldown = max(bot.harvest_cooldown, 0)
        elif action == 10 and bot.role == Role.FARMER:
            village = self._state.villages[team]
            village.resources.food += 1
        elif action == 11 and bot.role == Role.BUILDER:
            for v in self._state.villages:
                for b in v.buildings:
                    if b.team != team or b.hp >= b.max_hp:
                        continue
                    bx, by = b.position
                    if abs(bx - ax) + abs(by - ay) == 1:
                        b.hp = min(b.max_hp, b.hp + max(1, int(0.1 * b.max_hp)))

    def _heuristic_team(
        self,
        team: int,
        exclude_id: int | None,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
    ) -> None:
        assert self._state is not None
        enemy_team = 1 - team
        for bot in self._state.villages[team].bots:
            if not bot.is_alive or (exclude_id is not None and bot.bot_id == exclude_id):
                continue
            if bot.role == Role.WARRIOR:
                target = GameEnv._nearest_enemy_bot(self._state, team, bot.position)
                if target is not None:
                    tx, ty = target.position
                    bx, by = bot.position
                    dx = 0 if tx == bx else (1 if tx > bx else -1)
                    dy = 0 if ty == by else (1 if ty > by else -1)
                    prefer_dx = abs(tx - bx) >= abs(ty - by)
                    move = (dx, 0) if prefer_dx and dx != 0 else (0, dy) if dy != 0 else (0, 0)
                    if move != (0, 0):
                        self._apply_bot_action(
                            team,
                            bot.bot_id,
                            self._dirs_to_move_action(move),
                            melee_intents,
                        )
                    elif self._rng and self._rng.random() < 0.3:
                        self._apply_bot_action(
                            team,
                            bot.bot_id,
                            5 + self._attack_dir_toward((bx, by), (tx, ty)),
                            melee_intents,
                        )
            elif bot.role == Role.GATHERER:
                if self._rng and self._rng.random() < 0.5:
                    self._apply_bot_action(team, bot.bot_id, 1 + int(self._rng.integers(0, 4)), melee_intents)

    def _dirs_to_move_action(self, move: tuple[int, int]) -> int:
        if move == (0, 0):
            return 0
        return 1 + self._DIRS.index(move)

    def _attack_dir_toward(
        self,
        pos: tuple[int, int],
        tgt: tuple[int, int],
    ) -> int:
        bx, by = pos
        tx, ty = tgt
        if tx > bx:
            return 1
        if tx < bx:
            return 3
        if ty > by:
            return 2
        return 0

    @staticmethod
    def _nearest_enemy_bot(
        state: GameState,
        team: int,
        pos: tuple[int, int],
    ) -> BotState | None:
        best: BotState | None = None
        best_d = 10**9
        ex, ey = pos
        for b in state.villages[1 - team].bots:
            if not b.is_alive:
                continue
            d = abs(b.position[0] - ex) + abs(b.position[1] - ey)
            if d < best_d:
                best_d = d
                best = b
        return best

    @staticmethod
    def _unit_at(state: GameState, x: int, y: int, exclude: BotState | None) -> bool:
        for v in state.villages:
            for b in v.bots:
                if not b.is_alive:
                    continue
                if exclude is not None and b.bot_id == exclude.bot_id:
                    continue
                if b.position == (x, y):
                    return True
        return False

    @staticmethod
    def _merge_combat_events(
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(a)
        for k in ("damage_dealt", "damage_taken", "kills"):
            out[k] = {
                0: int(a[k].get(0, 0)) + int(b[k].get(0, 0)),
                1: int(a[k].get(1, 0)) + int(b[k].get(1, 0)),
            }
        out["building_damage"] = list(a.get("building_damage", [])) + list(
            b.get("building_damage", [])
        )
        return out

    @staticmethod
    def _terminal_update(state: GameState, config: Mapping[str, Any]) -> int | None:
        """Set ``is_done``/``winner`` if TH destroyed or stalemate."""
        for v in state.villages:
            ths = [b for b in v.buildings if b.building_type == BuildingType.TOWNHALL]
            if ths and ths[0].hp <= 0:
                state.is_done = True
                state.winner = 1 - v.team
                return int(state.winner)
        alive0 = sum(1 for b in state.villages[0].bots if b.is_alive)
        alive1 = sum(1 for b in state.villages[1].bots if b.is_alive)
        if alive0 == 0 and alive1 == 0:
            state.is_done = True
            state.winner = None
            return None
        if alive0 == 0:
            state.is_done = True
            state.winner = 1
            return 1
        if alive1 == 0:
            state.is_done = True
            state.winner = 0
            return 0

        thresh = int(config["rewards"]["village"]["stagnation_threshold"])
        if state.villages[0].ticks_without_progress >= thresh and state.villages[
            1
        ].ticks_without_progress >= thresh:
            state.is_done = True
            state.winner = None
            return None
        return None

    @staticmethod
    def _bot_events_for(
        bot: BotState,
        merged: Mapping[str, Any],
        action: int,
    ) -> dict[str, Any]:
        ev: dict[str, Any] = {"global_scale": 1.0}
        if action == 0:
            ev["noop"] = 1.0
        if merged.get("kills", {}).get(bot.team, 0) > 0:
            ev["kill"] = 1.0
        if merged.get("damage_taken", {}).get(bot.team, 0) > 0:
            ev["damage_taken"] = float(merged["damage_taken"][bot.team])
        if merged.get("damage_dealt", {}).get(bot.team, 0) > 0:
            ev["damage_dealt"] = float(merged["damage_dealt"][bot.team])
        if not bot.is_alive:
            ev["death"] = 1.0
        return ev
