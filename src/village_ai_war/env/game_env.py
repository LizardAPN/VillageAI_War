"""Gymnasium environment for Village AI War."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat, cast

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
        render_mode: ``"human"`` | ``"human_3d"`` | ``"rgb_array"`` | ``"rgb_array_3d"`` | ``None``.
    """

    metadata = {"render_modes": ["human", "human_3d", "rgb_array", "rgb_array_3d"]}

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
        self._opponent_controlled_bot_id: int = 0
        self._renderer: Any = None
        self._last_tick_merged: dict[str, Any] = {}
        self._tick_start_positions: dict[int, tuple[int, int]] = {}
        self._prev_distances: dict[int, float] = {}
        self._tick_food_by_bot: dict[int, int] = {}
        self._tick_builder_repair_pct: dict[int, float] = {}
        self._shaping_snapshot: dict[str, Any] = {}
        self._melee_intents: list[tuple[int, int, tuple[int, int]]] = []

        n = int(config["map"]["size"])
        max_bots = int(config["game"].get("max_bots_for_role_change", 32))
        self._village_space = VillageActionSpace(n, max_bots=max_bots)
        self._bot_obs = BotObsBuilder(n, config=config)
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

    @property
    def game_state(self) -> GameState | None:
        """Current simulation state (``None`` before ``reset``)."""
        return self._state

    def begin_mappo_tick(self) -> None:
        """Clear pending melee intents before MAPPO queues bot actions for one simulation tick."""
        self._melee_intents.clear()

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
        self._prev_distances.clear()
        self._tick_start_positions.clear()

        # Pick controlled bot for bot mode (optional role filter)
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
            opp_team = 1 - self.team
            opp_alive = [b for b in self._state.villages[opp_team].bots if b.is_alive]
            self._opponent_controlled_bot_id = opp_alive[0].bot_id if opp_alive else 0

        obs = self._build_obs()
        info = self._info_dict()
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._state is not None and self._rng is not None
        self.snapshot_bot_positions_for_tick()
        state = self._state

        manager_action: dict[str, Any] | None = None
        interval = int(self.config["game"]["manager_interval"])
        if self.mode in ("village", "full") and state.tick % interval == 0:
            dec = decode_village_action(self._village_space, int(action))
            manager_action = dec
            self._apply_village_decision(self.team, dec)
        if self.mode == "full" and state.tick % interval == 0:
            self._apply_village_decision(1 - self.team, {"kind": "noop"})

        melee_intents: list[tuple[int, int, tuple[int, int]]] = []
        learner_bot_action: int | None = None

        if self.mode == "bot":
            learner_bot_action = int(action)
            self._apply_bot_action(
                self.team, self._controlled_bot_id, learner_bot_action, melee_intents
            )
            ex = frozenset({(self.team, self._controlled_bot_id)})
            self._step_all_bots_with_policy(None, melee_intents, exclude=ex)
        else:
            self._step_all_bots_with_policy(None, melee_intents, exclude=None)

        return self._advance_tick_after_bots(
            melee_intents,
            manager_action=manager_action,
            learner_bot_action=learner_bot_action,
        )

    def _simulation_tick(
        self,
        *,
        manager_action: dict[str, Any] | None,
        learner_bot_action: int | None = None,
        learner_bot_actions: dict[int, int] | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run combat/economy/buildings after bot actions queued via :meth:`queue_bot_action`."""
        out = self._run_simulation_phase(
            self._melee_intents,
            manager_action=manager_action,
            learner_bot_action=learner_bot_action,
            learner_bot_actions=learner_bot_actions,
        )
        self._melee_intents.clear()
        return out

    def step_with_opponent(
        self,
        red_action: int,
        blue_action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Single tick in ``bot`` mode with explicit actions for both teams."""
        if self.mode != "bot":
            raise ValueError("step_with_opponent requires mode='bot'")
        assert self._state is not None and self._rng is not None
        self.snapshot_bot_positions_for_tick()
        melee_intents: list[tuple[int, int, tuple[int, int]]] = []
        self._apply_bot_action(0, self._controlled_bot_id, int(red_action), melee_intents)
        self._apply_bot_action(1, self._opponent_controlled_bot_id, int(blue_action), melee_intents)
        ex = frozenset(
            {
                (0, self._controlled_bot_id),
                (1, self._opponent_controlled_bot_id),
            }
        )
        self._step_all_bots_with_policy(None, melee_intents, exclude=ex)
        return self._advance_tick_after_bots(
            melee_intents,
            manager_action=None,
            learner_bot_action=int(red_action),
        )

    def queue_bot_action(self, team: int, bot_id: int, action: int) -> None:
        """Queue one bot action; melee intent goes to the instance buffer (MAPPO tick)."""
        self._apply_bot_action_to_list(team, bot_id, int(action), self._melee_intents)

    def step_village_only(
        self,
        red_village_action: int,
        blue_village_action: int,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply both managers' decisions, then resolve combat/economy (bots already moved)."""
        if self.mode not in ("village", "full"):
            raise ValueError("step_village_only requires mode='village' or 'full'")
        assert self._state is not None
        state = self._state
        interval = int(self.config["game"]["manager_interval"])
        manager_action: dict[str, Any] | None = None
        if state.tick % interval == 0:
            red_dec = decode_village_action(self._village_space, int(red_village_action))
            blue_dec = decode_village_action(self._village_space, int(blue_village_action))
            self._apply_village_decision(0, red_dec)
            self._apply_village_decision(1, blue_dec)
            manager_action = red_dec if self.team == 0 else blue_dec
        return self._advance_tick_after_bots(
            melee_intents,
            manager_action=manager_action,
            learner_bot_action=None,
        )

    def get_village_observation(self, team: int) -> dict[str, np.ndarray]:
        """Return the village manager observation for ``team`` (``map`` + ``village`` tensors).

        Args:
            team: ``0`` (red) or ``1`` (blue).

        Raises:
            ValueError: If ``mode`` is not ``village`` or ``full``.
        """
        if self.mode not in ("village", "full"):
            raise ValueError("get_village_observation requires mode='village' or 'full'")
        return self._get_village_obs(team)

    def run_bots_then_village_decisions(
        self,
        bot_policy: Any,
        red_village_action: int,
        blue_village_action: int,
        *,
        human_team: int | None = None,
        human_bot_actions: Mapping[int, int] | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run one tick: all bots act, then both managers' village actions are applied.

        Matches the order used in self-play village training. Use ``bot_policy=None``
        for random bot moves, or pass a policy with ``predict(obs)`` for scripted bots.

        Args:
            bot_policy: Optional policy for non-human bots, or ``None`` for random actions.
            red_village_action: Flattened village action index for team 0.
            blue_village_action: Flattened village action index for team 1.
            human_team: If set with ``human_bot_actions``, that team's alive bots use
                the given actions first; policy fills other bots (excluding those pairs).

        Raises:
            ValueError: If ``mode`` is not ``village`` or ``full``, or human actions
                omit an alive bot on ``human_team``.
        """
        if self.mode not in ("village", "full"):
            raise ValueError("run_bots_then_village_decisions requires mode='village' or 'full'")
        self.snapshot_bot_positions_for_tick()
        melee_intents: list[tuple[int, int, tuple[int, int]]] = []
        ex: frozenset[tuple[int, int]] | None = None
        if human_team is not None and human_bot_actions is not None:
            alive_ids = sorted(
                int(b.bot_id) for b in self._state.villages[int(human_team)].bots if b.is_alive
            )
            missing = [bid for bid in alive_ids if bid not in human_bot_actions]
            if missing:
                raise ValueError(
                    f"human_bot_actions must cover all alive bots on team {human_team}; "
                    f"missing bot_id(s): {missing}"
                )
            for bid in alive_ids:
                self._apply_bot_action(
                    int(human_team),
                    bid,
                    int(human_bot_actions[bid]),
                    melee_intents,
                )
            ex = frozenset((int(human_team), bid) for bid in alive_ids)
        self._step_all_bots_with_policy(bot_policy, melee_intents, exclude=ex)
        return self.step_village_only(
            int(red_village_action), int(blue_village_action), melee_intents
        )

    def action_masks(self, team: int | None = None) -> np.ndarray:
        """Boolean mask of valid village actions (MaskablePPO)."""
        assert self._state is not None
        t = self.team if team is None else team
        if self.mode not in ("village", "full"):
            return np.ones((self._village_space.n_actions,), dtype=bool)
        m = ActionMasker.compute_masks(self._state, t, self.config)
        interval = int(self.config["game"]["manager_interval"])
        if self._state.tick % interval != 0:
            m = np.zeros_like(m)
            m[self._village_space.offset_noop] = True
        return m

    def render(self, *, overlay_lines: Sequence[str] | None = None) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        mode = cast(str, self.render_mode)
        if self._renderer is not None:
            tag = getattr(self._renderer, "_render_backend", "pygame")
            want_3d = mode in ("human_3d", "rgb_array_3d")
            if (tag == "pygame" and want_3d) or (tag == "moderngl" and not want_3d):
                self._renderer.close()
                self._renderer = None

        if mode in ("human_3d", "rgb_array_3d"):
            from village_ai_war.rendering.moderngl_3d_renderer import Moderngl3DRenderer

            if self._renderer is None:
                self._renderer = Moderngl3DRenderer(self.config, self._state)
            return self._renderer.render(self._state, mode=mode)

        from village_ai_war.rendering.pygame_renderer import PygameRenderer

        if self._renderer is None:
            self._renderer = PygameRenderer(self.config, self._state)
        return self._renderer.render(self._state, mode=mode, overlay_lines=overlay_lines)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _build_obs(self) -> Any:
        assert self._state is not None
        if self.mode == "bot":
            return self._bot_obs.build(self._state, self._controlled_bot_id, self.config)
        return self._vil_obs.build(self._state, self.team)

    def _info_dict(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "tick": self._state.tick,
            "team": self.team,
            "mode": self.mode,
        }

    def _get_single_bot_obs(self, bot_id: int, team: int | None = None) -> np.ndarray:
        assert self._state is not None
        if team is not None:
            bot = next(
                (b for b in self._state.villages[team].bots if b.bot_id == bot_id),
                None,
            )
            if bot is None:
                return np.zeros((BotObsBuilder.OBS_DIM,), dtype=np.float32)
        return self._bot_obs.build(self._state, bot_id, self.config)

    def _get_bot_obs(self, team: int) -> np.ndarray | None:
        """Observation for the first alive bot on ``team`` (opponent policy input)."""
        assert self._state is not None
        alive = [b for b in self._state.villages[team].bots if b.is_alive]
        if not alive:
            return None
        return self._get_single_bot_obs(alive[0].bot_id)

    def _get_village_obs(self, team: int) -> dict[str, np.ndarray]:
        assert self._state is not None
        return self._vil_obs.build(self._state, team)

    def _step_all_bots_with_policy(
        self,
        bot_policy: Any,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
        *,
        exclude: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        """Step every alive bot with ``bot_policy.predict`` or random actions."""
        assert self._state is not None and self._rng is not None
        ex = exclude or frozenset()
        for team in (0, 1):
            for bot in self._state.villages[team].bots:
                if not bot.is_alive or (team, bot.bot_id) in ex:
                    continue
                obs = self._get_single_bot_obs(bot.bot_id)
                if bot_policy is not None:
                    act, _ = bot_policy.predict(obs, deterministic=False)
                    act_int = int(np.asarray(act).reshape(-1)[0])
                else:
                    act_int = int(self._rng.integers(0, self.BOT_ACTIONS))
                self._apply_bot_action(team, bot.bot_id, act_int, melee_intents)

    def snapshot_bot_positions_for_tick(self) -> None:
        """Call before bot movement each tick (movement shaping + per-tick counters)."""
        assert self._state is not None
        self._tick_start_positions = {
            b.bot_id: tuple(b.position)
            for v in self._state.villages
            for b in v.bots
            if b.is_alive
        }
        self._tick_food_by_bot = {}
        self._tick_builder_repair_pct = {}

    def _advance_tick_after_bots(
        self,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
        *,
        manager_action: dict[str, Any] | None,
        learner_bot_action: int | None = None,
        learner_bot_actions: dict[int, int] | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._run_simulation_phase(
            melee_intents,
            manager_action=manager_action,
            learner_bot_action=learner_bot_action,
            learner_bot_actions=learner_bot_actions,
        )

    def _run_simulation_phase(
        self,
        melee_intents: list[tuple[int, int, tuple[int, int]]],
        *,
        manager_action: dict[str, Any] | None,
        learner_bot_action: int | None = None,
        learner_bot_actions: dict[int, int] | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert self._state is not None
        state = self._state
        terminated = False
        truncated = False

        self._shaping_snapshot = GameEnv._build_shaping_snapshot(state)

        cmb = CombatSystem.apply_melee_intents(state, self.config, list(melee_intents))
        eco = EconomySystem.step(state, self.config)
        bld = BuildingSystem.construction_tick(state, self.config)
        tw = CombatSystem.apply_tower_fire(state, self.config)
        CombatSystem.tick_cooldowns(state)

        merged = GameEnv._merge_combat_events(cmb, tw)
        merged["losses"] = {0: cmb["kills"].get(1, 0), 1: cmb["kills"].get(0, 0)}
        merged["resource_collected"] = eco.get("resource_collected", {})
        merged["food_produced"] = eco.get("food_produced", {})
        merged["building_completed"] = bld.get("buildings_completed", [])
        merged["resource_collected_by_bot"] = dict(eco.get("resource_collected_by_bot", {}))
        fpb: dict[int, int] = {}
        for bid, amt in self._tick_food_by_bot.items():
            fpb[bid] = fpb.get(bid, 0) + amt
        merged["food_produced_by_bot"] = fpb
        merged["block_placed_by_bot"] = dict(bld.get("block_placed_by_bot", {}))
        merged["repair_pct_by_bot"] = dict(self._tick_builder_repair_pct)

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

        won, term_reason = GameEnv._terminal_update(state, self.config)
        if won is not None:
            terminated = True
        if state.tick >= state.max_ticks:
            truncated = True
            state.is_done = True
        elif state.is_done:
            # Draw (no alive bots on both sides), stagnation, etc.: ``won`` is None
            # but the episode must still end for Gymnasium / UI.
            terminated = True

        state.tick += 1

        reward: float
        if self.mode == "bot" and learner_bot_actions is not None:
            mode = state.villages[self.team].global_reward_mode
            reward = 0.0
            n_controlled = 0
            for bid, act in learner_bot_actions.items():
                bot = next(
                    (b for v in state.villages for b in v.bots if b.bot_id == bid),
                    None,
                )
                if bot is not None:
                    bev = self._bot_events_for(bot, merged, int(act), state)
                    reward += float(BotRewardCalculator.compute(bev, bot, mode, self.config))
                    n_controlled += 1
            reward = self._finalize_bot_team_reward(
                reward, n_controlled, merged, state, terminated, truncated
            )
        elif self.mode == "bot" and learner_bot_action is not None:
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
                bev = self._bot_events_for(bot, merged, learner_bot_action, state)
                reward = float(BotRewardCalculator.compute(bev, bot, mode, self.config))
                reward = self._finalize_bot_team_reward(
                    reward, 1, merged, state, terminated, truncated
                )
            else:
                reward = self._finalize_bot_team_reward(
                    0.0, 0, merged, state, terminated, truncated
                )
        else:
            reward = float(
                VillageRewardCalculator.compute(
                    merged,
                    state.villages[self.team],
                    self.config,
                    terminated or truncated,
                    True
                    if won == self.team
                    else (False if won is not None and won != self.team else None),
                )
            )

        self._last_tick_merged = merged

        obs = self._build_obs()
        info = {
            **self._info_dict(),
            "kills_this_tick": kills_this_tick,
            "resources_delta": resources_delta,
            "manager_action": manager_action,
            "bots_alive": bots_alive,
            "winner": state.winner,
        }
        if truncated:
            info["terminal_reason"] = "max_ticks"
            info["episode_outcome"] = "truncated"
        elif terminated:
            if term_reason is not None:
                info["terminal_reason"] = term_reason
            w = state.winner
            if w is None:
                info["episode_outcome"] = "draw"
            elif w == self.team:
                info["episode_outcome"] = "win"
            else:
                info["episode_outcome"] = "loss"
        return obs, reward, terminated, truncated, info

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
        self._apply_bot_action_to_list(team, bot_id, action, melee_intents)

    def _apply_bot_action_to_list(
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
            self._tick_food_by_bot[bot_id] = self._tick_food_by_bot.get(bot_id, 0) + 1
        elif action == 11 and bot.role == Role.BUILDER:
            for v in self._state.villages:
                for b in v.buildings:
                    if b.team != team or b.hp >= b.max_hp:
                        continue
                    bx, by = b.position
                    if abs(bx - ax) + abs(by - ay) == 1:
                        before = b.hp
                        delta = max(1, int(0.1 * b.max_hp))
                        b.hp = min(b.max_hp, b.hp + delta)
                        gained = b.hp - before
                        if gained > 0 and b.max_hp > 0:
                            self._tick_builder_repair_pct[bot_id] = (
                                self._tick_builder_repair_pct.get(bot_id, 0.0)
                                + float(gained) / float(b.max_hp)
                            )

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
    def _build_shaping_snapshot(state: GameState) -> dict[str, Any]:
        n = state.map_size
        amounts = np.asarray(state.resource_amounts, dtype=np.int32)
        res = np.asarray(state.resources, dtype=np.int32)
        res_cells: list[tuple[int, int]] = []
        field_cells: list[tuple[int, int]] = []
        for y in range(n):
            for x in range(n):
                if amounts[y, x] <= 0:
                    continue
                layer = int(res[y, x])
                if layer == int(ResourceLayer.NONE):
                    continue
                res_cells.append((x, y))
                if layer == int(ResourceLayer.FIELD):
                    field_cells.append((x, y))
        enemy_for: dict[int, list[tuple[int, int]]] = {}
        for t in (0, 1):
            et = 1 - t
            enemy_for[t] = [
                tuple(b.position) for b in state.villages[et].bots if b.is_alive
            ]
        bp_by_team: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
        for bp in state.blueprints:
            tm = int(bp["team"])
            bp_by_team.setdefault(tm, []).append(
                (int(bp["position"][0]), int(bp["position"][1]))
            )
        return {
            "enemy_for": enemy_for,
            "res_cells": res_cells,
            "field_cells": field_cells,
            "bp_by_team": bp_by_team,
        }

    @staticmethod
    def _nearest_dist(pos: tuple[int, int], targets: list[tuple[int, int]]) -> float:
        if not targets:
            return float("inf")
        px, py = pos
        return float(min(abs(px - tx) + abs(py - ty) for tx, ty in targets))

    def _finalize_bot_team_reward(
        self,
        per_bot_sum: float,
        n_controlled: int,
        merged: Mapping[str, Any],
        state: GameState,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Mean/sum over controlled bots, then team shaping and terminal win/loss."""
        rcfg = self.config["rewards"]["bot"]
        agg = str(rcfg.get("reward_aggregate", "mean")).lower()
        r = per_bot_sum
        if agg == "mean" and n_controlled > 0:
            r /= float(n_controlled)
        vil = state.villages[self.team]
        r += BotRewardCalculator.team_addon(merged, vil, self.config)
        r += BotRewardCalculator.terminal_addon(
            self.config, terminated or truncated, self.team, state.winner
        )
        return r

    def _bot_events_for(
        self,
        bot: BotState,
        merged: Mapping[str, Any],
        action: int,
        state: GameState,
    ) -> dict[str, Any]:
        snap = self._shaping_snapshot
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

        rcb = merged.get("resource_collected_by_bot") or {}
        if isinstance(rcb, Mapping) and bot.bot_id in rcb:
            ev["resource_collected"] = float(rcb[bot.bot_id])

        fpb = merged.get("food_produced_by_bot") or {}
        if isinstance(fpb, Mapping) and bot.bot_id in fpb:
            ev["food_produced"] = float(fpb[bot.bot_id])

        bpb = merged.get("block_placed_by_bot") or {}
        if isinstance(bpb, Mapping) and bot.bot_id in bpb:
            ev["block_placed"] = float(bpb[bot.bot_id])

        rpr = merged.get("repair_pct_by_bot") or {}
        if isinstance(rpr, Mapping) and bot.bot_id in rpr:
            ev["repair_pct"] = float(rpr[bot.bot_id])

        pos0 = self._tick_start_positions.get(bot.bot_id, bot.position)
        pos1 = bot.position
        t0 = (int(pos0[0]), int(pos0[1]))
        t1 = (int(pos1[0]), int(pos1[1]))

        if bot.role == Role.WARRIOR:
            enemies = list(snap.get("enemy_for", {}).get(bot.team, []))
            if enemies:
                d0 = GameEnv._nearest_dist(t0, enemies)
                d1 = GameEnv._nearest_dist(t1, enemies)
                if d1 < d0:
                    ev["approach_enemy"] = 1.0
                elif d1 > d0:
                    ev["retreat_penalty"] = 1.0
                self._prev_distances[bot.bot_id] = d1
        elif bot.role == Role.GATHERER:
            cells = list(snap.get("res_cells", []))
            if cells:
                d0 = GameEnv._nearest_dist(t0, cells)
                d1 = GameEnv._nearest_dist(t1, cells)
                if d1 < d0:
                    ev["approach_resource"] = 1.0
                self._prev_distances[bot.bot_id] = d1
            x, y = t1
            n = state.map_size
            if 0 <= x < n and 0 <= y < n:
                amt = int(np.asarray(state.resource_amounts, dtype=np.int32)[y, x])
                layer = int(np.asarray(state.resources, dtype=np.int32)[y, x])
                if amt > 0 and layer != int(ResourceLayer.NONE):
                    ev["idle_at_resource"] = 1.0
        elif bot.role == Role.FARMER:
            fields = list(snap.get("field_cells", []))
            if fields:
                d0 = GameEnv._nearest_dist(t0, fields)
                d1 = GameEnv._nearest_dist(t1, fields)
                if d1 < d0:
                    ev["approach_field"] = 1.0
                self._prev_distances[bot.bot_id] = d1
            x, y = t1
            n = state.map_size
            if 0 <= x < n and 0 <= y < n:
                amt = int(np.asarray(state.resource_amounts, dtype=np.int32)[y, x])
                layer = int(np.asarray(state.resources, dtype=np.int32)[y, x])
                if amt > 0 and layer == int(ResourceLayer.FIELD):
                    ev["idle_at_field"] = 1.0
        elif bot.role == Role.BUILDER:
            bps = list(snap.get("bp_by_team", {}).get(bot.team, []))
            if bps:
                d0 = GameEnv._nearest_dist(t0, bps)
                d1 = GameEnv._nearest_dist(t1, bps)
                if d1 < d0:
                    ev["approach_blueprint"] = 1.0
                self._prev_distances[bot.bot_id] = d1

        return ev

    @staticmethod
    def _terminal_update(
        state: GameState, config: Mapping[str, Any]
    ) -> tuple[int | None, str | None]:
        """Set ``is_done``/``winner`` if TH destroyed or stalemate.

        Returns ``(winner_team_id, terminal_reason)`` when the episode ends from this
        check; otherwise ``(None, None)``. ``winner_team_id`` is ``None`` for a draw.
        """
        for v in state.villages:
            ths = [b for b in v.buildings if b.building_type == BuildingType.TOWNHALL]
            if ths and ths[0].hp <= 0:
                state.is_done = True
                state.winner = 1 - v.team
                return int(state.winner), "townhall_destroyed"
        alive0 = sum(1 for b in state.villages[0].bots if b.is_alive)
        alive1 = sum(1 for b in state.villages[1].bots if b.is_alive)
        if alive0 == 0 and alive1 == 0:
            state.is_done = True
            state.winner = None
            return None, "mutual_elimination"
        if alive0 == 0:
            state.is_done = True
            state.winner = 1
            return 1, "team0_eliminated"
        if alive1 == 0:
            state.is_done = True
            state.winner = 0
            return 0, "team1_eliminated"

        thresh = int(config["rewards"]["village"]["stagnation_threshold"])
        if state.villages[0].ticks_without_progress >= thresh and state.villages[
            1
        ].ticks_without_progress >= thresh:
            state.is_done = True
            state.winner = None
            return None, "stagnation"
        return None, None
