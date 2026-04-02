"""Per-bot reward from event tallies and global mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from village_ai_war.rewards.global_reward import mode_coefficient
from village_ai_war.state import BotState, GlobalRewardMode, VillageState


class BotRewardCalculator:
    """Computes shaped bot rewards from structured events."""

    @staticmethod
    def compute(
        events: Mapping[str, Any],
        bot_state: BotState,
        mode: GlobalRewardMode,
        config: Mapping[str, Any],
    ) -> float:
        """Return scalar reward for ``bot_state`` this tick.

        ``events`` contains optional keys matching per-role weights in config
        (e.g. ``damage_dealt``, ``kill``, ``resource_collected``, ``noop``).

        Args:
            events: Per-bot or tick-local scaled counts (non-negative floats).
            bot_state: The learning bot.
            mode: Village ``GlobalRewardMode``.
            config: Merged config (``rewards.bot`` subtree).
        """
        rcfg = config["rewards"]["bot"]
        alpha = float(rcfg["alpha"])
        role_key = bot_state.role.name.lower()
        role_cfg: Mapping[str, Any] = rcfg[role_key]
        gcfg = rcfg["global_modes"]

        local = 0.0
        for key, weight in role_cfg.items():
            if key in events and isinstance(weight, (int, float)):
                local += float(weight) * float(events[key])

        gmult = mode_coefficient(
            mode,
            float(gcfg["defend_coeff"]),
            float(gcfg["attack_coeff"]),
            float(gcfg["gather_coeff"]),
        )
        global_part = gmult * float(events.get("global_scale", 1.0))
        return float(alpha) * local + (1.0 - float(alpha)) * global_part

    @staticmethod
    def team_addon(
        merged: Mapping[str, Any],
        village: VillageState,
        config: Mapping[str, Any],
    ) -> float:
        """Team-level shaping once per tick (hunger, food stock, net food gain)."""
        rcfg = config["rewards"]["bot"]
        raw = rcfg.get("team")
        if not isinstance(raw, Mapping):
            return 0.0
        team_cfg: Mapping[str, Any] = raw
        team = int(village.team)
        r = 0.0

        hd_map = merged.get("hunger_damage")
        hd = int(hd_map.get(team, 0)) if isinstance(hd_map, Mapping) else 0
        hp = float(team_cfg.get("hunger_damage_penalty", 0.0))
        if hp != 0.0 and hd > 0:
            r += hp * float(hd)

        alive = sum(1 for b in village.bots if b.is_alive)
        fbonus = float(team_cfg.get("fed_no_hunger_bonus", 0.0))
        if alive > 0 and hd == 0 and fbonus != 0.0:
            r += fbonus

        coeff = float(team_cfg.get("food_security_coeff", 0.0))
        if coeff != 0.0:
            thresh = int(team_cfg.get("food_security_threshold", 0))
            food = int(village.resources.food)
            if food > thresh:
                r += coeff * float(food - thresh) / 1000.0

        fd_map = merged.get("food_delta")
        if isinstance(fd_map, Mapping):
            fd = int(fd_map.get(team, 0))
            fdc = float(team_cfg.get("food_delta_positive_coeff", 0.0))
            if fdc != 0.0 and fd > 0:
                r += fdc * float(fd) / 100.0

        return r

    @staticmethod
    def terminal_addon(
        config: Mapping[str, Any],
        episode_done: bool,
        team: int,
        winner: int | None,
    ) -> float:
        """Sparse win/loss/draw bonus for bot training (mirrors village terminal intent)."""
        if not episode_done:
            return 0.0
        rcfg = config["rewards"]["bot"]
        tc = rcfg.get("terminal")
        if not isinstance(tc, Mapping):
            return 0.0
        if winner is None:
            return float(tc.get("draw", 0.0))
        if int(winner) == int(team):
            return float(tc.get("win", 0.0))
        return float(tc.get("loss", 0.0))
