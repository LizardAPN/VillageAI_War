"""Per-bot reward from event tallies and global mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from village_ai_war.rewards.global_reward import mode_coefficient
from village_ai_war.state import BotState, GlobalRewardMode


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
