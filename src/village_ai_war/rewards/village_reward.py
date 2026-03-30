"""High-level village manager reward."""

from __future__ import annotations

from typing import Any, Mapping

from village_ai_war.state import VillageState


class VillageRewardCalculator:
    """Dense shaping plus terminal win/loss bonuses."""

    @staticmethod
    def compute(
        events: Mapping[str, Any],
        village_state: VillageState,
        config: Mapping[str, Any],
        terminated: bool,
        won: bool | None,
    ) -> float:
        """Return village reward for the tick."""
        rcfg = config["rewards"]["village"]
        r = 0.0
        team = village_state.team
        eco = float(rcfg["economy_coeff"]) * (
            village_state.resources.wood
            + village_state.resources.stone
            + village_state.resources.food
        ) / 1000.0
        r += eco

        if "kills" in events and isinstance(events["kills"], dict):
            r += float(rcfg["kill_reward"]) * float(events["kills"].get(team, 0))
        if "losses" in events and isinstance(events["losses"], dict):
            r += float(rcfg["loss_penalty"]) * float(events["losses"].get(team, 0))

        if events.get("building_completed"):
            r += float(rcfg["building_reward"]) * len(events["building_completed"])

        if village_state.ticks_without_progress > int(rcfg["stagnation_threshold"]):
            r += float(rcfg["stagnation_penalty"])

        if terminated:
            if won is True:
                r += float(rcfg["win"])
            elif won is False:
                r += float(rcfg["loss"])
        return r
