"""Global reward mode helpers for bot reward shaping."""

from __future__ import annotations

from village_ai_war.state import GlobalRewardMode


def mode_coefficient(mode: GlobalRewardMode, defend: float, attack: float, gather: float) -> float:
    """Scalar multiplier for global-mode shaping."""
    if mode == GlobalRewardMode.DEFEND:
        return defend
    if mode == GlobalRewardMode.ATTACK:
        return attack
    if mode == GlobalRewardMode.GATHER:
        return gather
    return 0.0
