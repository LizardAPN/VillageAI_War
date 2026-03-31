"""Optional callback: last-step global_state from vectorized env infos (logging only)."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class GlobalStateCallback(BaseCallback):
    """Stores ``global_state`` from each sub-env ``info`` (debugging; not used for critic)."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.last_global_states: list[dict[str, Any] | None] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.last_global_states = [
            inf.get("global_state") if isinstance(inf, dict) else None for inf in infos
        ]
        return True
