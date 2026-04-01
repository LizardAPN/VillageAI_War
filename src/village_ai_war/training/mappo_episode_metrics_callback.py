"""TensorBoard metrics for MAPPO episode outcomes (win/loss/draw/trunc + terminal reasons)."""

from __future__ import annotations

from collections import Counter, deque

from stable_baselines3.common.callbacks import BaseCallback


class MAPPOEpisodeMetricsCallback(BaseCallback):
    """Rolling-window fractions of ``episode_outcome`` and ``terminal_reason`` from env info."""

    def __init__(self, window: int = 512, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._window = max(int(window), 1)
        self._outcomes: deque[str] = deque(maxlen=self._window)
        self._reasons: deque[str] = deque(maxlen=self._window)
        self._win_townhall_flags: deque[bool] = deque(maxlen=self._window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        updated = False
        for info in infos:
            if not isinstance(info, dict) or "episode_outcome" not in info:
                continue
            outcome = str(info["episode_outcome"])
            self._outcomes.append(outcome)
            tr = info.get("terminal_reason")
            if tr is not None:
                self._reasons.append(str(tr))
            self._win_townhall_flags.append(
                outcome == "win" and info.get("terminal_reason") == "townhall_destroyed"
            )
            updated = True

        if updated and self.logger is not None and self._outcomes:
            n = len(self._outcomes)
            for k, v in Counter(self._outcomes).items():
                self.logger.record(f"mappo/outcome_frac/{k}", float(v) / float(n))
            self.logger.record("mappo/episodes_in_window", float(n))
            wth = sum(1 for x in self._win_townhall_flags if x) / float(n)
            self.logger.record("mappo/win_townhall_frac", float(wth))
            rn = len(self._reasons)
            if rn > 0:
                for k, v in Counter(self._reasons).items():
                    self.logger.record(f"mappo/terminal_reason_frac/{k}", float(v) / float(rn))
        return True

    def outcome_fractions(self) -> dict[str, float]:
        """Normalized outcome counts in the current rolling window (empty if no episodes yet)."""
        if not self._outcomes:
            return {}
        n = len(self._outcomes)
        return {str(k): float(v) / float(n) for k, v in Counter(self._outcomes).items()}

    def win_townhall_frac(self) -> float:
        """Fraction of episodes in the window that are a win via opponent town hall destroyed."""
        if not self._win_townhall_flags:
            return 0.0
        n = len(self._win_townhall_flags)
        return float(sum(1 for x in self._win_townhall_flags if x)) / float(n)
