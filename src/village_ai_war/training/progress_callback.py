"""Loguru progress logging for a single SB3 ``learn()`` phase (unified training)."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class UnifiedPhaseProgressCallback(BaseCallback):
    """Log phase progress, ETA, env-steps/s, and SB3 iteration wall time (rollout+train)."""

    def __init__(
        self,
        phase_name: str,
        cycle_one_based: int,
        n_cycles: int,
        steps_budget: int,
        timesteps_start: int,
        log_interval_sec: float = 30.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.phase_name = phase_name
        self.cycle_one_based = cycle_one_based
        self.n_cycles = n_cycles
        self.steps_budget = max(1, int(steps_budget))
        self.timesteps_start = int(timesteps_start)
        self.log_interval_sec = float(log_interval_sec)
        self._t0 = time.perf_counter()
        self._last_log_t = self._t0
        self._rollout_idx = 0
        self._rollout_start_t: float | None = None
        self._last_iter_wall_s: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        if self._rollout_start_t is not None:
            self._last_iter_wall_s = now - self._rollout_start_t
        self._rollout_start_t = now

    def _on_rollout_end(self) -> None:
        assert self.model is not None
        self._rollout_idx += 1
        now = time.perf_counter()
        done = int(self.model.num_timesteps) - self.timesteps_start
        if done <= 0:
            return

        due_time = (now - self._last_log_t) >= self.log_interval_sec
        due_complete = done >= self.steps_budget
        first = self._rollout_idx == 1
        if not first and not due_time and not due_complete:
            return

        self._last_log_t = now
        elapsed = now - self._t0
        frac = min(1.0, done / self.steps_budget)
        remaining = max(0, self.steps_budget - done)
        eta_sec = (elapsed / done) * remaining if done > 0 else 0.0
        fps = done / elapsed if elapsed > 0 else 0.0

        iter_part = (
            f" | prev_full_iter ~{self._last_iter_wall_s:.1f}s"
            if self._last_iter_wall_s is not None
            else ""
        )
        if first:
            logger.info(
                "Unified progress | cycle {}/{} | phase {} | first rollout done | "
                "env_steps {}/{} ({:.1%}) | elapsed {:.0f}s | ETA ~{:.0f}s | "
                "~{:.1f} env_steps/s{} (first iter often slower — GPU warmup)",
                self.cycle_one_based,
                self.n_cycles,
                self.phase_name,
                done,
                self.steps_budget,
                frac,
                elapsed,
                eta_sec,
                fps,
                iter_part,
            )
        else:
            logger.info(
                "Unified progress | cycle {}/{} | phase {} | SB3 iter {} | "
                "env_steps {}/{} ({:.1%}) | elapsed {:.0f}s | ETA ~{:.0f}s | "
                "~{:.1f} env_steps/s{}",
                self.cycle_one_based,
                self.n_cycles,
                self.phase_name,
                self._rollout_idx,
                done,
                self.steps_budget,
                frac,
                elapsed,
                eta_sec,
                fps,
                iter_part,
            )


def make_progress_callback(
    model: Any,
    *,
    phase_name: str,
    cycle_zero_based: int,
    n_cycles: int,
    steps_budget: int,
    log_interval_sec: float,
) -> UnifiedPhaseProgressCallback:
    """Snapshot ``model.num_timesteps`` as start of this ``learn()`` call."""
    start = int(model.num_timesteps)
    return UnifiedPhaseProgressCallback(
        phase_name=phase_name,
        cycle_one_based=cycle_zero_based + 1,
        n_cycles=n_cycles,
        steps_budget=steps_budget,
        timesteps_start=start,
        log_interval_sec=log_interval_sec,
    )
