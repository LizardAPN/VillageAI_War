"""Unit tests for MAPPOEpisodeMetricsCallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("stable_baselines3")

from village_ai_war.training.mappo_episode_metrics_callback import MAPPOEpisodeMetricsCallback


def test_win_townhall_frac_counts_joint_outcome_and_reason() -> None:
    cb = MAPPOEpisodeMetricsCallback(window=10, verbose=0)
    cb.model = SimpleNamespace(logger=None)
    cb.locals = {
        "infos": [
            {"episode_outcome": "win", "terminal_reason": "townhall_destroyed"},
            {"episode_outcome": "win", "terminal_reason": "team1_eliminated"},
            {"episode_outcome": "loss", "terminal_reason": "townhall_destroyed"},
        ]
    }
    assert cb._on_step() is True
    assert cb.win_townhall_frac() == pytest.approx(1.0 / 3.0)


def test_win_townhall_frac_empty_window() -> None:
    cb = MAPPOEpisodeMetricsCallback(window=10, verbose=0)
    assert cb.win_townhall_frac() == 0.0
