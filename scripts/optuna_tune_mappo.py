#!/usr/bin/env python3
"""MAPPO hyperparameter search with Optuna (Hydra compose, no @hydra.main)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import optuna
from loguru import logger
from optuna import TrialPruned

from village_ai_war.config_load import load_project_config
from village_ai_war.training.train_mappo_bots import run_mappo_bots_training


def _study_path_slug(study_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", study_name).strip("_")
    return s or "study"


def _valid_batch_sizes(buffer: int) -> list[int]:
    opts = [32, 64, 128, 256, 512, 1024]
    valid = [b for b in opts if b <= buffer and buffer % b == 0]
    if valid:
        return valid
    for b in range(8, buffer + 1):
        if buffer % b == 0:
            return [b]
    if buffer >= 1:
        return [1]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for MAPPO.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (default 1 for RL).")
    parser.add_argument("--study-name", type=str, default="mappo_hpo")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna RDB URL, e.g. sqlite:///optuna_mappo.db (empty = in-memory).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-name", type=str, default="default")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--selfplay-iterations", type=int, default=2)
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Set logging.use_tensorboard=false for faster trials.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra-style overrides (repeatable). Applied after trial suggestions.",
    )
    args = parser.parse_args()

    slug = _study_path_slug(args.study_name)
    user_overrides: list[str] = list(args.override)

    def objective(trial: optuna.Trial) -> float:
        n_envs = int(args.n_envs)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
        buffer = n_envs * n_steps
        valid_bs = _valid_batch_sizes(buffer)
        if not valid_bs:
            raise TrialPruned(f"no valid batch_size for buffer={buffer}")

        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", valid_bs)
        n_epochs = trial.suggest_int("n_epochs", 4, 15)
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        critic_hidden_dim = trial.suggest_categorical("critic_hidden_dim", [128, 256, 512])

        trial_dir = f"{slug}/trial_{trial.number}"
        overrides: list[str] = [
            f"training.total_timesteps={args.total_timesteps}",
            f"training.selfplay_iterations={args.selfplay_iterations}",
            f"training.n_envs={n_envs}",
            f"training.n_steps={n_steps}",
            f"training.batch_size={batch_size}",
            f"training.n_epochs={n_epochs}",
            f"training.learning_rate={lr}",
            f"training.ent_coef={ent_coef}",
            f"training.gamma={gamma}",
            f"training.gae_lambda={gae_lambda}",
            f"training.clip_range={clip_range}",
            f"training.vf_coef={vf_coef}",
            f"training.critic_hidden_dim={critic_hidden_dim}",
            f"training.checkpoint_dir=checkpoints/optuna/{trial_dir}",
            f"training.pool_dir=checkpoints/optuna/{trial_dir}/pool",
            f"training.log_dir=logs/optuna/{trial_dir}",
        ]
        if args.disable_tensorboard:
            overrides.append("logging.use_tensorboard=false")
        overrides.extend(user_overrides)

        flat = load_project_config(_ROOT, config_name=args.config_name, overrides=overrides)
        metrics = run_mappo_bots_training(flat, return_metrics=True)
        assert metrics is not None
        win_frac = float(metrics["win_frac"])
        logger.info(
            "trial {} finished: win_frac={} outcomes={}",
            trial.number,
            win_frac,
            metrics.get("outcome_fractions"),
        )
        return win_frac

    storage = args.storage.strip() or None
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    best = study.best_trial
    logger.info("Best trial: {} value={}", best.number, best.value)
    logger.info("Best params: {}", best.params)


if __name__ == "__main__":
    main()
