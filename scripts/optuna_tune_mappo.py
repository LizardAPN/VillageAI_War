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

from village_ai_war.config_load import load_project_config
from village_ai_war.training.train_mappo_bots import run_mappo_bots_training


def _study_path_slug(study_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", study_name).strip("_")
    return s or "study"


# Optuna requires the same categorical choices for a parameter name in every trial
# (see CategoricalDistribution does not support dynamic value space).
_BATCH_SIZE_CHOICES = [32, 64, 128, 256, 512, 1024]
_N_STEPS_CHOICES = [128, 256, 512]

# Correlated (gamma, gae_lambda) presets — single categorical for TPE-friendly search.
_GAMMA_GAE_PRESETS = [
    "0.97/0.92",
    "0.98/0.94",
    "0.99/0.95",
    "0.995/0.97",
    "0.999/0.98",
]

_CHECKPOINT_INTERVAL_CHOICES = [50_000, 100_000, 200_000]


def _rollout_labels(n_envs: int) -> list[str]:
    """Valid (n_steps, batch_size) pairs encoded as n{n}_bs{b} for fixed Optuna categorical space."""
    labels: list[str] = []
    for n_steps in _N_STEPS_CHOICES:
        buf = n_envs * n_steps
        for bs in _BATCH_SIZE_CHOICES:
            if bs <= buf and buf % bs == 0:
                labels.append(f"n{n_steps}_bs{bs}")
    return sorted(labels)


def _parse_rollout(label: str) -> tuple[int, int]:
    if not label.startswith("n") or "_bs" not in label:
        raise ValueError(f"bad rollout label: {label!r}")
    left, right = label.split("_bs", 1)
    return int(left[1:]), int(right)


def _parse_gamma_gae(preset: str) -> tuple[float, float]:
    parts = preset.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"bad gamma/gae preset: {preset!r}")
    return float(parts[0]), float(parts[1])


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
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--selfplay-iterations", type=int, default=4)
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

    n_envs = int(args.n_envs)
    rollout_labels = _rollout_labels(n_envs)
    if not rollout_labels:
        logger.error(
            "No valid rollout (n_steps, batch_size) for n_envs={} with batch choices {}; "
            "increase --n-envs or adjust _BATCH_SIZE_CHOICES.",
            n_envs,
            _BATCH_SIZE_CHOICES,
        )
        sys.exit(1)

    slug = _study_path_slug(args.study_name)
    user_overrides: list[str] = list(args.override)

    logger.info(
        "HPO search space uses rollout={}, gamma_gae presets={}, "
        "checkpoint_intervals={}. If you changed param names vs an existing DB study, "
        "use a new --study-name or storage file.",
        len(rollout_labels),
        len(_GAMMA_GAE_PRESETS),
        _CHECKPOINT_INTERVAL_CHOICES,
    )

    def objective(trial: optuna.Trial) -> float:
        rollout = trial.suggest_categorical("rollout", rollout_labels)
        n_steps, batch_size = _parse_rollout(rollout)

        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 4, 15)
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        gamma_gae = trial.suggest_categorical("gamma_gae", _GAMMA_GAE_PRESETS)
        gamma, gae_lambda = _parse_gamma_gae(gamma_gae)
        clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        critic_hidden_dim = trial.suggest_categorical("critic_hidden_dim", [128, 256, 512])
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)
        pool_max_size = trial.suggest_int("pool_max_size", 8, 30)
        checkpoint_interval = trial.suggest_categorical(
            "checkpoint_interval", _CHECKPOINT_INTERVAL_CHOICES
        )

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
            f"training.max_grad_norm={max_grad_norm}",
            f"training.pool_max_size={pool_max_size}",
            f"training.checkpoint_interval={checkpoint_interval}",
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
        win_townhall_frac = float(metrics["win_townhall_frac"])
        win_frac = float(metrics["win_frac"])
        logger.info(
            "trial {} finished: win_townhall_frac={} win_frac={} outcomes={}",
            trial.number,
            win_townhall_frac,
            win_frac,
            metrics.get("outcome_fractions"),
        )
        return win_townhall_frac

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
    logger.info("Best trial: {} win_townhall_frac={}", best.number, best.value)
    logger.info("Best params: {}", best.params)


if __name__ == "__main__":
    main()
