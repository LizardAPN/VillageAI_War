# Village AI War — 2D Hierarchical RL Environment

Custom Gymnasium environment for hierarchical multi-agent reinforcement learning. The **baseline is fully RL-driven**: low-level units are controlled by a learned policy (no movement heuristics), and both teams can be trained with **self-play** against pools of past checkpoints.

## Architecture

- **Bot agents (low level):** One **role-conditioned** PPO policy (`RoleConditionedPolicy`) — shared backbone plus a learned role embedding from the observation one-hot (see `BotObsBuilder`). All roles share weights.
- **Village agent (high level):** Strategic manager — **MaskablePPO** with invalid-action masking (`MultiInputPolicy` on dict observations).
- **Self-play:** Stage 1 samples opponents from `checkpoints/pool/bots/`; stage 2 samples village opponents from `checkpoints/pool/village/`. Empty pools fall back to random opponent actions.
- **Reward shaping:** Dynamic global reward modes controlled by the village agent (unchanged).

## Quick Start

```bash
cd VillageAI_War
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_game.py
```

### Watching trained policies

[`scripts/run_game.py`](scripts/run_game.py) can load the same checkpoints produced by training. If a path is missing or fails to load, that component falls back to random valid actions (same as the old demo).

```bash
# Defaults: checkpoints/village/village_final.zip, checkpoints/bots/bot_final.zip
python scripts/run_game.py

# Explicit paths and deterministic manager actions
python scripts/run_game.py \
  --village-checkpoint checkpoints/village/village_final.zip \
  --opponent-village-checkpoint checkpoints/pool/village/village_iter5.zip \
  --bot-checkpoint checkpoints/bots/bot_final.zip \
  --deterministic --seed 42 --max-steps 2000
```

When any village, opponent, or bot checkpoint is loaded (or you pass a path that exists), the viewer runs **both** village managers each tick (trained or random per side), then resolves the tick in the same order as self-play training (`run_bots_then_village_decisions` in [`GameEnv`](src/village_ai_war/env/game_env.py)). If **no** checkpoints load, behavior matches the legacy script: one random manager action per step for team 0 only.

The pygame **human** window includes numeric **row/column axes**, a **legend** (terrain colors, harvest hints `w`/`s`/`f`, unit roles and team rings, building abbreviations), and a **bottom HUD** (tick, winner, resources, population, global mode per team). `rgb_array` mode is still the raw map only for headless frames.

### Training (Hydra)

The default config composes `training: train_bots_selfplay` (see [`configs/default.yaml`](configs/default.yaml)). Stages are selected with `training.stage`.

**Stage 1 — bot self-play (role-conditioned PPO)**

```bash
python scripts/run_training.py training.stage=1
```

Uses `game.initial_bots: 1` in [`configs/training/train_bots_selfplay.yaml`](configs/training/train_bots_selfplay.yaml) so each team has one controllable unit per episode (reproducible with `SubprocVecEnv`; multi-bot stage 1 would need extra machinery such as checkpoint sync or `DummyVecEnv`-only runs).

**Stage 2 — village self-play (MaskablePPO, frozen bot policy)**

Requires a stage-1 artifact at `checkpoints/bots/bot_final.zip`.

```bash
python scripts/run_training.py training=train_village_selfplay training.stage=2
```

**Stage 3 — joint fine-tuning**

Loads `checkpoints/village/village_final.zip` when present; bots use `checkpoints/bots/bot_final.zip` when present (otherwise random bot actions).

```bash
python scripts/run_training.py training=train_joint training.stage=3
```

**Useful overrides**

```bash
python scripts/run_training.py training.total_timesteps=2000 training.n_envs=1 logging.use_wandb=false
```

### Metrics (TensorBoard / W&B)

With `logging.use_tensorboard: true` (default) and `tensorboard` installed, stage 1 and 2 write training scalars under `logs/bots/` and `logs/village/`. Periodic evaluation logs `eval/mean_reward` (and related fields) under `logs/bots_eval/` and `logs/village_eval/` when `training.eval_freq > 0` (default `10000` **environment timesteps** between evals; internally scaled by `n_envs` per Stable-Baselines3). Tune with `training.n_eval_episodes`.

```bash
tensorboard --logdir logs/
```

If `logging.use_wandb` is on, `wandb.init` uses `sync_tensorboard=True` when TensorBoard is available so the same scalars appear in W&B.

**Best vs last checkpoint:** when evaluation is enabled and at least one eval run produced a best model, `checkpoints/bots/bot_final.zip` and `checkpoints/village/village_final.zip` are copies of the best eval checkpoint (also saved as `bot_best.zip` / `village_best.zip`). The last weights after all self-play iterations are kept as `bot_last.zip` / `village_last.zip`. Set `training.eval_freq=0` to keep the previous behavior (final = last iteration only). Stage 3 joint training does not add a separate eval pass yet; it still saves `checkpoints/joint/joint_final.zip` from the end of the run.

**Evaluation**

```bash
python scripts/evaluate.py
```

## Checkpoints (typical layout)

| Path | Contents |
|------|-----------|
| `checkpoints/bots/bot_final.zip` | Stage 1 policy (best eval mean reward when `training.eval_freq > 0`, else last iteration) |
| `checkpoints/pool/bots/*.zip` | Historical bot policies for self-play |
| `checkpoints/village/village_final.zip` | Stage 2 manager (same best-vs-last rule as bots) |
| `checkpoints/pool/village/*.zip` | Historical village policies for self-play |
| `checkpoints/joint/joint_final.zip` | Stage 3 output |

## Project layout (RL baseline)

- [`src/village_ai_war/env/game_env.py`](src/village_ai_war/env/game_env.py) — `step`, `step_with_opponent`, `step_village_only`, optional `game.bot_rl_checkpoint` / `training.bot_checkpoint` for frozen bot PPO
- [`src/village_ai_war/models/role_conditioned_policy.py`](src/village_ai_war/models/role_conditioned_policy.py)
- [`src/village_ai_war/training/self_play_env.py`](src/village_ai_war/training/self_play_env.py) — `SelfPlayBotEnv`, `SelfPlayVillageEnv`
- [`src/village_ai_war/training/train_bots_selfplay.py`](src/village_ai_war/training/train_bots_selfplay.py), [`train_village_selfplay.py`](src/village_ai_war/training/train_village_selfplay.py), [`train_joint.py`](src/village_ai_war/training/train_joint.py)

Legacy trainers [`train_bots.py`](src/village_ai_war/training/train_bots.py) and [`train_village.py`](src/village_ai_war/training/train_village.py) are not used by [`scripts/run_training.py`](scripts/run_training.py).

## Project status

- [x] Data structures (Pydantic)
- [x] Map generator
- [x] Economy / combat / building systems
- [x] Observation builders and action masker
- [x] GameEnv (Gymnasium), no bot heuristics
- [x] Role-conditioned bot policy + PPO self-play (stage 1)
- [x] Village MaskablePPO self-play with RL bots (stage 2)
- [x] Joint fine-tuning with RL bots (stage 3)
- [x] Pygame renderer
