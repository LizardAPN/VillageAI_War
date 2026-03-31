# Village AI War — 2D Hierarchical RL Environment

Custom Gymnasium environment for hierarchical multi-agent reinforcement learning. The **baseline is fully RL-driven**: low-level units are controlled by a learned policy (no movement heuristics), and both teams can be trained with **self-play** against pools of past checkpoints.

### Research baselines (bots)

| | Baseline v0 (legacy stage 1) | Baseline v1 (MAPPO, stage 4) |
|---|------------------------------|--------------------------------|
| Algorithm | PPO + `RoleConditionedPolicy` (shared actor–critic on local obs) | PPO + `MAPPOPolicy`: decentralized actor, **centralized critic** on global map + both villages |
| Coordination | Each bot optimizes its own value signal | Shared critic sees full state (CTDE-style training) |
| Observation | `BotObsBuilder` only (181-dim) | **K** stacked local bot vectors (`K = game.max_bots_for_role_change`) + flattened global map (team-0 POV) + both village vectors (see `MAPPOBotEnv`) |

**Baseline v0 (failed for the full game):** villages often **starved before the economy phase** because local critics did not coordinate gatherers/farmers; other issues included reward plumbing and map scale. That stack remains available as **stage 1** for ablations, but **v1 (MAPPO)** is the recommended bot baseline for comparing new architectures.

## Architecture

- **Bot agents (low level):** Default legacy path uses one **role-conditioned** PPO policy (`RoleConditionedPolicy`) — shared backbone plus a learned role embedding from the observation one-hot (see `BotObsBuilder`). All roles share weights. **MAPPO path (stage 4):** `MAPPOPolicy` with `MAPPOActorExtractor` over **K local slots** (shared trunk per slot) and `MAPPOCentralizedCritic` on the global tail. One RL step = one game tick with **all allied bots** acting (`MultiDiscrete` of size **K**); opponents still act every tick via the pool policy or random. The self-play pool loads **181-dim** PPO checkpoints for the opponent; the trained MAPPO policy expects the **extended** observation from `MAPPOBotEnv`. **Older MAPPO checkpoints** from the single-bot-per-tick layout are **not** compatible with the current stacked-obs + `MultiDiscrete` policy.
- **Village agent (high level):** Strategic manager — **MaskablePPO** with invalid-action masking (`MultiInputPolicy` on dict observations).
- **Self-play:** Stages 1 and 4 sample bot opponents from `checkpoints/pool/bots/`; stage 2 samples village opponents from `checkpoints/pool/village/`. Empty pools fall back to random opponent actions.
- **Unified training:** One Hydra entry point (`training=train_unified`, `training.stage=0`) alternates PPO on bots and MaskablePPO on the red manager. Each environment step matches village self-play order (all bots act, then both managers). Blue bots and blue manager are sampled from the same pools; the non-training partner on red is frozen from the last saved checkpoint until the next phase.
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

# Artifacts from unified training
python scripts/run_game.py \
  --village-checkpoint checkpoints/unified/village_final.zip \
  --bot-checkpoint checkpoints/unified/bot_final.zip \
  --deterministic --seed 42
```

### Human vs AI

**Against a trained MAPPO policy (micro only, matches `MAPPOBotEnv` training):** you play **BLUE**; **RED** bots are controlled by a `MAPPOPolicy` checkpoint. There is **no** village manager phase in this mode (same as stage-4 bot training). Requires a **2D pygame** window.

```bash
python scripts/run_game.py \
  --mappo-opponent checkpoints/bots_mappo/mappo_bot_final.zip \
  --seed 0 --max-steps 500
```

The checkpoint must match your config (`map.size`, `game.max_bots_for_role_change`). `--human red` is rejected (MAPPO is trained as team 0 with a fixed observation layout).

**Against MaskablePPO (village) + 181-dim PPO (bots):** use `--human red` or `--human blue`. Each tick you choose actions for **all** alive bots on your team, then (on manager ticks) a village action via `[` / `]` and Enter. The other side uses loaded checkpoints or random valid actions. Human play is **2D only** (`--human-3d` is ignored).

```bash
python scripts/run_game.py --human blue \
  --village-checkpoint checkpoints/village/village_final.zip \
  --bot-checkpoint checkpoints/bots/bot_final.zip
```

When any village, opponent, or bot checkpoint is loaded (or you pass a path that exists), the viewer runs **both** village managers each tick (trained or random per side), then resolves the tick in the same order as self-play training (`run_bots_then_village_decisions` in [`GameEnv`](src/village_ai_war/env/game_env.py)). If **no** checkpoints load, behavior matches the legacy script: one random manager action per step for team 0 only.

The pygame **human** window includes numeric **row/column axes**, a **legend** (terrain colors, harvest hints `w`/`s`/`f`, unit roles and team rings, building abbreviations), and a **bottom HUD** (tick, winner, resources, population, global mode per team). `rgb_array` mode is still the raw map only for headless frames.

**3D view** (OpenGL via [moderngl](https://github.com/moderngl/moderngl)): run `python scripts/run_game.py --human-3d` for an extruded terrain board, **building silhouettes** (town hall roof, tower spire, farm silo, etc.), **bots as team disk + role sphere + team ring**, and a **Russian legend panel** on the right (including a **live HUD**: tick, wood/stone/food, alive bots / pop cap, village AI mode per team). **Camera:** left-drag on the map to orbit (yaw/pitch); arrow keys nudge the camera; optional idle spin via `auto_rotate_deg_per_sec` (default `0`). Tuning: `window_width_3d`, `legend_width_3d`, `camera_dist_scale`, `orbit_mouse_sensitivity`, `orbit_key_deg_per_sec`, … in [`configs/default.yaml`](configs/default.yaml). In code, use `GameEnv(..., render_mode="human_3d")` or `render_mode="rgb_array_3d"` for full-window RGB captures (map + legend).

On **Linux**, Mesa packages expose `libGL.so.1` (not `libGL.so`); this project patches library loading so moderngl finds them. If you still see GL errors, install dri drivers: `sudo apt install -y libgl1-mesa-dri libegl1`. On **WSL2** you need a working GUI stack (WSLg); without it, use 2D or run from native Windows. If 3D fails, `run_game.py` falls back to the 2D pygame window and logs a warning.

### Training (Hydra)

The default config composes `training: train_bots_selfplay` (see [`configs/default.yaml`](configs/default.yaml)). Stages are selected with `training.stage` (`0` = unified, `1`–`3` = legacy pipeline, `4` = MAPPO bots).

**Stage 4 — MAPPO bot self-play (Baseline v1)**

Centralized critic + parameter sharing: **all** learner-team bots move each tick (`MAPPOBotEnv`, `MultiDiscrete` length `game.max_bots_for_role_change`). Checkpoints under `checkpoints/bots_mappo/`; self-play **snapshots** go to `checkpoints/pool/bots_mappo/mappo_bot_iter*.zip` (extended obs). **Opponents** are sampled only from `checkpoints/pool/bots/*.zip` and must be **181-dim** (stage 1 / unified bot). Uses **TensorBoard** only (`logging.use_tensorboard` in [`configs/training/train_mappo_bots.yaml`](configs/training/train_mappo_bots.yaml); scalars under `logs/mappo_bots/`). Episode diagnostics: rolling fractions under `mappo/outcome_frac/*` (win / loss / draw / truncated) and `mappo/terminal_reason_frac/*` (e.g. `townhall_destroyed`, `stagnation`, `max_ticks`), controlled by `training.mappo_metrics_window`.

Files named `mappo_bot*.zip` under `pool/bots/` are **ignored** (treat as misplaced MAPPO saves). If there is no usable 181-dim checkpoint, only **SubprocVecEnv worker 0** logs a single WARNING (other workers stay silent). Add at least one stage-1-style bot zip in `pool/bots/` for real self-play.

```bash
python scripts/run_training.py training=train_mappo_bots
```

Smoke run:

```bash
python scripts/run_training.py training=train_mappo_bots \
  training.total_timesteps=2000 training.n_envs=1 training.selfplay_iterations=2
```

**Stage 1 — bot self-play (role-conditioned PPO, Baseline v0)**

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

**Unified training (recommended)** — `training.stage=0`, single process alternating two learners

No prerequisite checkpoints. Team 0’s bot policy (PPO) and red manager (MaskablePPO) are updated in turns; after each phase the trainer saves so the other phase loads a frozen partner from disk. Opponents use the same self-play pools as stages 1–2; empty pools still fall back to random valid actions.

```bash
python scripts/run_training.py training=train_unified
```

You can also set `training.stage=0` with another `training=` group; built-in defaults for `unified.*` still apply, but prefer `training=train_unified` so values in [`configs/training/train_unified.yaml`](configs/training/train_unified.yaml) are loaded.

Configure macro steps with `unified.bot_steps_per_turn`, `unified.village_steps_per_turn`, `unified.n_cycles`, optional `unified.first_phase` (`bot` or `village`), and `unified.push_to_pool` (append snapshots to `checkpoints/pool/bots` and `.../village`) in [`configs/training/train_unified.yaml`](configs/training/train_unified.yaml). Each phase logs an estimated **SB3 iteration count** (`n_steps × n_envs` env-steps per iteration). `unified.progress_bar: true` enables Stable-Baselines’ tqdm progress bar for the current `learn()` (requires `tqdm` and `rich` in [`requirements.txt`](requirements.txt)). `unified.sb3_verbose` controls SB3’s stdout tables (`0` by default when using the progress bar). `unified.progress_log_interval_sec` throttles **Loguru** lines with progress, ETA, env-steps/s, and wall time of the previous full SB3 iteration (see `run_training.log` under the Hydra run directory). Set `unified.plot_metrics_on_finish: true` to write PNG grids of TensorBoard scalars into the same Hydra output folder when the run finishes (or run [`scripts/plot_tensorboard_scalars.py`](scripts/plot_tensorboard_scalars.py) manually). Outputs live under `checkpoints/unified/` (`bot_final.zip`, `village_final.zip`, plus per-cycle checkpoints and `bot_latest` / `village_latest` stems used between phases).

The bot phase uses `DummyVecEnv` only so every sub-env shares the in-process `bot_policy_holder`; do not use `SubprocVecEnv` for that phase. Default `game.initial_bots: 1` matches stage 1; more red bots require a live model in the holder for the extra units.

Stages 1–3 remain a supported alternative pipeline; stage 4 is the MAPPO baseline for multi-bot coordination research.

**Useful overrides**

```bash
python scripts/run_training.py training.total_timesteps=2000 training.n_envs=1
```

### Metrics (TensorBoard)

With `logging.use_tensorboard: true` (default) and `tensorboard` installed, stage 1 and 2 write training scalars under `logs/bots/` and `logs/village/`; stage 4 (MAPPO) under `logs/mappo_bots/`; unified training writes under `logs/unified_bots/` and `logs/unified_village/`. Periodic evaluation logs `eval/mean_reward` (and related fields) under `logs/bots_eval/` and `logs/village_eval/` when `training.eval_freq > 0` (default `10000` **environment timesteps** between evals; internally scaled by `n_envs` per Stable-Baselines3). The unified config sets `eval_freq: 0` by default (no separate eval pass yet). Tune with `training.n_eval_episodes`.

```bash
tensorboard --logdir logs/
```

After unified training (or anytime), generate static PNG grids from the latest run in each subfolder:

```bash
python scripts/plot_tensorboard_scalars.py --log-root logs
```

Default outputs: `logs/plots/unified_bots_scalars.png` and `logs/plots/unified_village_scalars.png` (or under the Hydra run directory when `unified.plot_metrics_on_finish` is enabled).

**Best vs last checkpoint:** when evaluation is enabled and at least one eval run produced a best model, `checkpoints/bots/bot_final.zip` and `checkpoints/village/village_final.zip` are copies of the best eval checkpoint (also saved as `bot_best.zip` / `village_best.zip`). The last weights after all self-play iterations are kept as `bot_last.zip` / `village_last.zip`. Set `training.eval_freq=0` to keep the previous behavior (final = last iteration only). Stage 3 joint training does not add a separate eval pass yet; it still saves `checkpoints/joint/joint_final.zip` from the end of the run. Unified training always ends with `bot_final.zip` / `village_final.zip` as the last full save after all cycles (no best-model selection unless you add eval later).

**Evaluation**

```bash
python scripts/evaluate.py
```

## Checkpoints (typical layout)

| Path | Contents |
|------|-----------|
| `checkpoints/bots/bot_final.zip` | Stage 1 policy (best eval mean reward when `training.eval_freq > 0`, else last iteration) |
| `checkpoints/bots_mappo/mappo_bot_final.zip` | Stage 4 MAPPO policy (last save after all self-play iterations) |
| `checkpoints/pool/bots/*.zip` | 181-dim bot policies (opponents for MAPPO and stage 1) |
| `checkpoints/pool/bots_mappo/*.zip` | MAPPO training snapshots (extended obs; not used as opponents) |
| `checkpoints/village/village_final.zip` | Stage 2 manager (same best-vs-last rule as bots) |
| `checkpoints/pool/village/*.zip` | Historical village policies for self-play |
| `checkpoints/joint/joint_final.zip` | Stage 3 output |
| `checkpoints/unified/bot_final.zip` | Unified loop bot policy (last save after all cycles) |
| `checkpoints/unified/village_final.zip` | Unified loop village policy (last save after all cycles) |
| `checkpoints/unified/bot_latest.zip` / `village_latest.zip` | Latest weights exchanged between alternating phases |
| `checkpoints/unified/bot_cycle*.zip` / `village_cycle*.zip` | Periodic `CheckpointCallback` snapshots during unified runs |

## Project layout (RL baseline)

- [`src/village_ai_war/env/game_env.py`](src/village_ai_war/env/game_env.py) — `step`, `step_with_opponent`, `step_village_only`, `queue_bot_action`, `_simulation_tick` (MAPPO tick path), optional `game.bot_rl_checkpoint` / `training.bot_checkpoint` for frozen bot PPO
- [`src/village_ai_war/agents/village_obs_builder.py`](src/village_ai_war/agents/village_obs_builder.py) — `build_map`, `build_village_vec` (global critic / MAPPO)
- [`src/village_ai_war/models/role_conditioned_policy.py`](src/village_ai_war/models/role_conditioned_policy.py)
- [`src/village_ai_war/models/mappo_actor.py`](src/village_ai_war/models/mappo_actor.py), [`mappo_critic.py`](src/village_ai_war/models/mappo_critic.py), [`mappo_policy.py`](src/village_ai_war/models/mappo_policy.py), [`mappo_layout.py`](src/village_ai_war/models/mappo_layout.py)
- [`src/village_ai_war/training/self_play_env.py`](src/village_ai_war/training/self_play_env.py) — `SelfPlayBotEnv`, `SelfPlayVillageEnv`, `UnifiedBotSelfPlayEnv`
- [`src/village_ai_war/training/mappo_env.py`](src/village_ai_war/training/mappo_env.py) — `MAPPOBotEnv`; [`global_state_callback.py`](src/village_ai_war/training/global_state_callback.py) (optional diagnostics from `info`)
- [`src/village_ai_war/training/train_bots_selfplay.py`](src/village_ai_war/training/train_bots_selfplay.py), [`train_mappo_bots.py`](src/village_ai_war/training/train_mappo_bots.py), [`train_village_selfplay.py`](src/village_ai_war/training/train_village_selfplay.py), [`train_joint.py`](src/village_ai_war/training/train_joint.py), [`train_unified.py`](src/village_ai_war/training/train_unified.py), [`tensorboard_plots.py`](src/village_ai_war/training/tensorboard_plots.py), [`plot_tensorboard_scalars.py`](scripts/plot_tensorboard_scalars.py)

Legacy trainers [`train_bots.py`](src/village_ai_war/training/train_bots.py) and [`train_village.py`](src/village_ai_war/training/train_village.py) are not used by [`scripts/run_training.py`](scripts/run_training.py).

## Project status

- [x] Data structures (Pydantic)
- [x] Map generator
- [x] Economy / combat / building systems
- [x] Observation builders and action masker
- [x] GameEnv (Gymnasium), no bot heuristics
- [x] Role-conditioned bot policy + PPO self-play (stage 1, Baseline v0)
- [x] MAPPO bot baseline — centralized critic, concat global obs, stage 4 (`train_mappo_bots`)
- [x] Village MaskablePPO self-play with RL bots (stage 2)
- [x] Joint fine-tuning with RL bots (stage 3)
- [x] Unified training loop (alternating bot PPO + village MaskablePPO)
- [x] Pygame renderer
