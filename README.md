# Village AI War

A **Gymnasium** environment for a small RTS-style village game. You can watch a quick demo, play against a trained **MAPPO** bot team, or train your own policies.

Training in this repo is **MAPPO only** (multi-agent PPO with a shared critic). Everything else is the game simulation, rendering, and helpers.

---

## Install and first run

```bash
cd VillageAI_War
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_game.py
```

That opens a **random demo**: the AI village manager picks valid actions at random (2D window). Add `--human-3d` for a passive 3D view (needs working OpenGL; on Linux/WSL you may need `libgl1` / Mesa — if 3D fails, the script falls back to 2D).

---

## Play against MAPPO

You control **blue**; **red** uses your checkpoint. Human play is **2D only** (MAPPO training matches this).

```bash
python scripts/run_game.py \
  --mappo-opponent checkpoints/bots_mappo/mappo_bot_final.zip \
  --seed 0 --max-steps 500
```

Use a checkpoint trained with the same **map size** and **`game.max_bots_for_role_change`** as in your config.

---

## Train MAPPO

```bash
python scripts/run_training.py
```

Quick test (small run):

```bash
python scripts/run_training.py \
  training.total_timesteps=2000 training.n_envs=1 training.selfplay_iterations=2
```

Settings live in [`configs/training/train_mappo_bots.yaml`](configs/training/train_mappo_bots.yaml) (merged via [`configs/default.yaml`](configs/default.yaml)). Override anything under `training.*` on the command line.

**TensorBoard:** logs go under `logs/mappo_bots/` when enabled. Then:

```bash
tensorboard --logdir logs/
```

**Plots:** `python scripts/plot_tensorboard_scalars.py --log-root logs` writes `logs/plots/mappo_bots_scalars.png` by default.

---

```bash
python scripts/evaluate.py
```

Runs many episodes with random village actions and prints rough stats.



