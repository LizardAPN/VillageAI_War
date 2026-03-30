# Village AI War — 2D Hierarchical RL Environment

Custom Gymnasium environment for hierarchical multi-agent reinforcement learning.

## Architecture

- **Bot Agents (Low Level):** Warriors, Gatherers, Farmers, Builders — PPO
- **Village Agent (High Level):** Strategic manager — MaskablePPO
- **Reward Shaping:** Dynamic global reward modes controlled by village agent

## Quick Start

```bash
cd VillageAI_War
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/run_game.py
python scripts/run_training.py training.stage=1
python scripts/run_training.py training.stage=2
python scripts/evaluate.py
```

Training entrypoint uses Hydra; override configs from the CLI (for example ``training.total_timesteps=50000``). Stage 2/3 configs can be composed via ``--config-name=training/train_village``.

## Training Stages

- **Stage 1:** Train bot policies in isolation
- **Stage 2:** Train village manager with frozen bots
- **Stage 3:** Joint fine-tuning

## Project Status

- [x] Data structures (Pydantic)
- [x] Map generator
- [x] Economy system
- [x] Combat system
- [x] Building system
- [x] Observation builders
- [x] Reward calculators
- [x] Action masker
- [x] GameEnv (Gymnasium)
- [x] Pygame renderer
- [x] Training scripts
