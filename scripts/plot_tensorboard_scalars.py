#!/usr/bin/env python3
"""Plot TensorBoard scalar runs (default: MAPPO logs/mappo_bots) to PNG grids."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from village_ai_war.training.tensorboard_plots import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
