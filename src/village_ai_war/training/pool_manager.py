"""Fixed-size FIFO pool of policy checkpoint paths for self-play."""

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path


class PoolManager:
    """Keeps up to ``max_size`` checkpoints in ``pool_dir`` (oldest removed first).

    Args:
        pool_dir: Directory containing pooled ``.zip`` checkpoints.
        max_size: Maximum number of checkpoints to retain.
    """

    def __init__(self, pool_dir: Path, max_size: int = 10) -> None:
        self.pool_dir = pool_dir
        self.max_size = max_size
        self._pool: deque[Path] = deque()
        for ckpt in sorted(pool_dir.glob("*.zip")):
            self._pool.append(ckpt)

    def add(self, checkpoint_path: Path) -> None:
        """Copy checkpoint into the pool (if needed) and enforce capacity."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        dest = self.pool_dir / checkpoint_path.name
        if checkpoint_path.resolve() != dest.resolve():
            shutil.copy2(checkpoint_path, dest)

        dest = dest.resolve()
        if dest in self._pool:
            return
        self._pool.append(dest)
        while len(self._pool) > self.max_size:
            oldest = self._pool.popleft()
            if oldest.exists():
                oldest.unlink()

    def __len__(self) -> int:
        return len(self._pool)
