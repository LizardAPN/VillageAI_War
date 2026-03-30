"""Tests for training pool manager."""

from __future__ import annotations

from pathlib import Path

from village_ai_war.training.pool_manager import PoolManager


def test_pool_manager_max_size(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pm = PoolManager(pool_dir, max_size=10)
    for i in range(15):
        f = pool_dir / f"ckpt_{i}.zip"
        f.write_bytes(b"x")
        pm.add(f)
    assert len(pm) == 10
    remaining = sorted(pool_dir.glob("*.zip"))
    assert len(remaining) == 10
    # Oldest five (0-4) should be gone
    names = {p.name for p in remaining}
    for i in range(5):
        assert f"ckpt_{i}.zip" not in names
    for i in range(5, 15):
        assert f"ckpt_{i}.zip" in names
