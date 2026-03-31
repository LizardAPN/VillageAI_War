"""Процедурные 3D-модели ландшафта и добываемых ресурсов для доски moderngl."""

from __future__ import annotations

import math

from village_ai_war.rendering.mesh_primitives import (
    add_cuboid,
    add_cylinder_y,
    add_pyramid,
    add_sphere,
)
from village_ai_war.state import ResourceLayer, TerrainType


def terrain_height(terrain_t: int) -> float:
    """Верх клетки по Y (как раньше в moderngl) — здания и юниты опираются на это значение."""
    return {
        int(TerrainType.GRASS): 0.14,
        int(TerrainType.MOUNTAIN): 0.62,
        int(TerrainType.FOREST): 0.22,
        int(TerrainType.STONE_DEPOSIT): 0.18,
        int(TerrainType.FIELD): 0.10,
    }.get(int(terrain_t), 0.12)


def _j(gx: int, gz: int, salt: int) -> float:
    """Детерминированный [0,1) для вариаций внутри клетки."""
    v = (gx * 374761393 + gz * 668265263 + salt * 1442695041) & 0x7FFFFFFF
    return v / 2147483647.0


def _tree(
    buf: list[float],
    tx: float,
    tz: float,
    y0: float,
    trunk_h: float,
    crown_r: float,
    trunk_rgb: tuple[float, float, float],
    leaf_rgb: tuple[float, float, float],
) -> None:
    add_cylinder_y(buf, tx, y0, tz, 0.038, trunk_h, 8, trunk_rgb)
    add_sphere(
        buf, tx, y0 + trunk_h + crown_r * 0.55, tz, crown_r, leaf_rgb, stacks=4, slices=8
    )


def add_terrain_cell(
    buf: list[float],
    wx: float,
    wz: float,
    terrain_t: int,
    gx: int,
    gz: int,
    cell: float = 0.92,
) -> None:
    """Один тайл карты: объёмный ландшафт под тип местности."""
    t = int(terrain_t)
    h = terrain_height(t)

    if t == int(TerrainType.GRASS):
        c0 = (0.14, 0.36, 0.24)
        c1 = (0.22, 0.50, 0.34)
        c2 = (0.28, 0.58, 0.38)
        add_cuboid(buf, wx, h * 0.38, wz, cell, h * 0.76, cell, c0)
        add_cuboid(buf, wx, h - 0.025, wz, cell * 0.94, 0.05, cell * 0.94, c1)
        for i in range(6):
            a = 2 * math.pi * i / 6 + _j(gx, gz, i) * 0.4
            rr = 0.22 + 0.12 * _j(gx, gz, i + 40)
            ox = wx + rr * math.cos(a)
            oz = wz + rr * math.sin(a)
            blade_h = 0.05 + 0.04 * _j(gx, gz, i + 80)
            add_cylinder_y(
                buf,
                ox,
                h * 0.88,
                oz,
                0.014 + 0.006 * _j(gx, gz, i + 10),
                blade_h,
                5,
                _mix3(c2, (0.1, 0.45, 0.2), _j(gx, gz, i + 20)),
            )

    elif t == int(TerrainType.FOREST):
        soil = (0.12, 0.28, 0.18)
        moss = (0.16, 0.38, 0.26)
        add_cuboid(buf, wx, h * 0.42, wz, cell, h * 0.84, cell, soil)
        add_cuboid(buf, wx, h - 0.03, wz, cell * 0.92, 0.06, cell * 0.92, moss)
        trunk = (0.38, 0.28, 0.18)
        leaf = (0.1, 0.42, 0.22)
        n_trees = 3 if _j(gx, gz, 1) > 0.25 else 2
        for i in range(n_trees):
            ang = 2 * math.pi * (i / max(n_trees, 1)) + _j(gx, gz, 50 + i) * 0.7
            rad = 0.18 + 0.14 * _j(gx, gz, 60 + i)
            tx = wx + rad * math.cos(ang)
            tz = wz + rad * math.sin(ang)
            th = 0.11 + 0.06 * _j(gx, gz, 70 + i)
            cr = 0.09 + 0.04 * _j(gx, gz, 90 + i)
            _tree(buf, tx, tz, h * 0.92, th, cr, trunk, leaf)

    elif t == int(TerrainType.MOUNTAIN):
        rock_lo = (0.32, 0.34, 0.38)
        rock_hi = (0.48, 0.50, 0.55)
        snow = (0.82, 0.86, 0.90)
        add_cuboid(buf, wx, h * 0.22, wz, cell * 0.95, h * 0.44, cell * 0.95, rock_lo)
        yb = h * 0.38
        add_pyramid(buf, wx, yb, wz, 0.40, h * 0.88, rock_hi)
        ox = wx + 0.22 * (1 if (gx + gz) % 2 == 0 else -1)
        oz = wz + 0.18 * (1 if gz % 3 else -1)
        add_pyramid(buf, ox, yb + h * 0.08, oz, 0.22, h * 0.72, _mix3(rock_hi, rock_lo, 0.4))
        add_sphere(buf, wx, h * 0.82, wz, 0.12, snow, stacks=3, slices=6)

    elif t == int(TerrainType.STONE_DEPOSIT):
        base = (0.42, 0.44, 0.48)
        rock = (0.55, 0.56, 0.62)
        add_cuboid(buf, wx, h * 0.45, wz, cell, h * 0.9, cell, base)
        for k in range(5):
            jr = 0.05 + 0.05 * _j(gx, gz, 100 + k)
            ja = 2 * math.pi * _j(gx, gz, 110 + k)
            ox = wx + (0.12 + 0.22 * _j(gx, gz, 120 + k)) * math.cos(ja)
            oz = wz + (0.12 + 0.22 * _j(gx, gz, 130 + k)) * math.sin(ja)
            add_sphere(buf, ox, h - jr * 0.3, oz, jr, rock, stacks=3, slices=6)
        add_cuboid(buf, wx + 0.18, h * 0.55, wz - 0.12, 0.16, h * 0.5, 0.14, _mix3(rock, base, 0.3))

    elif t == int(TerrainType.FIELD):
        earth = (0.52, 0.44, 0.28)
        crop = (0.72, 0.62, 0.32)
        add_cuboid(buf, wx, h * 0.48, wz, cell, h * 0.96, cell, earth)
        for row in range(4):
            zz = wz + (row - 1.5) * 0.18
            add_cuboid(buf, wx, h - 0.02, zz, cell * 0.88, 0.035, 0.06, crop)
        for i in range(8):
            sx = wx + (i - 3.5) * 0.1 + (_j(gx, gz, i) - 0.5) * 0.04
            add_cuboid(
                buf,
                sx,
                h + 0.04 * _j(gx, gz, 200 + i),
                wz + (_j(gx, gz, 210 + i) - 0.5) * 0.35,
                0.04,
                0.06 + 0.05 * _j(gx, gz, 220 + i),
                0.04,
                (0.58, 0.72, 0.28),
            )

    else:
        rgb = (0.32, 0.34, 0.36)
        add_cuboid(buf, wx, h * 0.5, wz, cell, h, cell, rgb)


def _mix3(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    t = min(1.0, max(0.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def add_resource_prop(
    buf: list[float],
    wx: float,
    wz: float,
    base_h: float,
    layer: int,
    amount: int,
    gx: int,
    gz: int,
) -> None:
    """Над поверхностью клетки (base_h): модель запаса леса / камня / поля."""
    if int(layer) == int(ResourceLayer.NONE) or amount <= 0:
        return
    sc = min(1.0, 0.2 + 0.8 * (amount / (amount + 30.0)))
    y = float(base_h) + 0.02

    if int(layer) == int(ResourceLayer.FOREST):
        wood = (0.55, 0.40, 0.26)
        wood2 = (0.42, 0.32, 0.22)
        for k in range(max(2, int(4 * sc))):
            ox = wx + (k - 1.5) * 0.12 * sc + (_j(gx, gz, 300 + k) - 0.5) * 0.06
            oz = wz + (_j(gx, gz, 310 + k) - 0.5) * 0.1
            ly = y + k * 0.045 * sc
            add_cuboid(
                buf,
                ox,
                ly + 0.03 * sc,
                oz,
                0.22 * sc,
                0.055 * sc,
                0.07 * sc,
                wood if k % 2 == 0 else wood2,
            )
        add_cuboid(buf, wx, y + 0.02, wz, 0.12 * sc, 0.04 * sc, 0.12 * sc, (0.22, 0.38, 0.18))

    elif int(layer) == int(ResourceLayer.STONE):
        ore = (0.62, 0.58, 0.72)
        ore_d = (0.42, 0.38, 0.52)
        cry = (0.75, 0.72, 0.88)
        add_sphere(buf, wx, y + 0.08 * sc, wz, 0.1 * sc, ore_d, stacks=4, slices=7)
        add_pyramid(buf, wx, y, wz, 0.12 * sc, y + 0.22 * sc, ore)
        add_pyramid(
            buf,
            wx - 0.1 * sc,
            y + 0.04,
            wz + 0.08 * sc,
            0.06 * sc,
            y + 0.18 * sc,
            cry,
        )
        add_pyramid(
            buf,
            wx + 0.11 * sc,
            y + 0.03,
            wz - 0.07 * sc,
            0.05 * sc,
            y + 0.16 * sc,
            _mix3(ore, cry, 0.5),
        )

    elif int(layer) == int(ResourceLayer.FIELD):
        gold = (0.88, 0.72, 0.22)
        gold2 = (0.72, 0.58, 0.18)
        n = max(3, int(7 * sc))
        for i in range(n):
            sx = wx + (i - n * 0.5 + 0.5) * 0.09 * sc
            sh = (0.1 + 0.12 * _j(gx, gz, 400 + i)) * sc
            add_cuboid(buf, sx, y + sh * 0.5, wz + (_j(gx, gz, 410 + i) - 0.5) * 0.12, 0.03, sh, 0.03, gold if i % 2 == 0 else gold2)
        add_cuboid(buf, wx, y - 0.01, wz, 0.35 * sc, 0.025, 0.28 * sc, (0.62, 0.52, 0.28))
