"""Detailed procedural building meshes for the 3D board (team-tinted stone/wood)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from village_ai_war.rendering.mesh_primitives import (
    add_cuboid,
    add_cylinder_y,
    add_prism_y,
    add_pyramid,
    add_quad,
    add_sphere,
    add_tri,
)
from village_ai_war.state import BuildingType


def _mix(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    t = min(1.0, max(0.0, t))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _stone(team: tuple[float, float, float]) -> tuple[float, float, float]:
    base = (0.52, 0.54, 0.58)
    return _mix(base, team, 0.22)


def _wood() -> tuple[float, float, float]:
    return (0.62, 0.46, 0.30)


def _wood_dark() -> tuple[float, float, float]:
    return (0.42, 0.30, 0.20)


def _window() -> tuple[float, float, float]:
    return (0.18, 0.22, 0.32)


def _roof_tile(team: tuple[float, float, float]) -> tuple[float, float, float]:
    return (min(team[0] * 0.55 + 0.25, 0.95), min(team[1] * 0.55 + 0.22, 0.9), min(team[2] * 0.6 + 0.2, 0.92))


def _trim_gold(team: tuple[float, float, float]) -> tuple[float, float, float]:
    return _mix((0.92, 0.78, 0.42), team, 0.15)


def _add_merlons_ring(
    buf: list[float],
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    n: int,
    w: float,
    h: float,
    rgb: tuple[float, float, float],
) -> None:
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        ox = float(cx + radius * math.cos(a))
        oz = float(cz + radius * math.sin(a))
        add_cuboid(buf, ox, cy + h * 0.5, oz, w, h, w, rgb)


def _pitched_roof_along_z(
    buf: list[float],
    cx: float,
    cz0: float,
    cz1: float,
    y0: float,
    half_w: float,
    ridge_h: float,
    rgb: tuple[float, float, float],
) -> None:
    """Двускатная крыша: конёк вдоль Z на x=cx, карнизы на x=cx±half_w."""
    r, g, b = rgb
    ridge_y = y0 + ridge_h
    # Левый скат (-X)
    p0 = (cx - half_w, y0, cz0)
    p1 = (cx - half_w, y0, cz1)
    p2 = (cx, ridge_y, cz1)
    p3 = (cx, ridge_y, cz0)
    ex = p1[0] - p0[0]
    ey = p1[1] - p0[1]
    ez = p1[2] - p0[2]
    fx = p3[0] - p0[0]
    fy = p3[1] - p0[1]
    fz = p3[2] - p0[2]
    nx = ey * fz - ez * fy
    ny = ez * fx - ex * fz
    nz = ex * fy - ey * fx
    ln = float(np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
    nl = (nx / ln, ny / ln, nz / ln)
    add_quad(buf, p0, p1, p2, p3, nl, (r * 0.9, g * 0.9, b * 0.9))
    # Правый скат (+X)
    q0 = (cx + half_w, y0, cz0)
    q1 = (cx + half_w, y0, cz1)
    q2 = (cx, ridge_y, cz1)
    q3 = (cx, ridge_y, cz0)
    ex2 = q1[0] - q0[0]
    ey2 = q1[1] - q0[1]
    ez2 = q1[2] - q0[2]
    fx2 = q3[0] - q0[0]
    fy2 = q3[1] - q0[1]
    fz2 = q3[2] - q0[2]
    nx2 = ey2 * fz2 - ez2 * fy2
    ny2 = ez2 * fx2 - ex2 * fz2
    nz2 = ex2 * fy2 - ey2 * fx2
    ln2 = float(np.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2) + 1e-8)
    nr = (nx2 / ln2, ny2 / ln2, nz2 / ln2)
    add_quad(buf, q1, q0, q3, q2, nr, (r * 0.82, g * 0.82, b * 0.82))
    # Фронтон z = cz1
    add_tri(
        buf,
        (cx - half_w, y0, cz1),
        (cx + half_w, y0, cz1),
        (cx, ridge_y, cz1),
        (0.0, 0.0, 1.0),
        (r * 0.88, g * 0.88, b * 0.88),
    )
    # Фронтон z = cz0
    add_tri(
        buf,
        (cx + half_w, y0, cz0),
        (cx - half_w, y0, cz0),
        (cx, ridge_y, cz0),
        (0.0, 0.0, -1.0),
        (r * 0.78, g * 0.78, b * 0.78),
    )


def add_building_variant(
    buf: list[float],
    wx: float,
    wz: float,
    base_h: float,
    b: Any,
    tr: float,
    tg: float,
    tb: float,
) -> None:
    hp_f = b.hp / max(b.max_hp, 1)
    team = (tr, tg, tb)
    roof = (min(tr * 1.12, 0.98), min(tg * 1.12, 0.98), min(tb * 1.12, 0.98))
    st = _stone(team)
    bt = b.building_type
    y = base_h

    if bt == BuildingType.TOWNHALL:
        h_main = 0.38 + 0.42 * hp_f
        # Ступенчатый цоколь
        add_cuboid(buf, wx, y + 0.04, wz, 0.92, 0.08, 0.92, _mix(st, (0.35, 0.36, 0.4), 0.5))
        y += 0.08
        add_prism_y(buf, wx, y, wz, 0.40, h_main, 10, st, cap_bottom=False, cap_top=True)
        y += h_main
        # Второй ярус — уже
        add_prism_y(buf, wx, y, wz, 0.30, 0.18, 10, _mix(team, st, 0.35), cap_bottom=False, cap_top=True)
        y += 0.18
        # Колонны по углам вписанного квадрата
        col_y = base_h + 0.08 + h_main * 0.35
        col_h = h_main * 0.55
        off = 0.34
        for dx, dz in ((-off, -off), (off, -off), (-off, off), (off, off)):
            add_cylinder_y(buf, wx + dx, col_y, wz + dz, 0.045, col_h, 10, _trim_gold(team))
        # Купол
        add_sphere(buf, wx, y + 0.14, wz, 0.20, roof, stacks=6, slices=14)
        add_cylinder_y(buf, wx, y, wz, 0.22, 0.06, 16, _trim_gold(team))
        # Флагшток и полотнище
        pole_y = y + 0.28
        add_cylinder_y(buf, wx + 0.14, pole_y, wz - 0.06, 0.014, 0.26, 6, (0.55, 0.52, 0.5))
        fy = pole_y + 0.22
        add_quad(
            buf,
            (wx + 0.16, fy - 0.08, wz - 0.04),
            (wx + 0.16, fy + 0.02, wz - 0.04),
            (wx + 0.28, fy + 0.02, wz - 0.04),
            (wx + 0.28, fy - 0.08, wz - 0.04),
            (0.0, 0.0, 1.0),
            team,
        )

    elif bt == BuildingType.TOWER:
        h0, h1, h2 = 0.18 + 0.12 * hp_f, 0.28 + 0.22 * hp_f, 0.16 + 0.1 * hp_f
        r0, r1, r2 = 0.24, 0.19, 0.14
        add_cylinder_y(buf, wx, y, wz, r0 + 0.02, 0.06, 12, _mix(st, (0.4, 0.4, 0.44), 0.6))
        y += 0.05
        add_cylinder_y(buf, wx, y, wz, r0, h0, 12, st)
        y += h0
        add_cylinder_y(buf, wx, y, wz, r1, h1, 12, _mix(st, team, 0.12))
        y += h1
        add_cylinder_y(buf, wx, y, wz, r2, h2, 10, st)
        y += h2
        _add_merlons_ring(buf, wx, y, wz, r2 - 0.02, 8, 0.07, 0.08, st)
        y += 0.08
        add_pyramid(buf, wx, y, wz, 0.11, y + 0.38, _roof_tile(team))
        # Стрельницы — малые выступы
        for i in range(4):
            a = math.pi * 0.25 + i * math.pi * 0.5
            ox = wx + (r1 - 0.02) * math.cos(a)
            oz = wz + (r1 - 0.02) * math.sin(a)
            add_cuboid(buf, ox, base_h + h0 * 0.5, oz, 0.1, h0 * 0.6, 0.1, _mix(st, _window(), 0.25))

    elif bt == BuildingType.BARRACKS:
        h = 0.34 + 0.38 * hp_f
        hw, hd = 0.44, 0.34
        add_cuboid(buf, wx, y + 0.035, wz, hw * 2 + 0.08, 0.07, hd * 2 + 0.08, _mix(st, (0.4, 0.42, 0.45), 0.55))
        y += 0.07
        add_cuboid(buf, wx, y + h * 0.5, wz, hw * 2, h, hd * 2, _mix(team, st, 0.18))
        # Окна (тёмные вставки на фасаде +Z)
        win_y = y + h * 0.55
        for ox in (-0.28, 0.0, 0.28):
            add_cuboid(buf, wx + ox, win_y, wz + hd - 0.02, 0.1, 0.12, 0.04, _window())
        add_cuboid(buf, wx, y + h * 0.22, wz + hd - 0.02, 0.14, 0.2, 0.05, _wood_dark())
        # Двускатная крыша
        cz0, cz1 = wz - hd, wz + hd
        _pitched_roof_along_z(buf, wx, cz0, cz1, y + h, hw, 0.22, _roof_tile(team))
        # Труба
        add_cylinder_y(
            buf,
            wx + hw * 0.45,
            y + h + 0.12,
            wz - hd * 0.3,
            0.05,
            0.16,
            8,
            (0.45, 0.44, 0.48),
        )

    elif bt == BuildingType.FARM:
        h_wall = 0.16 + 0.1 * hp_f
        hw_b, hd_b = 0.40, 0.32
        add_cuboid(buf, wx, y + h_wall * 0.5, wz, hw_b * 2, h_wall, hd_b * 2, _wood_dark())
        y2 = y + h_wall
        _pitched_roof_along_z(buf, wx, wz - hd_b, wz + hd_b, y2, hw_b, 0.2, _roof_tile(team))
        # Силос
        silo_x = wx + 0.34
        add_cylinder_y(buf, silo_x, y, wz + 0.08, 0.11, 0.52 + 0.15 * hp_f, 14, (0.78, 0.76, 0.72))
        sy = y + 0.52 + 0.15 * hp_f
        add_pyramid(buf, silo_x, sy, wz + 0.08, 0.09, sy + 0.16, (0.72, 0.58, 0.38))
        # Стог / навес
        add_cuboid(buf, wx - 0.32, y + 0.08, wz - 0.22, 0.22, 0.12, 0.2, (0.82, 0.72, 0.28))
        add_cuboid(buf, wx - 0.32, y + 0.18, wz - 0.22, 0.24, 0.03, 0.22, _wood())

    elif bt == BuildingType.WALL:
        h = 0.16 + 0.22 * hp_f
        add_cuboid(buf, wx, y + h * 0.5, wz, 0.9, h, 0.2, st)
        top = y + h
        n_m = 5
        for i in range(n_m):
            t = (i + 0.5) / n_m - 0.5
            mx = wx + t * 0.72
            if i % 2 == 0:
                add_cuboid(buf, mx, top + 0.06, wz, 0.12, 0.12, 0.16, _mix(st, team, 0.08))
            else:
                add_cuboid(buf, mx, top + 0.03, wz, 0.14, 0.06, 0.18, _mix(st, (0.45, 0.46, 0.5), 0.4))

    elif bt == BuildingType.CITADEL:
        h_base = 0.36 + 0.35 * hp_f
        add_prism_y(buf, wx, y, wz, 0.39, h_base, 8, st, cap_bottom=False, cap_top=True)
        y += h_base
        add_cuboid(buf, wx, y + 0.16, wz, 0.52, 0.32, 0.52, _mix(team, st, 0.2))
        add_prism_y(buf, wx, y + 0.32, wz, 0.22, 0.2, 8, _roof_tile(team), cap_bottom=False, cap_top=True)
        for i in range(4):
            a = math.pi * 0.25 + i * math.pi * 0.5
            ox = wx + 0.33 * math.cos(a)
            oz = wz + 0.33 * math.sin(a)
            add_cylinder_y(buf, ox, base_h, oz, 0.08, h_base + 0.1, 10, st)
            add_pyramid(buf, ox, base_h + h_base + 0.1, oz, 0.06, base_h + h_base + 0.28, roof)
        add_cuboid(buf, wx, y + 0.34, wz, 0.56, 0.08, 0.56, _trim_gold(team))

    elif bt == BuildingType.STORAGE:
        h = 0.26 + 0.32 * hp_f
        add_cuboid(buf, wx, y + 0.04, wz, 0.88, 0.08, 0.76, _mix(st, (0.48, 0.5, 0.54), 0.45))
        y += 0.08
        add_cuboid(buf, wx, y + h * 0.5, wz, 0.82, h, 0.70, _mix(st, team, 0.12))
        # Навес погрузки
        add_cuboid(buf, wx + 0.36, y + h * 0.72, wz + 0.38, 0.34, 0.06, 0.42, _wood())
        add_cuboid(buf, wx + 0.36, y + h * 0.45, wz + 0.52, 0.06, h * 0.5, 0.06, _wood_dark())
        add_cuboid(buf, wx + 0.36, y + h * 0.45, wz + 0.24, 0.06, h * 0.5, 0.06, _wood_dark())
        add_cuboid(buf, wx - 0.12, y + h * 0.35, wz - 0.18, 0.22, 0.18, 0.2, _wood_dark())
        add_cuboid(buf, wx - 0.22, y + h * 0.2, wz + 0.2, 0.16, 0.14, 0.16, (0.55, 0.5, 0.42))
        add_cuboid(buf, wx - 0.28, y + h * 0.52, wz - 0.22, 0.2, 0.12, 0.18, (0.58, 0.52, 0.4))

    else:
        h = 0.35 + 0.45 * hp_f
        add_cuboid(buf, wx, y + h * 0.5 + 0.02, wz, 0.72, h, 0.72, team)
