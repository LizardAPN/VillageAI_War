"""Shared mesh helpers for 3D rendering (interleaved pos, normal, rgb per vertex)."""

from __future__ import annotations

import math

import numpy as np


def add_quad(
    buf: list[float],
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    n: tuple[float, float, float],
    rgb: tuple[float, float, float],
) -> None:
    nx, ny, nz = n
    r, g, b = rgb
    for px, py, pz in (p0, p1, p2):
        buf.extend([px, py, pz, nx, ny, nz, r, g, b])
    for px, py, pz in (p0, p2, p3):
        buf.extend([px, py, pz, nx, ny, nz, r, g, b])


def add_tri(
    buf: list[float],
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    n: tuple[float, float, float],
    rgb: tuple[float, float, float],
) -> None:
    nx, ny, nz = n
    r, g, b = rgb
    for px, py, pz in (p0, p1, p2):
        buf.extend([px, py, pz, nx, ny, nz, r, g, b])


def add_cuboid(
    buf: list[float],
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    rgb: tuple[float, float, float],
) -> None:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    r, g, b = rgb
    add_quad(
        buf,
        (cx - hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (0, 1, 0),
        (r * 1.1, g * 1.1, b * 1.1),
    )
    add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (0, -1, 0),
        (r * 0.55, g * 0.55, b * 0.55),
    )
    add_quad(
        buf,
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (1, 0, 0),
        (r * 0.85, g * 0.85, b * 0.85),
    )
    add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (-1, 0, 0),
        (r * 0.75, g * 0.75, b * 0.75),
    )
    add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (0, 0, 1),
        (r * 0.9, g * 0.9, b * 0.9),
    )
    add_quad(
        buf,
        (cx + hx, cy - hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (0, 0, -1),
        (r * 0.7, g * 0.7, b * 0.7),
    )


def add_sphere(
    buf: list[float],
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    rgb: tuple[float, float, float],
    stacks: int = 5,
    slices: int = 8,
) -> None:
    r, g, b = rgb
    for i in range(stacks):
        lat0 = np.pi * (-0.5 + i / stacks)
        lat1 = np.pi * (-0.5 + (i + 1) / stacks)
        z0, zr0 = np.sin(lat0) * radius, np.cos(lat0) * radius
        z1, zr1 = np.sin(lat1) * radius, np.cos(lat1) * radius
        for j in range(slices):
            lng0 = 2 * np.pi * j / slices
            lng1 = 2 * np.pi * (j + 1) / slices
            p00 = (
                float(cx + zr0 * np.cos(lng0)),
                float(cy + z0),
                float(cz + zr0 * np.sin(lng0)),
            )
            p01 = (
                float(cx + zr0 * np.cos(lng1)),
                float(cy + z0),
                float(cz + zr0 * np.sin(lng1)),
            )
            p10 = (
                float(cx + zr1 * np.cos(lng0)),
                float(cy + z1),
                float(cz + zr1 * np.sin(lng0)),
            )
            p11 = (
                float(cx + zr1 * np.cos(lng1)),
                float(cy + z1),
                float(cz + zr1 * np.sin(lng1)),
            )
            ex = p01[0] - p00[0]
            ey = p01[1] - p00[1]
            ez = p01[2] - p00[2]
            fx = p10[0] - p00[0]
            fy = p10[1] - p00[1]
            fz = p10[2] - p00[2]
            nx = ey * fz - ez * fy
            ny = ez * fx - ex * fz
            nz = ex * fy - ey * fx
            ln = float(np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
            nn = (nx / ln, ny / ln, nz / ln)
            add_quad(buf, p00, p01, p11, p10, nn, (r, g, b))


def add_cylinder_y(
    buf: list[float],
    cx: float,
    cy_bottom: float,
    cz: float,
    radius: float,
    height: float,
    segs: int,
    rgb: tuple[float, float, float],
    *,
    cap_top: bool = True,
    cap_bottom: bool = True,
) -> None:
    r, g, b = rgb
    y0, y1 = cy_bottom, cy_bottom + height
    for i in range(segs):
        a0 = 2 * np.pi * i / segs
        a1 = 2 * np.pi * (i + 1) / segs
        x0 = float(cx + radius * np.cos(a0))
        z0 = float(cz + radius * np.sin(a0))
        x1 = float(cx + radius * np.cos(a1))
        z1 = float(cz + radius * np.sin(a1))
        am = (a0 + a1) * 0.5
        nn = (float(np.cos(am)), 0.0, float(np.sin(am)))
        add_quad(buf, (x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0), nn, (r, g, b))
    if cap_top:
        for i in range(segs):
            a0 = 2 * np.pi * i / segs
            a1 = 2 * np.pi * (i + 1) / segs
            x0 = float(cx + radius * np.cos(a0))
            z0 = float(cz + radius * np.sin(a0))
            x1 = float(cx + radius * np.cos(a1))
            z1 = float(cz + radius * np.sin(a1))
            add_tri(
                buf,
                (cx, y1, cz),
                (x0, y1, z0),
                (x1, y1, z1),
                (0, 1, 0),
                (r * 1.08, g * 1.08, b * 1.08),
            )
    if cap_bottom:
        for i in range(segs):
            a0 = 2 * np.pi * i / segs
            a1 = 2 * np.pi * (i + 1) / segs
            x0 = float(cx + radius * np.cos(a0))
            z0 = float(cz + radius * np.sin(a0))
            x1 = float(cx + radius * np.cos(a1))
            z1 = float(cz + radius * np.sin(a1))
            add_tri(
                buf,
                (cx, y0, cz),
                (x1, y0, z1),
                (x0, y0, z0),
                (0, -1, 0),
                (r * 0.65, g * 0.65, b * 0.65),
            )


def add_pyramid(
    buf: list[float],
    cx: float,
    cy_base: float,
    cz: float,
    half_w: float,
    cy_apex: float,
    rgb: tuple[float, float, float],
) -> None:
    r, g, b = rgb
    hw = half_w
    yb, ya = cy_base, cy_apex
    nw = (cx - hw, yb, cz - hw)
    ne = (cx + hw, yb, cz - hw)
    se = (cx + hw, yb, cz + hw)
    sw = (cx - hw, yb, cz + hw)
    apex = (cx, ya, cz)

    for p0, p1, p2 in ((nw, ne, apex), (ne, se, apex), (se, sw, apex), (sw, nw, apex)):
        ex = p1[0] - p0[0]
        ey = p1[1] - p0[1]
        ez = p1[2] - p0[2]
        fx = p2[0] - p0[0]
        fy = p2[1] - p0[1]
        fz = p2[2] - p0[2]
        nx = ey * fz - ez * fy
        ny = ez * fx - ex * fz
        nz = ex * fy - ey * fx
        ln = float(np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8)
        nn = (nx / ln, ny / ln, nz / ln)
        add_tri(buf, p0, p1, p2, nn, (r, g, b))


def add_prism_y(
    buf: list[float],
    cx: float,
    cy_bot: float,
    cz: float,
    radius: float,
    height: float,
    sides: int,
    rgb: tuple[float, float, float],
    *,
    cap_bottom: bool = True,
    cap_top: bool = True,
) -> None:
    """Вертикальная призма с многоугольным основанием в плоскости XZ."""
    y_top = cy_bot + height
    r, g, b = rgb
    for i in range(sides):
        a0 = 2 * math.pi * i / sides
        a1 = 2 * math.pi * (i + 1) / sides
        x0 = float(cx + radius * math.cos(a0))
        z0 = float(cz + radius * math.sin(a0))
        x1 = float(cx + radius * math.cos(a1))
        z1 = float(cz + radius * math.sin(a1))
        am = (a0 + a1) * 0.5
        nn = (float(math.cos(am)), 0.0, float(math.sin(am)))
        add_quad(buf, (x0, cy_bot, z0), (x1, cy_bot, z1), (x1, y_top, z1), (x0, y_top, z0), nn, (r, g, b))
    if cap_bottom:
        for i in range(sides):
            a0 = 2 * math.pi * i / sides
            a1 = 2 * math.pi * (i + 1) / sides
            x0 = float(cx + radius * math.cos(a0))
            z0 = float(cz + radius * math.sin(a0))
            x1 = float(cx + radius * math.cos(a1))
            z1 = float(cz + radius * math.sin(a1))
            add_tri(
                buf,
                (cx, cy_bot, cz),
                (x1, cy_bot, z1),
                (x0, cy_bot, z0),
                (0, -1, 0),
                (r * 0.62, g * 0.62, b * 0.62),
            )
    if cap_top:
        for i in range(sides):
            a0 = 2 * math.pi * i / sides
            a1 = 2 * math.pi * (i + 1) / sides
            x0 = float(cx + radius * math.cos(a0))
            z0 = float(cz + radius * math.sin(a0))
            x1 = float(cx + radius * math.cos(a1))
            z1 = float(cz + radius * math.sin(a1))
            add_tri(
                buf,
                (cx, y_top, cz),
                (x0, y_top, z0),
                (x1, y_top, z1),
                (0, 1, 0),
                (r * 1.05, g * 1.05, b * 1.05),
            )
