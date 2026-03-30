"""OpenGL 3D board view (moderngl + pygame) — optional ``human_3d`` / ``rgb_array_3d``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from village_ai_war.state import (
    BuildingType,
    GameState,
    ResourceLayer,
    Role,
    TerrainType,
)

# Match pygame palette (roughly) for terrain
_TERRAIN_COLOR: dict[int, tuple[float, float, float]] = {
    int(TerrainType.GRASS): (0.22, 0.48, 0.34),
    int(TerrainType.MOUNTAIN): (0.38, 0.40, 0.46),
    int(TerrainType.FOREST): (0.14, 0.38, 0.26),
    int(TerrainType.STONE_DEPOSIT): (0.48, 0.50, 0.56),
    int(TerrainType.FIELD): (0.65, 0.58, 0.32),
}
# Buildings / team tint (slightly softer than unit bases)
_TEAM_RGB = ((0.82, 0.22, 0.28), (0.22, 0.48, 0.95))
# Bot **disk** — максимально различимые «КРАСНЫЙ / СИНИЙ» поля
_TEAM_NEON = ((0.98, 0.12, 0.22), (0.08, 0.55, 1.0))
# Роли — не смешиваем с телом; воин не «красный как команда»
_ROLE_RGB: dict[Role, tuple[float, float, float]] = {
    Role.WARRIOR: (1.0, 0.48, 0.12),
    Role.GATHERER: (0.98, 0.9, 0.22),
    Role.FARMER: (0.22, 0.9, 0.42),
    Role.BUILDER: (0.75, 0.35, 0.98),
}

_LEGEND_WIDTH_PX = 268


def _terrain_height(t: int) -> float:
    return {
        int(TerrainType.GRASS): 0.14,
        int(TerrainType.MOUNTAIN): 0.62,
        int(TerrainType.FOREST): 0.22,
        int(TerrainType.STONE_DEPOSIT): 0.18,
        int(TerrainType.FIELD): 0.10,
    }.get(t, 0.12)


def _mul4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b).astype(np.float32)


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, center: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """World → view (column vectors: gl_Position = P @ V @ vec4(pos,1))."""
    eye = eye.astype(np.float32)
    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, world_up.astype(np.float32))
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    v = np.eye(4, dtype=np.float32)
    v[0, :3] = right
    v[1, :3] = up
    v[2, :3] = -forward
    v[0, 3] = -float(np.dot(right, eye))
    v[1, 3] = -float(np.dot(up, eye))
    v[2, 3] = float(np.dot(forward, eye))
    return v


def _add_quad(
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


def _add_tri(
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


def _add_cuboid(
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
    # Six faces; +Y is up
    _add_quad(
        buf,
        (cx - hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (0, 1, 0),
        (r * 1.1, g * 1.1, b * 1.1),
    )
    _add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (0, -1, 0),
        (r * 0.55, g * 0.55, b * 0.55),
    )
    _add_quad(
        buf,
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (1, 0, 0),
        (r * 0.85, g * 0.85, b * 0.85),
    )
    _add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (-1, 0, 0),
        (r * 0.75, g * 0.75, b * 0.75),
    )
    _add_quad(
        buf,
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (0, 0, 1),
        (r * 0.9, g * 0.9, b * 0.9),
    )
    _add_quad(
        buf,
        (cx + hx, cy - hy, cz - hz),
        (cx - hx, cy - hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (0, 0, -1),
        (r * 0.7, g * 0.7, b * 0.7),
    )


def _add_sphere(
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
            _add_quad(buf, p00, p01, p11, p10, nn, (r, g, b))


def _add_cylinder_y(
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
        _add_quad(buf, (x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0), nn, (r, g, b))
    if cap_top:
        for i in range(segs):
            a0 = 2 * np.pi * i / segs
            a1 = 2 * np.pi * (i + 1) / segs
            x0 = float(cx + radius * np.cos(a0))
            z0 = float(cz + radius * np.sin(a0))
            x1 = float(cx + radius * np.cos(a1))
            z1 = float(cz + radius * np.sin(a1))
            _add_tri(
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
            _add_tri(
                buf,
                (cx, y0, cz),
                (x1, y0, z1),
                (x0, y0, z0),
                (0, -1, 0),
                (r * 0.65, g * 0.65, b * 0.65),
            )


def _add_pyramid(
    buf: list[float],
    cx: float,
    cy_base: float,
    cz: float,
    half_w: float,
    cy_apex: float,
    rgb: tuple[float, float, float],
) -> None:
    """Квадратное основание в плоскости XZ, вершина на оси Y."""
    r, g, b = rgb
    hw = half_w
    yb, ya = cy_base, cy_apex
    # углы: NW, NE, SE, SW (вид сверху, Z+ «вниз» как строки карты)
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
        _add_tri(buf, p0, p1, p2, nn, (r, g, b))


_VS = """
#version 330
uniform mat4 vp;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_nrm;
out vec3 v_col;
void main() {
    v_nrm = in_normal;
    v_col = in_color;
    gl_Position = vp * vec4(in_pos, 1.0);
}
"""

_FS = """
#version 330
out vec4 f_color;
in vec3 v_nrm;
in vec3 v_col;
void main() {
    vec3 L = normalize(vec3(0.45, 0.82, 0.38));
    float ndl = max(dot(normalize(v_nrm), L), 0.12);
    vec3 amb = v_col * 0.28;
    vec3 dif = v_col * 0.72 * ndl;
    f_color = vec4(amb + dif, 1.0);
}
"""


def _install_linux_gl_soname_patch() -> Any:
    """moderngl loads ``libGL.so`` / ``libEGL.so``; Debian/Ubuntu ship ``libGL.so.1`` only.

    Temporarily map those names for moderngl's internal ``ctypes.CDLL`` calls.
    Returns ``restore()`` to undo the patch (call after ``create_context``).
    """
    import ctypes
    import sys

    if not sys.platform.startswith("linux"):
        return lambda: None

    _real = ctypes.CDLL

    def _cdll(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "libEGL.so":
            for alt in ("libEGL.so.1",):
                try:
                    return _real(alt, *args, **kwargs)
                except OSError:
                    continue
        if name == "libGL.so":
            import ctypes.util

            found = ctypes.util.find_library("GL")
            if found:
                try:
                    return _real(found, *args, **kwargs)
                except OSError:
                    pass
            for alt in ("libGL.so.1",):
                try:
                    return _real(alt, *args, **kwargs)
                except OSError:
                    continue
        return _real(name, *args, **kwargs)

    ctypes.CDLL = _cdll

    def _restore() -> None:
        ctypes.CDLL = _real

    return _restore


class Moderngl3DRenderer:
    """Extruded terrain tiles, box buildings, sphere bots; orbit camera."""

    def __init__(self, config: Mapping[str, Any], state: GameState | None) -> None:
        try:
            import moderngl
            import pygame
        except ImportError as e:
            raise ImportError(
                "3D rendering requires packages: pip install moderngl (pygame is already required)."
            ) from e

        self._moderngl = moderngl
        self._pygame = pygame
        self._config = config
        rend = config.get("rendering", {})
        self._n = int(config["map"]["size"])
        self._win_w = int(rend.get("window_width_3d", 960))
        self._win_h = int(rend.get("window_height_3d", 720))
        self._fov = float(rend.get("camera_fov_deg", 50.0))
        self._dist_scale = float(rend.get("camera_dist_scale", 1.4))
        self._auto_rotate = float(rend.get("auto_rotate_deg_per_sec", 10.0))
        self._pitch_deg = float(rend.get("camera_pitch_deg", 32.0))
        self._yaw_deg = float(rend.get("camera_yaw_deg", 40.0))
        self._fps = max(1, int(rend.get("fps", 30)))

        pygame.init()
        pygame.display.set_mode(
            (self._win_w, self._win_h),
            pygame.OPENGL | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("Village AI War — 3D")

        restore_gl_cdll = _install_linux_gl_soname_patch()
        try:
            self._ctx = moderngl.create_context()
        except Exception as e:
            err = str(e).lower()
            if (
                "libgl" in err
                or "libegl" in err
                or "cannot open shared object" in err
                or "no such file" in err
            ):
                try:
                    pygame.display.quit()
                except Exception:
                    pass
                raise RuntimeError(
                    "Could not load OpenGL (Mesa). Packages libgl1/libegl1 are installed but "
                    "the driver may be missing (e.g. WSL needs WSLg or GPU passthrough). "
                    "Try: sudo apt install -y libgl1-mesa-dri  |  Or use 2D: python scripts/run_game.py\n"
                    f"Underlying error: {e}"
                ) from e
            raise
        finally:
            restore_gl_cdll()

        self._ctx.enable(self._moderngl.DEPTH_TEST)
        self._ctx.gc_mode = "auto"

        self._prog = self._ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self._vbo: Any = None
        self._vao: Any = None
        self._terrain_sig: bytes | None = None
        self._clock = pygame.time.Clock()
        self._render_backend = "moderngl"

    def _grid_to_world(self, gx: int, gz: int) -> tuple[float, float]:
        n = self._n
        return float(gx) - n * 0.5 + 0.5, float(gz) - n * 0.5 + 0.5

    def _build_static_terrain(self, state: GameState) -> None:
        terrain = np.asarray(state.terrain, dtype=np.int32)
        sig = terrain.tobytes()
        if sig == self._terrain_sig and self._vao is not None:
            return
        self._terrain_sig = sig
        buf: list[float] = []
        n = self._n
        cell = 0.92
        for gz in range(n):
            for gx in range(n):
                wx, wz = self._grid_to_world(gx, gz)
                t = int(terrain[gz, gx])
                h = _terrain_height(t)
                rgb = _TERRAIN_COLOR.get(t, (0.3, 0.3, 0.35))
                _add_cuboid(buf, wx, h * 0.5, wz, cell, h, cell, rgb)
        arr = np.asarray(buf, dtype=np.float32)
        if self._vbo is not None:
            self._vbo.release()
        self._vbo = self._ctx.buffer(arr.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
        )

    def _build_dynamic_geometry(self, state: GameState) -> np.ndarray:
        buf: list[float] = []
        res_layer = np.asarray(state.resources, dtype=np.int32)
        n = self._n
        for gz in range(n):
            for gx in range(n):
                layer = int(res_layer[gz, gx])
                if layer == int(ResourceLayer.NONE):
                    continue
                wx, wz = self._grid_to_world(gx, gz)
                t = int(state.terrain[gz][gx])
                base_h = _terrain_height(t)
                tint = (0.95, 0.92, 0.55) if layer == int(ResourceLayer.FOREST) else (0.85, 0.88, 0.95)
                _add_cuboid(buf, wx, base_h + 0.08, wz, 0.25, 0.06, 0.25, tint)

        for v in state.villages:
            tr, tg, tb = _TEAM_RGB[v.team] if v.team < 2 else (0.6, 0.6, 0.6)
            for b in v.buildings:
                if b.hp <= 0:
                    continue
                bx, bz = b.position
                wx, wz = self._grid_to_world(int(bx), int(bz))
                t = int(state.terrain[int(bz)][int(bx)])
                base_h = _terrain_height(t)
                hp_f = b.hp / max(b.max_hp, 1)
                h = 0.35 + 0.45 * hp_f
                if b.building_type == BuildingType.TOWNHALL:
                    h += 0.15
                elif b.building_type == BuildingType.TOWER:
                    h += 0.25
                _add_cuboid(buf, wx, base_h + h * 0.5 + 0.02, wz, 0.72, h, 0.72, (tr, tg, tb))

        for v in state.villages:
            for bot in v.bots:
                if not bot.is_alive:
                    continue
                bx, bz = bot.position
                wx, wz = self._grid_to_world(int(bx), int(bz))
                t = int(state.terrain[int(bz)][int(bx)])
                base_h = _terrain_height(t)
                rr, rg, rb = _ROLE_RGB.get(bot.role, (0.8, 0.8, 0.8))
                tr, tg, tb = _TEAM_RGB[v.team] if v.team < 2 else (0.7, 0.7, 0.7)
                mix = (
                    0.58 * rr + 0.42 * tr,
                    0.58 * rg + 0.42 * tg,
                    0.58 * rb + 0.42 * tb,
                )
                cy = base_h + 0.28
                _add_sphere(buf, wx, cy, wz, 0.24, mix)

        return np.asarray(buf, dtype=np.float32)

    def _view_proj(self) -> np.ndarray:
        n = self._n
        center = np.array([0.0, n * 0.12, 0.0], dtype=np.float32)
        dist = max(n * self._dist_scale, 8.0)
        yaw = np.radians(self._yaw_deg)
        pitch = np.radians(self._pitch_deg)
        eye = center + np.array(
            [
                dist * np.cos(pitch) * np.cos(yaw),
                dist * np.sin(pitch),
                dist * np.cos(pitch) * np.sin(yaw),
            ],
            dtype=np.float32,
        )
        proj = perspective(self._fov, self._win_w / max(self._win_h, 1), 0.1, 200.0)
        view = look_at(eye, center, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return _mul4(proj, view)

    def render(self, state: GameState, mode: str) -> np.ndarray | None:
        import pygame

        self._build_static_terrain(state)
        dyn = self._build_dynamic_geometry(state)
        vp = self._view_proj()

        self._ctx.clear(0.12, 0.14, 0.20)
        self._prog["vp"].write(np.ascontiguousarray(vp.T, dtype=np.float32).tobytes())

        assert self._vao is not None
        self._vao.render()

        dyn_vbo = self._ctx.buffer(dyn.tobytes()) if dyn.size else None
        if dyn_vbo is not None:
            dyn_vao = self._ctx.vertex_array(
                self._prog,
                [(dyn_vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            )
            dyn_vao.render()
            dyn_vao.release()
            dyn_vbo.release()

        pygame.display.flip()
        pygame.event.pump()

        dt_ms = self._clock.tick(self._fps)
        self._yaw_deg += self._auto_rotate * (dt_ms / 1000.0)
        self._yaw_deg %= 360.0

        win = state.winner
        win_s = "…" if win is None else ("Red" if win == 0 else "Blue" if win == 1 else "draw")
        pygame.display.set_caption(
            f"Village AI War — 3D | tick {state.tick}/{state.max_ticks} | {win_s}"
        )

        if mode == "rgb_array_3d":
            data = self._ctx.screen.read(components=3)
            arr = np.frombuffer(data, dtype=np.uint8).reshape((self._win_h, self._win_w, 3))
            return np.ascontiguousarray(arr[::-1])

        return None

    def close(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        self._terrain_sig = None
        self._pygame.display.quit()
        self._pygame.quit()
