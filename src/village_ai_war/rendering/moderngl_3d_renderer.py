"""OpenGL 3D board view (moderngl + pygame) — optional ``human_3d`` / ``rgb_array_3d``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from village_ai_war.rendering.building_models_3d import add_building_variant as _add_building_variant
from village_ai_war.rendering.world_scenery_3d import (
    add_resource_prop as _add_resource_prop,
    add_terrain_cell as _add_terrain_cell,
    terrain_height as _terrain_height,
)
from village_ai_war.rendering.mesh_primitives import (
    add_cuboid as _add_cuboid,
    add_cylinder_y as _add_cylinder_y,
    add_pyramid as _add_pyramid,
    add_quad as _add_quad,
    add_sphere as _add_sphere,
    add_tri as _add_tri,
)
from village_ai_war.state import (
    GameState,
    GlobalRewardMode,
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
# Khronos GL_SCISSOR_TEST (moderngl.Context has no SCISSOR_TEST flag alias)
_GL_SCISSOR_TEST = 0x0C11
_LEGEND_HUD_HEIGHT_MIN = 102
_TEAM_HUD_RGB = ((255, 130, 130), (150, 185, 255))
_HUD_ROW_BG = (30, 33, 44)
_ACCENT_HUD = (92, 140, 220)


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

_VS_UI = """
#version 330
uniform mat4 ortho;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = ortho * vec4(in_pos, 0.0, 1.0);
}
"""

_FS_UI = """
#version 330
uniform sampler2D tex0;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(tex0, v_uv);
}
"""


def _lerp_rgb_int(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _mode_label_ru(m: GlobalRewardMode) -> str:
    return {
        GlobalRewardMode.NEUTRAL: "нейтр.",
        GlobalRewardMode.DEFEND: "защита",
        GlobalRewardMode.ATTACK: "атака",
        GlobalRewardMode.GATHER: "сбор",
    }.get(m, "?")


def _legend_hud_height_for_state(
    _pygame: Any, _small: Any, _body: Any, _state: GameState, _legend_w: int
) -> int:
    """Высота нижней панели: тик + две строки команд (HP рисуется над юнитами на карте)."""
    tick_block = 6 + 16 + 4
    row_h = 32
    gap = 3
    return max(_LEGEND_HUD_HEIGHT_MIN, tick_block + 2 * (row_h + gap) + 10)


def _ortho_pixel(win_w: float, win_h: float) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / win_w
    m[1, 1] = 2.0 / win_h
    m[0, 3] = -1.0
    m[1, 3] = -1.0
    m[2, 2] = -1.0
    m[3, 3] = 1.0
    return m


def _draw_legend_hud(
    surf: Any,
    pygame: Any,
    small: Any,
    body: Any,
    w: int,
    h: int,
    state: GameState,
    hud_h: int,
) -> None:
    """Нижняя зона панели: тик и ресурсы по командам (HP — на модели юнита над клеткой)."""
    y0 = h - hud_h
    pygame.draw.rect(surf, (22, 24, 32), pygame.Rect(0, y0, w, hud_h))
    pygame.draw.line(surf, _ACCENT_HUD, (0, y0), (w, y0), 2)

    win = state.winner
    if win is None:
        win_s = "идёт"
    elif win == 0:
        win_s = "RED"
    elif win == 1:
        win_s = "BLUE"
    else:
        win_s = "ничья"

    yy = y0 + 6
    head = f"тик {state.tick}/{state.max_ticks}  ·  {win_s}"
    surf.blit(small.render(head, True, (200, 205, 220)), (10, yy))
    yy += 16

    names = ("КРАСНЫЙ", "СИНИЙ")
    row_h = 32
    for i, v in enumerate(state.villages[:2]):
        col = _TEAM_HUD_RGB[i] if i < 2 else (200, 200, 210)
        name = names[i] if i < len(names) else f"команда {v.team}"
        r = v.resources
        alive_n = sum(1 for b in v.bots if b.is_alive)
        mode = _mode_label_ru(v.global_reward_mode)
        panel = pygame.Rect(6, yy, w - 12, row_h)
        pygame.draw.rect(surf, _HUD_ROW_BG, panel)
        pygame.draw.rect(surf, _lerp_rgb_int(col, (255, 255, 255), 0.65), panel, 1)
        pygame.draw.rect(surf, col, pygame.Rect(panel.left, panel.top, 3, panel.height))
        line1 = f"{name}   боты {alive_n}/{v.pop_cap}   AI: {mode}"
        line2 = f"дер. {r.wood}   кам. {r.stone}   еда {r.food}"
        surf.blit(small.render(line1, True, col), (panel.left + 8, panel.top + 4))
        surf.blit(body.render(line2, True, (210, 212, 220)), (panel.left + 8, panel.top + 17))
        yy += row_h + 3


def _add_bot_figure(
    buf: list[float],
    wx: float,
    base_h: float,
    wz: float,
    team_rgb: tuple[float, float, float],
    role_rgb: tuple[float, float, float],
    role: Role,
) -> None:
    """Силуэт: ноги и туловище — цвет команды; шлем и аксессуары роли — цвет роли."""
    y = float(base_h)
    tr, tg, tb = team_rgb
    rr, rg, rb = role_rgb
    leg_tint = (tr * 0.52 + 0.04, tg * 0.52 + 0.04, tb * 0.52 + 0.04)
    torso = (min(tr * 1.05, 0.99), min(tg * 1.05, 0.99), min(tb * 1.05, 0.99))
    arm_col = (tr * 0.78, tg * 0.78, tb * 0.78)

    leg_h, leg_w = 0.13, 0.062
    spread = 0.092
    for sx in (-spread, spread):
        _add_cuboid(
            buf,
            wx + sx,
            y + leg_h * 0.5,
            wz,
            leg_w,
            leg_h,
            leg_w * 1.08,
            leg_tint,
        )
    y += leg_h

    tw, th, td = 0.24, 0.30, 0.13
    ty = y + th * 0.5
    _add_cuboid(buf, wx, ty, wz, tw, th, td, torso)

    arm_y = y + th * 0.52
    arm_l, aw, ad = 0.13, 0.052, 0.052
    ox = tw * 0.5 + aw * 0.48
    for sx in (-ox, ox):
        _add_cuboid(buf, wx + sx, arm_y, wz + 0.018, aw, arm_l, ad, arm_col)

    y += th
    head_y = y + 0.095
    skin = (0.91, 0.79, 0.64)
    _add_sphere(buf, wx, head_y, wz, 0.092, skin, stacks=4, slices=8)
    helm = (min(rr * 1.08, 0.98), min(rg * 1.08, 0.98), min(rb * 1.08, 0.98))
    _add_cuboid(buf, wx, head_y + 0.105, wz, 0.15, 0.048, 0.15, helm)

    if role == Role.WARRIOR:
        blade = (0.75, 0.76, 0.82)
        _add_cuboid(buf, wx + ox * 0.85, arm_y + 0.02, wz + 0.16, 0.035, 0.045, 0.20, blade)
    elif role == Role.GATHERER:
        bag = (rr * 0.42 + 0.32, rg * 0.38 + 0.28, rb * 0.32 + 0.22)
        _add_cuboid(buf, wx - ox * 0.85, arm_y, wz + 0.12, 0.08, 0.06, 0.08, bag)
    elif role == Role.FARMER:
        hat = (min(rr * 0.72 + 0.2, 0.96), min(rg * 0.68 + 0.18, 0.92), min(rb * 0.35 + 0.12, 0.85))
        _add_cylinder_y(
            buf,
            wx,
            head_y + 0.14,
            wz,
            0.11,
            0.04,
            8,
            hat,
            cap_top=True,
            cap_bottom=True,
        )
    elif role == Role.BUILDER:
        brick = (rr * 0.45 + 0.35, rg * 0.42 + 0.33, rb * 0.4 + 0.32)
        _add_cuboid(buf, wx - ox * 0.9, arm_y - 0.02, wz + 0.1, 0.05, 0.08, 0.05, brick)


def _add_bot_hp_bar(
    buf: list[float],
    wx: float,
    wz: float,
    base_h: float,
    hp_frac: float,
) -> None:
    """Полоска HP над головой: светлый трек + яркая заливка слоем выше (без z-fight с треком)."""
    f = float(np.clip(hp_frac, 0.0, 1.0))
    bar_w, bar_d = 0.46, 0.075
    h_track = 0.048
    h_fill = 0.024
    # Центр трека — над шлемом/шляпой
    cy_track = float(base_h) + 0.79
    rim = (0.28, 0.30, 0.36)
    track = (0.48, 0.50, 0.56)
    _add_cuboid(
        buf, wx, cy_track, wz, bar_w + 0.024, h_track + 0.012, bar_d + 0.018, rim
    )
    _add_cuboid(buf, wx, cy_track, wz, bar_w, h_track, bar_d, track)
    if f <= 0.001:
        return
    margin = bar_w * 0.08
    inner = bar_w - 2.0 * margin
    fw = max(inner * f, 0.02)
    left = wx - bar_w * 0.5 + margin
    fx = left + fw * 0.5
    if f > 0.35:
        fill = (0.22, 0.98, 0.42)
    elif f > 0.15:
        fill = (1.0, 0.78, 0.18)
    else:
        fill = (1.0, 0.32, 0.28)
    # Заливка целиком над верхней гранью трека — всегда видна при типичном свете (amb+dif)
    cy_fill = cy_track + h_track * 0.5 + h_fill * 0.5 + 0.004
    _add_cuboid(buf, fx, cy_fill, wz, fw, h_fill, bar_d * 0.88, fill)


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
    """3D board: terrain columns, varied buildings, team+role bots, side legend texture."""

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
        self._legend_w = int(rend.get("legend_width_3d", _LEGEND_WIDTH_PX))
        self._map_vp_w = max(120, self._win_w - self._legend_w)
        self._fov = float(rend.get("camera_fov_deg", 50.0))
        self._dist_scale = float(rend.get("camera_dist_scale", 1.4))
        self._zoom_mul = 1.0
        self._zoom_mul_min = float(rend.get("camera_zoom_mul_min", 0.22))
        self._zoom_mul_max = float(rend.get("camera_zoom_mul_max", 2.85))
        self._zoom_wheel_step = float(rend.get("camera_zoom_wheel_step", 0.11))
        self._dist_min = float(rend.get("camera_dist_min", 3.5))
        self._auto_rotate = float(rend.get("auto_rotate_deg_per_sec", 0.0))
        self._pitch_deg = float(rend.get("camera_pitch_deg", 32.0))
        self._yaw_deg = float(rend.get("camera_yaw_deg", 40.0))
        self._fps = max(1, int(rend.get("fps", 30)))
        self._orbit_sens = float(rend.get("orbit_mouse_sensitivity", 0.35))
        self._orbit_key_dps = float(rend.get("orbit_key_deg_per_sec", 72.0))
        self._pitch_min = float(rend.get("camera_pitch_min_deg", 8.0))
        self._pitch_max = float(rend.get("camera_pitch_max_deg", 85.0))
        self._orbit_drag = False
        self._orbit_last: tuple[int, int] = (0, 0)

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

        self._f_leg_title = pygame.font.SysFont("consolas", 16, bold=True)
        self._f_leg_body = pygame.font.SysFont("consolas", 13)
        self._f_leg_small = pygame.font.SysFont("consolas", 11)
        self._tex_legend = self._ctx.texture((self._legend_w, self._win_h), 4)
        if state is not None:
            self._upload_legend_texture(state)
        else:
            blank = bytes([22, 24, 32, 255]) * (self._legend_w * self._win_h)
            self._tex_legend.write(blank)

        self._prog_ui = self._ctx.program(vertex_shader=_VS_UI, fragment_shader=_FS_UI)
        mx = float(self._map_vp_w)
        ww = float(self._win_w)
        wh = float(self._win_h)
        ui_data = np.array(
            [
                mx,
                0.0,
                0.0,
                1.0,
                ww,
                0.0,
                1.0,
                1.0,
                ww,
                wh,
                1.0,
                0.0,
                mx,
                0.0,
                0.0,
                1.0,
                ww,
                wh,
                1.0,
                0.0,
                mx,
                wh,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        self._vbo_ui = self._ctx.buffer(ui_data.tobytes())
        self._vao_ui = self._ctx.vertex_array(
            self._prog_ui,
            [(self._vbo_ui, "2f 2f", "in_pos", "in_uv")],
        )

    def _upload_legend_texture(self, state: GameState) -> None:
        """Легенда (статика) + нижний HUD с ресурсами и ботами; каждый кадр из ``render``."""
        pygame = self._pygame
        surf = pygame.Surface((self._legend_w, self._win_h))
        surf.fill((22, 24, 32))
        title = self._f_leg_title
        body = self._f_leg_body
        small = self._f_leg_small
        hud_h = _legend_hud_height_for_state(pygame, small, body, state, self._legend_w)
        y = 10
        hud_top = self._win_h - hud_h

        def bl(txt: str, dy: int, color: tuple[int, int, int], font: Any = body) -> None:
            nonlocal y
            if y > hud_top - 8:
                return
            surf.blit(font.render(txt, True, color), (12, y))
            y += dy

        bl("Легенда (3D)", 22, (245, 248, 255), title)
        bl("Боты", 16, (120, 170, 230), small)
        bl("Фигурки: тело — команда", 16, (200, 200, 210))
        bl("шлем/убор — роль W/G/F/B", 16, (200, 200, 210))
        bl("HP — полоска над юнитом", 16, (175, 205, 235))
        y += 4
        for i, label in enumerate(("RED (0)", "BLUE (1)")):
            if y > hud_top - 24:
                break
            c = tuple(int(255 * x) for x in _TEAM_NEON[i])
            pygame.draw.rect(surf, c, pygame.Rect(12, y, 10, 14))
            pygame.draw.rect(surf, (60, 65, 80), pygame.Rect(12, y, 10, 14), 1)
            surf.blit(body.render(label, True, (220, 222, 230)), (28, y - 1))
            y += 20
        y += 6
        bl("Роли (цвет шлема)", 16, (120, 170, 230), small)
        for role, lab in (
            (Role.WARRIOR, "W  воин (оранж.)"),
            (Role.GATHERER, "G  сборщик"),
            (Role.FARMER, "F  фермер"),
            (Role.BUILDER, "B  строитель"),
        ):
            if y > hud_top - 22:
                break
            rr, rg, rb = _ROLE_RGB[role]
            c = (int(rr * 255), int(rg * 255), int(rb * 255))
            pygame.draw.rect(surf, c, pygame.Rect(15, y + 1, 14, 12))
            pygame.draw.rect(surf, (40, 44, 55), pygame.Rect(15, y + 1, 14, 12), 1)
            surf.blit(body.render(lab, True, (210, 212, 220)), (34, y))
            y += 20
        y += 6
        bl("Ландшафт", 16, (120, 170, 230), small)
        terrain_order = (
            TerrainType.GRASS,
            TerrainType.FOREST,
            TerrainType.MOUNTAIN,
            TerrainType.STONE_DEPOSIT,
            TerrainType.FIELD,
        )
        names = {
            int(TerrainType.GRASS): "трава",
            int(TerrainType.FOREST): "лес",
            int(TerrainType.MOUNTAIN): "гора",
            int(TerrainType.STONE_DEPOSIT): "камень",
            int(TerrainType.FIELD): "поле",
        }
        for tt in terrain_order:
            if y > hud_top - 20:
                break
            tr, tg, tb = _TERRAIN_COLOR[int(tt)]
            c = (int(tr * 255), int(tg * 255), int(tb * 255))
            pygame.draw.rect(surf, c, pygame.Rect(12, y, 18, 14))
            pygame.draw.rect(surf, (50, 55, 70), pygame.Rect(12, y, 18, 14), 1)
            surf.blit(body.render(names[int(tt)], True, (210, 212, 220)), (36, y - 1))
            y += 18
        y += 6
        bl("Здания — детальные модели", 16, (120, 170, 230), small)
        bl("TH ратуша, казармы, ферма, склады…", 16, (170, 175, 190), small)
        bl("Ресурсы: 3D (лес/руда/урожай)", 16, (170, 175, 190), small)
        bl("Колёсико / +/- : масштаб", 16, (170, 175, 190), small)

        _draw_legend_hud(
            surf, pygame, small, body, self._legend_w, self._win_h, state, hud_h
        )
        self._tex_legend.write(pygame.image.tobytes(surf, "RGBA"))

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
                _add_terrain_cell(buf, wx, wz, t, gx, gz, cell)
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
        amounts = np.asarray(state.resource_amounts, dtype=np.int32)
        n = self._n
        for gz in range(n):
            for gx in range(n):
                layer = int(res_layer[gz, gx])
                if layer == int(ResourceLayer.NONE):
                    continue
                amt = int(amounts[gz, gx])
                if amt <= 0:
                    continue
                wx, wz = self._grid_to_world(gx, gz)
                t = int(state.terrain[gz][gx])
                base_h = _terrain_height(t)
                _add_resource_prop(buf, wx, wz, base_h, layer, amt, gx, gz)

        for v in state.villages:
            tr, tg, tb = _TEAM_RGB[v.team] if v.team < 2 else (0.6, 0.6, 0.6)
            for b in v.buildings:
                if b.hp <= 0:
                    continue
                bx, bz = b.position
                wx, wz = self._grid_to_world(int(bx), int(bz))
                t = int(state.terrain[int(bz)][int(bx)])
                base_h = _terrain_height(t)
                _add_building_variant(buf, wx, wz, base_h, b, tr, tg, tb)

        for v in state.villages:
            for bot in v.bots:
                if not bot.is_alive:
                    continue
                bx, bz = bot.position
                wx, wz = self._grid_to_world(int(bx), int(bz))
                t = int(state.terrain[int(bz)][int(bx)])
                base_h = _terrain_height(t)
                team_idx = v.team
                tn = _TEAM_NEON[team_idx] if team_idx < 2 else (0.55, 0.55, 0.58)
                rr, rg, rb = _ROLE_RGB.get(bot.role, (0.85, 0.85, 0.88))
                _add_bot_figure(buf, wx, base_h, wz, tn, (rr, rg, rb), bot.role)
                hp_f = bot.hp / max(bot.max_hp, 1)
                _add_bot_hp_bar(buf, wx, wz, base_h, hp_f)

        return np.asarray(buf, dtype=np.float32)

    def _view_proj(self) -> np.ndarray:
        n = self._n
        center = np.array([0.0, n * 0.12, 0.0], dtype=np.float32)
        dist = max(n * self._dist_scale * self._zoom_mul, self._dist_min)
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
        proj = perspective(self._fov, self._map_vp_w / max(self._win_h, 1), 0.1, 200.0)
        view = look_at(eye, center, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return _mul4(proj, view)

    def render(self, state: GameState, mode: str) -> np.ndarray | None:
        import pygame

        self._upload_legend_texture(state)
        self._build_static_terrain(state)
        dyn = self._build_dynamic_geometry(state)
        vp = self._view_proj()

        self._ctx.enable_direct(_GL_SCISSOR_TEST)
        self._ctx.scissor = (0, 0, self._map_vp_w, self._win_h)
        self._ctx.viewport = (0, 0, self._map_vp_w, self._win_h)
        self._ctx.clear(0.11, 0.12, 0.17)
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

        self._ctx.scissor = None
        self._ctx.disable_direct(_GL_SCISSOR_TEST)
        self._ctx.viewport = (0, 0, self._win_w, self._win_h)
        self._ctx.disable(self._moderngl.DEPTH_TEST)
        ortho = _ortho_pixel(float(self._win_w), float(self._win_h))
        self._prog_ui["ortho"].write(np.ascontiguousarray(ortho.T, dtype=np.float32).tobytes())
        self._tex_legend.use(0)
        self._prog_ui["tex0"] = 0
        self._vao_ui.render()
        self._ctx.enable(self._moderngl.DEPTH_TEST)

        pygame.display.flip()

        dt_ms = self._clock.tick(self._fps)
        dt = dt_ms / 1000.0
        def _apply_zoom(direction: float) -> None:
            """direction > 0 — приблизить (меньше расстояние), < 0 — отдалить."""
            if direction == 0.0:
                return
            step = self._zoom_wheel_step
            if direction > 0:
                self._zoom_mul *= max(0.5, 1.0 - step)
            else:
                self._zoom_mul *= 1.0 + step
            self._zoom_mul = max(
                self._zoom_mul_min, min(self._zoom_mul_max, self._zoom_mul)
            )

        _mw = getattr(pygame, "MOUSEWHEEL", None)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < self._map_vp_w:
                    self._orbit_drag = True
                    self._orbit_last = (event.pos[0], event.pos[1])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._orbit_drag = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < self._map_vp_w:
                if event.button == 4:
                    _apply_zoom(1.0)
                elif event.button == 5:
                    _apply_zoom(-1.0)
            elif _mw is not None and event.type == _mw:
                mx, _my = pygame.mouse.get_pos()
                if mx < self._map_vp_w:
                    y = float(getattr(event, "y", 0))
                    if y > 0:
                        _apply_zoom(1.0)
                    elif y < 0:
                        _apply_zoom(-1.0)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_PAGEUP):
                    _apply_zoom(1.0)
                elif event.key in (pygame.K_MINUS, pygame.K_PAGEDOWN):
                    _apply_zoom(-1.0)
            elif event.type == pygame.MOUSEMOTION and self._orbit_drag:
                x, y = event.pos[0], event.pos[1]
                lx, ly = self._orbit_last
                self._yaw_deg += (x - lx) * self._orbit_sens
                self._pitch_deg -= (y - ly) * self._orbit_sens
                self._pitch_deg = max(self._pitch_min, min(self._pitch_max, self._pitch_deg))
                self._orbit_last = (x, y)

        keys = pygame.key.get_pressed()
        k = self._orbit_key_dps * dt
        if keys[pygame.K_LEFT]:
            self._yaw_deg -= k
        if keys[pygame.K_RIGHT]:
            self._yaw_deg += k
        if keys[pygame.K_UP]:
            self._pitch_deg = max(self._pitch_min, self._pitch_deg - k)
        if keys[pygame.K_DOWN]:
            self._pitch_deg = min(self._pitch_max, self._pitch_deg + k)

        if self._auto_rotate != 0.0:
            self._yaw_deg += self._auto_rotate * dt
        self._yaw_deg %= 360.0

        win = state.winner
        win_s = "…" if win is None else ("Red" if win == 0 else "Blue" if win == 1 else "draw")
        pygame.display.set_caption(
            f"Village AI War — 3D | tick {state.tick}/{state.max_ticks} | {win_s} | "
            "LMB: orbit · wheel/±: zoom · arrows: tilt"
        )

        if mode == "rgb_array_3d":
            data = self._ctx.screen.read(components=3)
            arr = np.frombuffer(data, dtype=np.uint8).reshape((self._win_h, self._win_w, 3))
            return np.ascontiguousarray(arr[::-1])

        return None

    def close(self) -> None:
        if getattr(self, "_vao_ui", None) is not None:
            self._vao_ui.release()
            self._vao_ui = None
        if getattr(self, "_vbo_ui", None) is not None:
            self._vbo_ui.release()
            self._vbo_ui = None
        if getattr(self, "_tex_legend", None) is not None:
            self._tex_legend.release()
            self._tex_legend = None
        if getattr(self, "_prog_ui", None) is not None:
            self._prog_ui.release()
            self._prog_ui = None
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        self._terrain_sig = None
        self._pygame.display.quit()
        self._pygame.quit()
