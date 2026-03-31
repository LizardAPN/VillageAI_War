"""Pygame grid renderer for ``human`` and ``rgb_array`` modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from village_ai_war.state import (
    BuildingType,
    GameState,
    GlobalRewardMode,
    ResourceLayer,
    Role,
    TerrainType,
)

# --- Shared palette (map + legend stay in sync) ---
_TERRAIN_RGB: dict[int, tuple[int, int, int]] = {
    int(TerrainType.GRASS): (52, 118, 82),
    int(TerrainType.MOUNTAIN): (88, 92, 108),
    int(TerrainType.FOREST): (28, 92, 58),
    int(TerrainType.STONE_DEPOSIT): (108, 112, 128),
    int(TerrainType.FIELD): (158, 142, 72),
}
_TERRAIN_HIGHLIGHT: dict[int, tuple[int, int, int]] = {
    int(TerrainType.GRASS): (72, 148, 102),
    int(TerrainType.MOUNTAIN): (118, 122, 138),
    int(TerrainType.FOREST): (48, 122, 78),
    int(TerrainType.STONE_DEPOSIT): (138, 142, 158),
    int(TerrainType.FIELD): (188, 168, 98),
}
_DEFAULT_TERRAIN = (48, 52, 58)
_GRID_LINE = (36, 40, 50)

_TEAM_FILL = ((196, 72, 88), (72, 128, 220))
_TEAM_DARK = ((120, 40, 52), (40, 72, 140))
_TEAM_HUD = ((255, 130, 130), (150, 185, 255))

_ROLE_FILL: dict[Role, tuple[int, int, int]] = {
    Role.WARRIOR: (232, 96, 96),
    Role.GATHERER: (240, 210, 88),
    Role.FARMER: (110, 220, 130),
    Role.BUILDER: (190, 130, 240),
}

# UI chrome (human window)
_BG_DEEP = (14, 16, 22)
_BG_PANEL = (26, 29, 38)
_BG_PANEL_EDGE = (48, 54, 68)
_ACCENT = (92, 140, 220)
_TITLE_FG = (210, 215, 230)
_MAP_FRAME = (58, 64, 82)
_HUD_BAR_BG = (22, 24, 32)
_HUD_ROW_BG = (30, 33, 44)
_HP_BG = (24, 28, 32)
_HP_OK = (52, 200, 120)
_HP_LOW = (220, 180, 60)

# Layout for human mode: map + legend + HUD (rgb_array stays map-only).
_MARGIN_LEFT = 26
_MARGIN_TOP = 22
_LEGEND_WIDTH = 280
_LEGEND_PAD = 14
_BOTTOM_HUD_HEIGHT = 96
_TITLE_BAR = 26
# Legend text needs a minimum height so it is not clipped on small maps.
_MIN_LEGEND_BLOCK_H = 500


def _terrain_label(t: int) -> str:
    return {
        int(TerrainType.GRASS): "Grass",
        int(TerrainType.MOUNTAIN): "Mountain",
        int(TerrainType.FOREST): "Forest",
        int(TerrainType.STONE_DEPOSIT): "Stone site",
        int(TerrainType.FIELD): "Field",
    }.get(t, "?")


def _building_abbr(bt: BuildingType) -> str:
    return {
        BuildingType.TOWNHALL: "TH",
        BuildingType.BARRACKS: "Bx",
        BuildingType.STORAGE: "St",
        BuildingType.FARM: "Fm",
        BuildingType.TOWER: "Tw",
        BuildingType.WALL: "Wl",
        BuildingType.CITADEL: "Ct",
    }.get(bt, "?")


def _resource_corner_char(layer: int) -> str | None:
    """Small map hint for harvest overlay (not terrain name)."""
    return {
        int(ResourceLayer.NONE): None,
        int(ResourceLayer.FOREST): "w",
        int(ResourceLayer.STONE): "s",
        int(ResourceLayer.FIELD): "f",
    }.get(layer)


def _mode_label(m: GlobalRewardMode) -> str:
    return {
        GlobalRewardMode.NEUTRAL: "Neutral",
        GlobalRewardMode.DEFEND: "Defend",
        GlobalRewardMode.ATTACK: "Attack",
        GlobalRewardMode.GATHER: "Gather",
    }.get(m, "?")


def _lerp_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class PygameRenderer:
    """Draw terrain, units, buildings, coordinate axes, legend, and HUD."""

    def __init__(self, config: Mapping[str, Any], state: GameState | None) -> None:
        import pygame

        self._pygame = pygame
        self._config = config
        self._cell = int(config["rendering"]["cell_size"])
        self._n = int(config["map"]["size"])
        self._screen = None
        self._surface = None
        self._render_backend = "pygame"
        self._fonts_cell: int | None = None
        self._font_hint: Any = None
        self._font_building: Any = None
        self._font_role: Any = None
        self._font_ui_title: Any = None
        self._font_ui_body: Any = None
        self._font_ui_small: Any = None
        self._font_axis: Any = None
        pygame.init()

    def _ensure_map_fonts(self, cell: int) -> None:
        pygame = self._pygame
        if self._fonts_cell == cell:
            return
        self._fonts_cell = cell

        def _mono(size: int, bold: bool = False) -> Any:
            for name in ("consolas", "couriernew", "monospace"):
                try:
                    return pygame.font.SysFont(name, size, bold=bold)
                except OSError:
                    continue
            return pygame.font.SysFont("monospace", size, bold=bold)

        self._font_hint = _mono(max(9, cell // 3), bold=True) if cell >= 14 else None
        self._font_building = _mono(max(10, cell // 3), bold=True) if cell >= 18 else None
        self._font_role = _mono(max(8, cell // 4), bold=True) if cell >= 20 else None

    def _ensure_ui_fonts(self) -> None:
        pygame = self._pygame
        if self._font_ui_title is not None:
            return
        for name in ("consolas", "couriernew", "monospace"):
            try:
                self._font_ui_title = pygame.font.SysFont(name, 16, bold=True)
                self._font_ui_body = pygame.font.SysFont(name, 14)
                self._font_ui_small = pygame.font.SysFont(name, 12)
                self._font_axis = pygame.font.SysFont(name, 12)
                return
            except OSError:
                continue
        self._font_ui_title = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_ui_body = pygame.font.SysFont("monospace", 14)
        self._font_ui_small = pygame.font.SysFont("monospace", 12)
        self._font_axis = pygame.font.SysFont("monospace", 12)

    def _grid_px(self) -> tuple[int, int]:
        return self._cell * self._n, self._cell * self._n

    def render(
        self,
        state: GameState,
        mode: str,
        *,
        overlay_lines: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray | None:
        """Blit world; return RGB array when ``mode == \"rgb_array\"`` (map only)."""
        pygame = self._pygame
        gw, gh = self._grid_px()
        if self._surface is None or self._surface.get_size() != (gw, gh):
            self._surface = pygame.Surface((gw, gh))
            self._fonts_cell = None

        self._draw_map_grid(self._surface, state)

        if mode == "human":
            self._render_human_window(state, gw, gh, overlay_lines=overlay_lines or ())
            return None

        rgb = pygame.surfarray.array3d(self._surface)
        return np.transpose(rgb, (1, 0, 2)).astype(np.uint8)

    def _draw_map_grid(self, surf: Any, state: GameState) -> None:
        pygame = self._pygame
        n = self._n
        cell = self._cell
        self._ensure_map_fonts(cell)
        terrain = np.asarray(state.terrain, dtype=np.int32)
        res_layer = np.asarray(state.resources, dtype=np.int32)

        surf.fill(_DEFAULT_TERRAIN)
        for y in range(n):
            for x in range(n):
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                t = int(terrain[y, x])
                base = _TERRAIN_RGB.get(t, _DEFAULT_TERRAIN)
                hi = _TERRAIN_HIGHLIGHT.get(t, base)
                pygame.draw.rect(surf, base, rect)
                # Light top edge + left edge (simple “bevel”)
                if cell >= 6:
                    pygame.draw.line(surf, hi, rect.topleft, (rect.right - 1, rect.top), 1)
                    pygame.draw.line(surf, hi, rect.topleft, (rect.left, rect.bottom - 1), 1)
                pygame.draw.rect(surf, _GRID_LINE, rect, 1)
                hint = _resource_corner_char(int(res_layer[y, x]))
                if hint and self._font_hint is not None:
                    txt = self._font_hint.render(hint, True, (255, 252, 220))
                    shadow = self._font_hint.render(hint, True, (20, 18, 10))
                    surf.blit(shadow, (rect.right - txt.get_width(), rect.bottom - txt.get_height() - 1))
                    surf.blit(
                        txt,
                        (rect.right - txt.get_width() - 1, rect.bottom - txt.get_height() - 2),
                    )

        def team_color(team: int) -> tuple[int, int, int]:
            return _TEAM_FILL[team] if team < len(_TEAM_FILL) else (160, 160, 160)

        def team_dark(team: int) -> tuple[int, int, int]:
            return _TEAM_DARK[team] if team < len(_TEAM_DARK) else (80, 80, 80)

        for v in state.villages:
            for b in v.buildings:
                if b.hp <= 0:
                    continue
                cx, cy = b.position
                pad = max(2, cell // 10)
                rect = pygame.Rect(cx * cell + pad, cy * cell + pad, cell - 2 * pad, cell - 2 * pad)
                tc = team_color(v.team)
                td = team_dark(v.team)
                # “Shadow” tile under building
                sh = rect.move(1, 2)
                pygame.draw.rect(surf, (12, 14, 18), sh)
                pygame.draw.rect(surf, td, rect.inflate(2, 2))
                pygame.draw.rect(surf, tc, rect)
                pygame.draw.rect(surf, (12, 14, 20), rect, 1)
                pygame.draw.rect(surf, _lerp_rgb(tc, (255, 255, 255), 0.35), rect, 1)

                frac = b.hp / max(b.max_hp, 1)
                bar_y = rect.top - 4
                bar_w = rect.width
                pygame.draw.rect(surf, _HP_BG, pygame.Rect(rect.left, bar_y, bar_w, 4))
                hp_color = _HP_OK if frac > 0.35 else _HP_LOW
                pygame.draw.rect(
                    surf,
                    hp_color,
                    pygame.Rect(rect.left, bar_y, max(0, int(bar_w * frac)), 4),
                )
                pygame.draw.rect(surf, (0, 0, 0), pygame.Rect(rect.left, bar_y, bar_w, 4), 1)

                if self._font_building is not None:
                    abbr = _building_abbr(b.building_type)
                    label = self._font_building.render(abbr, True, (248, 250, 255))
                    outline = self._font_building.render(abbr, True, (16, 18, 24))
                    lx = rect.centerx - label.get_width() // 2
                    ly = rect.centery - label.get_height() // 2
                    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        surf.blit(outline, (lx + ox, ly + oy))
                    surf.blit(label, (lx, ly))

        for v in state.villages:
            for bot in v.bots:
                if not bot.is_alive:
                    continue
                bx, by = bot.position
                cx = bx * cell + cell // 2
                cy = by * cell + cell // 2
                r = max(cell // 4, 4)
                role_col = _ROLE_FILL.get(bot.role, (200, 200, 200))
                team_col = _TEAM_FILL[v.team] if v.team < len(_TEAM_FILL) else (160, 160, 160)
                skin = (228, 198, 168)
                head_r = max(3, int(r * 0.44))
                body_w = max(6, int(r * 1.65))
                body_h = max(5, int(r * 1.05))
                foot_y = cy + max(2, r // 3)
                body = pygame.Rect(cx - body_w // 2, foot_y - body_h, body_w, body_h)
                hc_y = foot_y - body_h - head_r - 1
                frac = bot.hp / max(bot.max_hp, 1)
                bar_w = min(cell - 4, max(body_w + 4, 18))
                bar_h = max(3, min(5, cell // 6))
                bar_y = hc_y - head_r - bar_h - 3
                bar_x = cx - bar_w // 2
                pygame.draw.rect(surf, _HP_BG, pygame.Rect(bar_x, bar_y, bar_w, bar_h))
                hp_color = _HP_OK if frac > 0.35 else _HP_LOW
                pygame.draw.rect(
                    surf,
                    hp_color,
                    pygame.Rect(bar_x, bar_y, max(0, int(bar_w * frac)), bar_h),
                )
                pygame.draw.rect(surf, (0, 0, 0), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 1)
                if self._font_hint is not None and bar_w >= 14:
                    hs = f"{bot.hp}"
                    ht = self._font_hint.render(hs, True, (248, 250, 255))
                    ho = self._font_hint.render(hs, True, (12, 14, 18))
                    tx = cx - ht.get_width() // 2
                    ty = bar_y + (bar_h - ht.get_height()) // 2
                    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        surf.blit(ho, (tx + ox, ty + oy))
                    surf.blit(ht, (tx, ty))
                pygame.draw.ellipse(surf, (18, 20, 26), body.move(1, 2))
                pygame.draw.ellipse(surf, team_col, body)
                pygame.draw.ellipse(surf, team_dark(v.team), body, 2)
                pygame.draw.ellipse(
                    surf, _lerp_rgb(team_col, (255, 255, 255), 0.35), body, 1
                )
                pygame.draw.circle(surf, (14, 16, 20), (cx + 1, hc_y + 2), head_r + 1)
                pygame.draw.circle(surf, skin, (cx, hc_y), head_r)
                pygame.draw.circle(surf, team_dark(v.team), (cx, hc_y), head_r, 2)
                hw = max(4, int(head_r * 1.9))
                hh = max(3, int(head_r * 0.88))
                helmet = pygame.Rect(cx - hw // 2, hc_y - head_r - 1, hw, hh)
                pygame.draw.ellipse(surf, role_col, helmet)
                pygame.draw.ellipse(
                    surf, _lerp_rgb(role_col, (12, 14, 18), 0.5), helmet, 1
                )
                if self._font_role is not None:
                    rn = bot.role.name[:1]
                    t = self._font_role.render(rn, True, (248, 250, 255))
                    to = self._font_role.render(rn, True, (16, 18, 22))
                    bx = cx - t.get_width() // 2
                    by = foot_y - body_h // 2 - t.get_height() // 2
                    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        surf.blit(to, (bx + ox, by + oy))
                    surf.blit(t, (bx, by))

    def _render_human_window(
        self,
        state: GameState,
        gw: int,
        gh: int,
        *,
        overlay_lines: tuple[str, ...] = (),
    ) -> None:
        pygame = self._pygame
        self._ensure_ui_fonts()
        map_block_h = _MARGIN_TOP + gh
        content_h = max(map_block_h, _MIN_LEGEND_BLOCK_H)
        win_w = _MARGIN_LEFT + gw + _LEGEND_PAD + _LEGEND_WIDTH
        win_h = _TITLE_BAR + content_h + _BOTTOM_HUD_HEIGHT
        if self._screen is None:
            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Village AI War")

        screen = self._screen
        screen.fill(_BG_DEEP)
        pygame.draw.rect(screen, _ACCENT, pygame.Rect(0, 0, win_w, 3))
        title_txt = "Village AI War  ·  x = column →  ·  y = row ↓"
        screen.blit(self._font_ui_title.render(title_txt, True, _TITLE_FG), (10, 6))

        ox, oy = _MARGIN_LEFT, _TITLE_BAR + _MARGIN_TOP
        legend_x = ox + gw + _LEGEND_PAD

        # Map drop shadow + frame
        shadow_off = 3
        shadow_surf = pygame.Surface((gw + 4, gh + 4))
        shadow_surf.set_alpha(70)
        shadow_surf.fill((0, 0, 0))
        screen.blit(shadow_surf, (ox + shadow_off, oy + shadow_off))
        pygame.draw.rect(screen, _MAP_FRAME, pygame.Rect(ox - 2, oy - 2, gw + 4, gh + 4), 2)
        screen.blit(self._surface, (ox, oy))

        if content_h > map_block_h:
            gap_rect = pygame.Rect(0, _TITLE_BAR + map_block_h, legend_x, content_h - map_block_h)
            pygame.draw.rect(screen, _BG_DEEP, gap_rect)

        self._draw_coordinate_axes(screen, ox, oy, gw, gh)
        self._draw_legend_panel(screen, legend_x, _TITLE_BAR, _LEGEND_WIDTH, content_h)
        if overlay_lines:
            self._draw_overlay_lines(screen, legend_x + 8, _TITLE_BAR + 420, overlay_lines)
        self._draw_bottom_hud(screen, state, 0, _TITLE_BAR + content_h, win_w)
        pygame.display.flip()
        pygame.event.pump()
        pygame.time.delay(int(1000 / max(int(self._config["rendering"]["fps"]), 1)))

    def _draw_overlay_lines(
        self,
        screen: Any,
        x: int,
        y0: int,
        lines: tuple[str, ...],
    ) -> None:
        pygame = self._pygame
        small = self._font_ui_small
        yy = y0
        panel_w = _LEGEND_WIDTH - 16
        line_h = 16
        h = min(len(lines) * line_h + 8, 140)
        if y0 + h > screen.get_height() - _BOTTOM_HUD_HEIGHT - 8:
            yy = max(_TITLE_BAR + 8, screen.get_height() - _BOTTOM_HUD_HEIGHT - h - 8)
        rect = pygame.Rect(x, yy, panel_w, h)
        s = pygame.Surface((rect.width, rect.height))
        s.set_alpha(230)
        s.fill((18, 22, 32))
        screen.blit(s, rect.topleft)
        pygame.draw.rect(screen, _ACCENT, rect, 1)
        ty = yy + 4
        for line in lines[:8]:
            screen.blit(small.render(line[:72], True, (230, 232, 240)), (x + 6, ty))
            ty += line_h

    def _draw_coordinate_axes(self, screen: Any, ox: int, oy: int, gw: int, gh: int) -> None:
        pygame = self._pygame
        font = self._font_axis
        fg = (190, 195, 210)
        dim = (100, 105, 120)
        n = self._n
        cell = self._cell
        for x in range(n):
            label = font.render(str(x), True, fg)
            tx = ox + x * cell + cell // 2 - label.get_width() // 2
            pygame.draw.line(screen, dim, (tx + label.get_width() // 2, oy - 2), (tx + label.get_width() // 2, oy), 1)
            screen.blit(label, (tx, oy - 18))
        for y in range(n):
            label = font.render(str(y), True, fg)
            ty = oy + y * cell + cell // 2 - label.get_height() // 2
            pygame.draw.line(screen, dim, (ox - 2, ty + label.get_height() // 2), (ox, ty + label.get_height() // 2), 1)
            screen.blit(label, (ox - 22, ty))

    def _draw_legend_panel(self, screen: Any, x: int, y: int, w: int, h: int) -> None:
        pygame = self._pygame
        fg = (220, 222, 232)
        muted = (150, 155, 170)
        body = self._font_ui_body
        small = self._font_ui_small
        bold = self._font_ui_title

        pygame.draw.rect(screen, _BG_PANEL, pygame.Rect(x, y, w, h))
        pygame.draw.rect(screen, _BG_PANEL_EDGE, pygame.Rect(x, y, w, h), 1)
        pygame.draw.rect(screen, _ACCENT, pygame.Rect(x, y, w, 3))

        cy = y + 10
        line_h = 18

        def line(text: str, color: tuple[int, int, int] = fg, fnt: Any = body) -> None:
            nonlocal cy
            screen.blit(fnt.render(text, True, color), (x + 12, cy))
            cy += line_h

        screen.blit(bold.render("Legend", True, (255, 255, 255)), (x + 12, cy))
        cy += line_h + 6

        screen.blit(small.render("TERRAIN", True, _ACCENT), (x + 12, cy))
        cy += line_h - 2
        for tt in (
            TerrainType.GRASS,
            TerrainType.MOUNTAIN,
            TerrainType.FOREST,
            TerrainType.STONE_DEPOSIT,
            TerrainType.FIELD,
        ):
            col = _TERRAIN_RGB.get(int(tt), _DEFAULT_TERRAIN)
            swatch = pygame.Rect(x + 12, cy + 2, 18, 14)
            pygame.draw.rect(screen, col, swatch)
            pygame.draw.rect(screen, _GRID_LINE, swatch, 1)
            screen.blit(body.render(_terrain_label(int(tt)), True, fg), (x + 36, cy))
            cy += line_h
        cy += 6

        screen.blit(small.render("HARVEST HINT (corner)", True, _ACCENT), (x + 12, cy))
        cy += line_h - 2
        line("w wood   s stone   f farm field", muted, small)
        cy += 6

        screen.blit(small.render("UNITS", True, _ACCENT), (x + 12, cy))
        cy += line_h - 2
        line("Swatch = role cap (torso = team)", muted, small)
        for role, name in (
            (Role.WARRIOR, "Warrior"),
            (Role.GATHERER, "Gatherer"),
            (Role.FARMER, "Farmer"),
            (Role.BUILDER, "Builder"),
        ):
            col = _ROLE_FILL.get(role, (200, 200, 200))
            pygame.draw.circle(screen, col, (x + 21, cy + 8), 7)
            pygame.draw.circle(screen, (24, 26, 32), (x + 21, cy + 8), 7, 1)
            screen.blit(body.render(name, True, fg), (x + 36, cy))
            cy += line_h
        line("Torso = team; cap = role color", muted, small)
        line("Bar above unit = HP (number = current)", muted, small)
        cy += 6

        screen.blit(small.render("BUILDINGS", True, _ACCENT), (x + 12, cy))
        cy += line_h - 2
        line("Square fill = team; label = type", muted, small)
        line("TH Bx St Fm Tw Wl Ct", muted, small)
        line("Bar above = HP", muted, small)

    def _draw_bottom_hud(self, screen: Any, state: GameState, _x: int, y: int, full_w: int) -> None:
        pygame = self._pygame
        pygame.draw.rect(screen, _HUD_BAR_BG, pygame.Rect(0, y, full_w, _BOTTOM_HUD_HEIGHT))
        pygame.draw.line(screen, _ACCENT, (0, y), (full_w, y), 2)

        font = self._font_ui_body
        small = self._font_ui_small
        yy = y + 8
        tick = state.tick
        win = state.winner
        if win is None:
            win_s = "in progress"
        elif win == 0:
            win_s = "Red"
        elif win == 1:
            win_s = "Blue"
        else:
            win_s = "draw"

        head = f"Tick {tick} / {state.max_ticks}   ·   Match: {win_s}"
        screen.blit(font.render(head, True, _TITLE_FG), (12, yy))
        yy += 24

        row_h = 26
        for i, name in ((0, "RED"), (1, "BLUE")):
            col = _TEAM_HUD[i]
            v = state.villages[i]
            r = v.resources
            alive = sum(1 for b in v.bots if b.is_alive)
            mode = _mode_label(v.global_reward_mode)
            panel = pygame.Rect(8, yy, full_w - 16, row_h)
            pygame.draw.rect(screen, _HUD_ROW_BG, panel)
            pygame.draw.rect(screen, _lerp_rgb(col, (255, 255, 255), 0.65), panel, 1)
            pygame.draw.rect(screen, col, pygame.Rect(panel.left, panel.top, 4, panel.height))
            txt = (
                f"{name}   wood {r.wood}   stone {r.stone}   food {r.food}   "
                f"bots {alive}/{v.pop_cap}   AI mode: {mode}"
            )
            screen.blit(small.render(txt, True, col), (panel.left + 12, panel.top + 6))
            yy += row_h + 4

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
            self._screen = None
        self._surface = None
        self._fonts_cell = None
        self._font_ui_title = None
        self._font_ui_body = None
        self._font_ui_small = None
        self._font_axis = None
        self._pygame.quit()
