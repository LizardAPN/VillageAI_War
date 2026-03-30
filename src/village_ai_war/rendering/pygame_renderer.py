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

# Layout for human mode: map + legend + HUD (rgb_array stays map-only).
_MARGIN_LEFT = 26
_MARGIN_TOP = 22
_LEGEND_WIDTH = 268
_LEGEND_PAD = 10
_BOTTOM_HUD_HEIGHT = 88
_TITLE_BAR = 22
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
        pygame.init()

    def _grid_px(self) -> tuple[int, int]:
        return self._cell * self._n, self._cell * self._n

    def render(self, state: GameState, mode: str) -> np.ndarray | None:
        """Blit world; return RGB array when ``mode == \"rgb_array\"`` (map only)."""
        pygame = self._pygame
        gw, gh = self._grid_px()
        if self._surface is None or self._surface.get_size() != (gw, gh):
            self._surface = pygame.Surface((gw, gh))

        self._draw_map_grid(self._surface, state)

        if mode == "human":
            self._render_human_window(state, gw, gh)
            return None

        rgb = pygame.surfarray.array3d(self._surface)
        return np.transpose(rgb, (1, 0, 2)).astype(np.uint8)

    def _draw_map_grid(self, surf: Any, state: GameState) -> None:
        pygame = self._pygame
        n = self._n
        cell = self._cell
        terrain = np.asarray(state.terrain, dtype=np.int32)
        res_layer = np.asarray(state.resources, dtype=np.int32)

        surf.fill((24, 28, 36))
        for y in range(n):
            for x in range(n):
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                t = int(terrain[y, x])
                color = {
                    int(TerrainType.GRASS): (60, 110, 70),
                    int(TerrainType.MOUNTAIN): (90, 90, 100),
                    int(TerrainType.FOREST): (30, 80, 40),
                    int(TerrainType.STONE_DEPOSIT): (120, 120, 130),
                    int(TerrainType.FIELD): (140, 130, 70),
                }.get(t, (50, 50, 50))
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (0, 0, 0), rect, 1)
                hint = _resource_corner_char(int(res_layer[y, x]))
                if hint and cell >= 14:
                    font_tiny = pygame.font.SysFont("monospace", max(9, cell // 3))
                    txt = font_tiny.render(hint, True, (255, 255, 200))
                    surf.blit(
                        txt,
                        (rect.right - txt.get_width() - 1, rect.bottom - txt.get_height() - 1),
                    )

        def team_color(team: int) -> tuple[int, int, int]:
            return (180, 60, 60) if team == 0 else (60, 90, 200)

        for v in state.villages:
            for b in v.buildings:
                if b.hp <= 0:
                    continue
                cx, cy = b.position
                rect = pygame.Rect(cx * cell + 2, cy * cell + 2, cell - 4, cell - 4)
                pygame.draw.rect(surf, team_color(v.team), rect)
                frac = b.hp / max(b.max_hp, 1)
                bar = pygame.Rect(rect.left, rect.top - 3, int(rect.width * frac), 2)
                pygame.draw.rect(surf, (0, 220, 80), bar)
                if cell >= 18:
                    font_b = pygame.font.SysFont("monospace", max(10, cell // 3))
                    abbr = _building_abbr(b.building_type)
                    label = font_b.render(abbr, True, (255, 255, 255))
                    lx = rect.centerx - label.get_width() // 2
                    ly = rect.centery - label.get_height() // 2
                    surf.blit(label, (lx, ly))

        role_colors = {
            Role.WARRIOR: (220, 80, 80),
            Role.GATHERER: (220, 200, 80),
            Role.FARMER: (120, 220, 120),
            Role.BUILDER: (180, 140, 220),
        }
        for v in state.villages:
            for bot in v.bots:
                if not bot.is_alive:
                    continue
                bx, by = bot.position
                cx = bx * cell + cell // 2
                cy = by * cell + cell // 2
                r = max(cell // 5, 3)
                pygame.draw.circle(surf, role_colors.get(bot.role, (200, 200, 200)), (cx, cy), r)
                pygame.draw.circle(surf, team_color(v.team), (cx, cy), r, 2)
                if cell >= 22:
                    font_r = pygame.font.SysFont("monospace", 9)
                    rn = bot.role.name[:1]
                    t = font_r.render(rn, True, (20, 20, 20))
                    surf.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))

    def _render_human_window(self, state: GameState, gw: int, gh: int) -> None:
        pygame = self._pygame
        map_block_h = _MARGIN_TOP + gh
        content_h = max(map_block_h, _MIN_LEGEND_BLOCK_H)
        win_w = _MARGIN_LEFT + gw + _LEGEND_PAD + _LEGEND_WIDTH
        win_h = _TITLE_BAR + content_h + _BOTTOM_HUD_HEIGHT
        if self._screen is None:
            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Village AI War — map + legend")

        screen = self._screen
        screen.fill((18, 20, 26))
        title_font = pygame.font.SysFont("monospace", 15)
        title_txt = "Village AI War — x=column (right), y=row (down)"
        screen.blit(title_font.render(title_txt, True, (190, 195, 210)), (8, 3))
        ox, oy = _MARGIN_LEFT, _TITLE_BAR + _MARGIN_TOP
        legend_x = ox + gw + _LEGEND_PAD
        screen.blit(self._surface, (ox, oy))
        if content_h > map_block_h:
            gap_rect = pygame.Rect(0, _TITLE_BAR + map_block_h, legend_x, content_h - map_block_h)
            pygame.draw.rect(screen, (18, 20, 26), gap_rect)
        self._draw_coordinate_axes(screen, ox, oy, gw, gh)
        self._draw_legend_panel(screen, legend_x, _TITLE_BAR, _LEGEND_WIDTH, content_h)
        self._draw_bottom_hud(screen, state, 0, _TITLE_BAR + content_h, win_w)
        pygame.display.flip()
        pygame.event.pump()
        pygame.time.delay(int(1000 / max(int(self._config["rendering"]["fps"]), 1)))

    def _draw_coordinate_axes(self, screen: Any, ox: int, oy: int, gw: int, gh: int) -> None:
        pygame = self._pygame
        font = pygame.font.SysFont("monospace", 12)
        fg = (200, 200, 210)
        n = self._n
        cell = self._cell
        for x in range(n):
            label = font.render(str(x), True, fg)
            screen.blit(label, (ox + x * cell + cell // 2 - label.get_width() // 2, oy - 16))
        for y in range(n):
            label = font.render(str(y), True, fg)
            screen.blit(label, (ox - 20, oy + y * cell + cell // 2 - label.get_height() // 2))

    def _draw_legend_panel(self, screen: Any, x: int, y: int, w: int, h: int) -> None:
        pygame = self._pygame
        fg = (220, 220, 230)
        pygame.draw.rect(screen, (32, 36, 44), pygame.Rect(x, y, w, h))
        pygame.draw.rect(screen, (70, 75, 90), pygame.Rect(x, y, w, h), 1)
        font = pygame.font.SysFont("monospace", 13)
        try:
            font_bold = pygame.font.SysFont("monospace", 14, bold=True)
        except TypeError:
            font_bold = font
        cy = y + 8
        line_h = 16

        def line(text: str, color: tuple[int, int, int] = (220, 220, 230)) -> None:
            nonlocal cy
            screen.blit(font.render(text, True, color), (x + 8, cy))
            cy += line_h

        screen.blit(font_bold.render("Legend", True, (255, 255, 255)), (x + 8, cy))
        cy += line_h + 4
        screen.blit(font_bold.render("Terrain (tile fill)", True, (180, 190, 210)), (x + 8, cy))
        cy += line_h
        samples = [
            (TerrainType.GRASS, (60, 110, 70)),
            (TerrainType.MOUNTAIN, (90, 90, 100)),
            (TerrainType.FOREST, (30, 80, 40)),
            (TerrainType.STONE_DEPOSIT, (120, 120, 130)),
            (TerrainType.FIELD, (140, 130, 70)),
        ]
        for tt, col in samples:
            pygame.draw.rect(screen, col, pygame.Rect(x + 8, cy, 14, 12))
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x + 8, cy, 14, 12), 1)
            screen.blit(font.render(_terrain_label(int(tt)), True, fg), (x + 28, cy - 1))
            cy += line_h
        cy += 4
        screen.blit(font_bold.render("Harvest hint (corner)", True, (180, 190, 210)), (x + 8, cy))
        cy += line_h
        line("w = wood (forest cap)", (200, 200, 160))
        line("s = stone", (200, 200, 160))
        line("f = farm field", (200, 200, 160))
        cy += 4
        screen.blit(font_bold.render("Units (disk)", True, (180, 190, 210)), (x + 8, cy))
        cy += line_h
        role_rows = [
            (Role.WARRIOR, (220, 80, 80), "Warrior"),
            (Role.GATHERER, (220, 200, 80), "Gatherer"),
            (Role.FARMER, (120, 220, 120), "Farmer"),
            (Role.BUILDER, (180, 140, 220), "Builder"),
        ]
        for role, col, name in role_rows:
            pygame.draw.circle(screen, col, (x + 15, cy + 6), 6)
            pygame.draw.circle(screen, (40, 40, 40), (x + 15, cy + 6), 6, 1)
            screen.blit(font.render(name, True, fg), (x + 28, cy))
            cy += line_h
        line("Ring: Red=left team, Blue=right", (160, 170, 190))
        cy += 4
        screen.blit(font_bold.render("Buildings (square)", True, (180, 190, 210)), (x + 8, cy))
        cy += line_h
        line("Fill color = team; label = type", (160, 170, 190))
        line("TH Townhall  Bx Barracks  St Storage", (140, 150, 170))
        line("Fm Farm  Tw Tower  Wl Wall  Ct Citadel", (140, 150, 170))
        line("Green bar above = HP fraction", (140, 150, 170))

    def _draw_bottom_hud(self, screen: Any, state: GameState, _x: int, y: int, full_w: int) -> None:
        pygame = self._pygame
        pygame.draw.rect(screen, (28, 30, 38), pygame.Rect(0, y, full_w, _BOTTOM_HUD_HEIGHT))
        pygame.draw.line(screen, (60, 65, 80), (0, y), (full_w, y), 1)
        font = pygame.font.SysFont("monospace", 13)
        yy = y + 6
        tick = state.tick
        win = state.winner
        if win is None:
            win_s = "ongoing"
        elif win == 0:
            win_s = "Red"
        elif win == 1:
            win_s = "Blue"
        else:
            win_s = "draw"
        screen.blit(
            font.render(f"Tick {tick}/{state.max_ticks}   Winner: {win_s}", True, (240, 240, 250)),
            (8, yy),
        )
        yy += 20
        for i, name, col in (
            (0, "RED", (255, 120, 120)),
            (1, "BLUE", (140, 170, 255)),
        ):
            v = state.villages[i]
            r = v.resources
            alive = sum(1 for b in v.bots if b.is_alive)
            mode = _mode_label(v.global_reward_mode)
            txt = (
                f"{name}  wood={r.wood} stone={r.stone} food={r.food}  "
                f"bots_alive={alive}/{v.pop_cap}  mode={mode}"
            )
            screen.blit(font.render(txt, True, col), (8, yy))
            yy += 20

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
            self._screen = None
        self._surface = None
        self._pygame.quit()
