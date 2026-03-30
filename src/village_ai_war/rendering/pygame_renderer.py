"""Pygame grid renderer for ``human`` and ``rgb_array`` modes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.state import GameState, Role, TerrainType


class PygameRenderer:
    """Draw terrain, units, buildings, and HUD resources."""

    def __init__(self, config: Mapping[str, Any], state: GameState | None) -> None:
        import pygame

        self._pygame = pygame
        self._config = config
        self._cell = int(config["rendering"]["cell_size"])
        self._n = int(config["map"]["size"])
        self._screen = None
        self._surface = None
        pygame.init()

    def render(self, state: GameState, mode: str) -> np.ndarray | None:
        """Blit world; return RGB array when ``mode == \"rgb_array\"``."""
        pygame = self._pygame
        w = h = self._cell * self._n
        if self._surface is None:
            self._surface = pygame.Surface((w, h))

        surf = self._surface
        surf.fill((24, 28, 36))
        terrain = np.asarray(state.terrain, dtype=np.int32)

        for y in range(self._n):
            for x in range(self._n):
                rect = pygame.Rect(x * self._cell, y * self._cell, self._cell, self._cell)
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

        def team_color(team: int) -> tuple[int, int, int]:
            return (180, 60, 60) if team == 0 else (60, 90, 200)

        for v in state.villages:
            for b in v.buildings:
                if b.hp <= 0:
                    continue
                cx, cy = b.position
                rect = pygame.Rect(
                    cx * self._cell + 2,
                    cy * self._cell + 2,
                    self._cell - 4,
                    self._cell - 4,
                )
                pygame.draw.rect(surf, team_color(v.team), rect)
                frac = b.hp / max(b.max_hp, 1)
                bar = pygame.Rect(rect.left, rect.top - 3, int(rect.width * frac), 2)
                pygame.draw.rect(surf, (0, 255, 0), bar)

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
                cx = bx * self._cell + self._cell // 2
                cy = by * self._cell + self._cell // 2
                pygame.draw.circle(surf, role_colors.get(bot.role, (200, 200, 200)), (cx, cy), self._cell // 4)
                pygame.draw.circle(surf, team_color(v.team), (cx, cy), self._cell // 4, 2)

        if mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode((w, h + 40))
            self._screen.fill((20, 20, 20))
            self._screen.blit(surf, (0, 0))
            font = pygame.font.SysFont("monospace", 14)
            hud0 = f"R wood={state.villages[0].resources.wood} food={state.villages[0].resources.food}"
            hud1 = f"B wood={state.villages[1].resources.wood} food={state.villages[1].resources.food}"
            self._screen.blit(font.render(hud0, True, (255, 255, 255)), (4, h + 4))
            self._screen.blit(font.render(hud1, True, (255, 255, 255)), (4, h + 20))
            pygame.display.flip()
            pygame.event.pump()
            self._pygame.time.delay(int(1000 / max(int(self._config["rendering"]["fps"]), 1)))
            return None

        rgb = pygame.surfarray.array3d(surf)
        return np.transpose(rgb, (1, 0, 2)).astype(np.uint8)

    def close(self) -> None:
        if self._screen is not None:
            self._pygame.display.quit()
            self._screen = None
        self._surface = None
        self._pygame.quit()
