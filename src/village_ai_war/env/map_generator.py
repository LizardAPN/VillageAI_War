"""Symmetric map generation and initial ``GameState`` construction."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from village_ai_war.state import (
    BotState,
    BuildingState,
    BuildingType,
    GameState,
    ResourceLayer,
    ResourceStock,
    Role,
    TerrainType,
    VillageState,
)


def _building_type_from_name(name: str) -> BuildingType:
    mapping = {
        "townhall": BuildingType.TOWNHALL,
        "barracks": BuildingType.BARRACKS,
        "storage": BuildingType.STORAGE,
        "farm": BuildingType.FARM,
        "tower": BuildingType.TOWER,
        "wall": BuildingType.WALL,
        "citadel": BuildingType.CITADEL,
    }
    return mapping[name.lower()]


def _hp_for_type(bt: BuildingType, buildings_cfg: Mapping[str, Any]) -> int:
    key = BuildingType(bt).name.lower()
    if key == "townhall":
        return int(buildings_cfg["townhall"]["hp"])
    return int(buildings_cfg[key]["hp"])


def generate_initial_state(
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> GameState:
    """Build a new symmetric ``GameState`` from Hydra-style ``config``.

    Args:
        config: Merged config with at least ``map``, ``buildings``, ``game`` keys.
        rng: NumPy random generator for reproducible maps.

    Returns:
        Initialized ``GameState`` with two villages and mirrored terrain.
    """
    mcfg = config["map"]
    gcfg = config["game"]
    bcfg = config["buildings"]
    n = int(mcfg["size"])
    half = n // 2

    terrain = np.zeros((n, n), dtype=np.int32)
    resource_layer = np.zeros((n, n), dtype=np.int32)
    resource_amounts = np.zeros((n, n), dtype=np.int32)

    forest_cap = int(mcfg["resource_capacity"]["forest"])
    stone_cap = int(mcfg["resource_capacity"]["stone"])
    field_cap = int(mcfg["resource_capacity"]["field"])
    res_density = float(mcfg["resource_density"])
    mount_density = float(mcfg["mountain_density"])

    for y in range(n):
        for x in range(half):
            u = rng.random()
            if u < mount_density:
                terrain[y, x] = int(TerrainType.MOUNTAIN)
            elif u < mount_density + res_density * 0.4:
                terrain[y, x] = int(TerrainType.FOREST)
                resource_layer[y, x] = int(ResourceLayer.FOREST)
                resource_amounts[y, x] = forest_cap
            elif u < mount_density + res_density * 0.7:
                terrain[y, x] = int(TerrainType.STONE_DEPOSIT)
                resource_layer[y, x] = int(ResourceLayer.STONE)
                resource_amounts[y, x] = stone_cap
            elif u < mount_density + res_density:
                terrain[y, x] = int(TerrainType.FIELD)
                resource_layer[y, x] = int(ResourceLayer.FIELD)
                resource_amounts[y, x] = min(field_cap, 10_000)
            else:
                terrain[y, x] = int(TerrainType.GRASS)

    # Mirror left half to right (flip x)
    for y in range(n):
        for x in range(half):
            mx = n - 1 - x
            terrain[y, mx] = terrain[y, x]
            resource_layer[y, mx] = resource_layer[y, x]
            resource_amounts[y, mx] = resource_amounts[y, x]

    th0 = (2, n // 2)
    th1 = (n - 3, n // 2)
    for pos in (th0, th1):
        terrain[pos[1], pos[0]] = int(TerrainType.GRASS)
        resource_layer[pos[1], pos[0]] = int(ResourceLayer.NONE)
        resource_amounts[pos[1], pos[0]] = 0

    max_ticks = int(gcfg["max_ticks"])
    init_res = gcfg["initial_resources"]
    n_bots = int(gcfg["initial_bots"])
    init_building_names: list[str] = list(gcfg["initial_buildings"])

    villages: list[VillageState] = []
    next_bid = 0
    next_building_id = 0

    for team, th_pos in enumerate((th0, th1)):
        stock = ResourceStock(
            wood=int(init_res["wood"]),
            stone=int(init_res["stone"]),
            food=int(init_res["food"]),
        )
        buildings: list[BuildingState] = []
        th = BuildingState(
            building_id=next_building_id,
            team=team,
            building_type=BuildingType.TOWNHALL,
            position=th_pos,
            hp=_hp_for_type(BuildingType.TOWNHALL, bcfg),
            max_hp=_hp_for_type(BuildingType.TOWNHALL, bcfg),
            is_under_construction=False,
            construction_progress=1.0,
        )
        next_building_id += 1
        buildings.append(th)

        offsets = [(3, 0), (0, 2), (0, -2), (4, 1)]
        for i, name in enumerate(init_building_names):
            ox, oy = offsets[i % len(offsets)]
            if team == 0:
                bx, by = th_pos[0] + ox, th_pos[1] + oy
            else:
                bx, by = th_pos[0] - ox, th_pos[1] + oy
            bx = int(np.clip(bx, 0, n - 1))
            by = int(np.clip(by, 0, n - 1))
            bt = _building_type_from_name(name)
            if terrain[by, bx] == int(TerrainType.MOUNTAIN):
                terrain[by, bx] = int(TerrainType.GRASS)
            resource_layer[by, bx] = int(ResourceLayer.NONE)
            resource_amounts[by, bx] = 0
            buildings.append(
                BuildingState(
                    building_id=next_building_id,
                    team=team,
                    building_type=bt,
                    position=(bx, by),
                    hp=_hp_for_type(bt, bcfg),
                    max_hp=_hp_for_type(bt, bcfg),
                    is_under_construction=False,
                    construction_progress=1.0,
                )
            )
            next_building_id += 1

        bots: list[BotState] = []
        roles_cycle = [Role.WARRIOR, Role.GATHERER, Role.FARMER, Role.BUILDER]
        for i in range(n_bots):
            angle = 2 * np.pi * i / max(n_bots, 1)
            r = 1 + (i % 3)
            bx = int(np.clip(th_pos[0] + int(r * np.cos(angle)), 0, n - 1))
            by = int(np.clip(th_pos[1] + int(r * np.sin(angle)), 0, n - 1))
            if terrain[by, bx] == int(TerrainType.MOUNTAIN):
                terrain[by, bx] = int(TerrainType.GRASS)
            role = roles_cycle[i % 4]
            stats = config["combat"]["stats"]
            role_key = Role(role).name.lower()
            hp = int(stats[role_key]["hp"])
            bots.append(
                BotState(
                    bot_id=next_bid,
                    team=team,
                    role=role,
                    position=(bx, by),
                    hp=hp,
                    max_hp=hp,
                )
            )
            next_bid += 1

        villages.append(
            VillageState(
                team=team,
                resources=stock,
                pop_cap=10,
                bots=bots,
                buildings=buildings,
            )
        )

    return GameState(
        tick=0,
        max_ticks=max_ticks,
        map_size=n,
        terrain=terrain.tolist(),
        resources=resource_layer.tolist(),
        resource_amounts=resource_amounts.tolist(),
        blueprints=[],
        villages=villages,
        is_done=False,
        winner=None,
        next_bot_id=next_bid,
        next_building_id=next_building_id,
    )
