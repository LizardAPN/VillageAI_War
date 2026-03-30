"""Terrain and resource grid encodings (integer enums)."""

from enum import IntEnum


class TerrainType(IntEnum):
    """Cell terrain classification for the map grid."""

    GRASS = 0
    MOUNTAIN = 1
    FOREST = 2
    STONE_DEPOSIT = 3
    FIELD = 4


class ResourceLayer(IntEnum):
    """Resource capacity overlay; 0 means no harvestable resource on cell."""

    NONE = 0
    FOREST = 1
    STONE = 2
    FIELD = 3
