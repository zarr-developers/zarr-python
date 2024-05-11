from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zarr.abc.metadata import Metadata
from zarr.common import (
    JSON,
    ChunkCoords,
    ChunkCoordsLike,
    parse_named_configuration,
    parse_shapelike,
)

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class ChunkGrid(Metadata):
    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> ChunkGrid:
        if isinstance(data, ChunkGrid):
            return data

        name_parsed, _ = parse_named_configuration(data)
        if name_parsed == "regular":
            return RegularChunkGrid.from_dict(data)
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.")


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape: ChunkCoordsLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": list(self.chunk_shape)}}
