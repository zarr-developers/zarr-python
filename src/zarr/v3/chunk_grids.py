from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from dataclasses import dataclass
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import JSON, ChunkCoords, ChunkCoordsLike, parse_name, parse_shapelike

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class ChunkGrid(Metadata):
    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        if isinstance(data, ChunkGrid):
            return data
        if data["name"] == "regular":
            return RegularChunkGrid.from_dict(data)
        raise ValueError(f"Unknown chunk grid. Got {data['name']}.")


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape: ChunkCoordsLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        parse_name(data["name"], "regular")
        return cls(**data["configuration"])

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": self.chunk_shape}}
