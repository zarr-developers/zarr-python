from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Union
from dataclasses import asdict, dataclass, field
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import JSON, ChunkCoords, parse_name

if TYPE_CHECKING:
    from typing_extensions import Self


def parse_chunk_shape(data: JSON) -> ChunkCoords:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    if not all(isinstance(a, int) for a in data):
        raise TypeError(f"Expected an iterable of integers. Got {data} instead.")
    return tuple(data)


@dataclass(frozen=True)
class ChunkGrid(Metadata):

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        if isinstance(data, ChunkGrid):
            return data
        if data["name"] == "regular":
            return RegularChunkGrid.from_dict(data)
        raise ValueError(f"Unknown chunk grid, got {data['name']}")


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape) -> None:
        chunk_shape_parsed = parse_chunk_shape(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        parse_name(data["name"], "regular")
        return cls(**data["configuration"])

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": self.chunk_shape}}
