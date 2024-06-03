from __future__ import annotations

import itertools
import operator
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

from zarr.abc.metadata import Metadata
from zarr.common import (
    JSON,
    ChunkCoords,
    ChunkCoordsLike,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.indexing import ceildiv

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class ChunkGrid(Metadata):
    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkGrid) -> ChunkGrid:
        if isinstance(data, ChunkGrid):
            return data

        name_parsed, _ = parse_named_configuration(data)
        if name_parsed == "regular":
            return RegularChunkGrid._from_dict(data)
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.")

    @abstractmethod
    def all_chunk_coords(self, array_shape: ChunkCoords) -> Iterator[ChunkCoords]:
        pass

    @abstractmethod
    def get_nchunks(self, array_shape: ChunkCoords) -> int:
        pass


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape: ChunkCoordsLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def _from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": list(self.chunk_shape)}}

    def all_chunk_coords(self, array_shape: ChunkCoords) -> Iterator[ChunkCoords]:
        return itertools.product(
            *(range(0, ceildiv(s, c)) for s, c in zip(array_shape, self.chunk_shape, strict=False))
        )

    def get_nchunks(self, array_shape: ChunkCoords) -> int:
        return reduce(
            operator.mul,
            (ceildiv(s, c) for s, c in zip(array_shape, self.chunk_shape, strict=True)),
            1,
        )
