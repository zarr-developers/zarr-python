from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr.core.chunk_grids.common import ChunkGrid
from zarr.core.common import (
    JSON,
    NamedConfig,
    ShapeLike,
    ceildiv,
    parse_named_configuration,
    parse_shapelike,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: tuple[int, ...]

    def __init__(self, *, chunk_shape: ShapeLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON] | NamedConfig[str, Any]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": tuple(self.chunk_shape)}}

    def update_shape(self, new_shape: tuple[int, ...]) -> Self:
        return self

    def all_chunk_coords(self, array_shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        return itertools.product(
            *(range(ceildiv(s, c)) for s, c in zip(array_shape, self.chunk_shape, strict=False))
        )

    def get_nchunks(self, array_shape: tuple[int, ...]) -> int:
        return reduce(
            operator.mul,
            itertools.starmap(ceildiv, zip(array_shape, self.chunk_shape, strict=True)),
            1,
        )

    def get_chunk_shape(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            int(min(self.chunk_shape[i], array_shape[i] - chunk_coord[i] * self.chunk_shape[i]))
            for i in range(len(array_shape))
        )

    def get_chunk_start(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            coord * size for coord, size in zip(chunk_coord, self.chunk_shape, strict=False)
        )

    def array_index_to_chunk_coord(
        self, array_shape: tuple[int, ...], array_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            0 if size == 0 else idx // size
            for idx, size in zip(array_index, self.chunk_shape, strict=False)
        )

    def array_indices_to_chunk_dim(
        self, array_shape: tuple[int, ...], dim: int, indices: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        chunk_size = self.chunk_shape[dim]
        if chunk_size == 0:
            return np.zeros_like(indices)
        return indices // chunk_size

    def chunks_per_dim(self, array_shape: tuple[int, ...], dim: int) -> int:
        return ceildiv(array_shape[dim], self.chunk_shape[dim])

    def get_chunk_grid_shape(self, array_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            ceildiv(array_len, chunk_len)
            for array_len, chunk_len in zip(array_shape, self.chunk_shape, strict=False)
        )
