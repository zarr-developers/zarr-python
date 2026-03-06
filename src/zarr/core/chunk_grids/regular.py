from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Self

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


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    _array_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]

    def __init__(self, *, chunk_shape: ShapeLike, array_shape: ShapeLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        array_shape_parsed = parse_shapelike(array_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)
        object.__setattr__(self, "_array_shape", array_shape_parsed)

    @property
    def array_shape(self) -> tuple[int, ...]:
        return self._array_shape

    @classmethod
    def from_dict(  # type: ignore[override]
        cls, data: dict[str, JSON] | NamedConfig[str, Any], *, array_shape: ShapeLike
    ) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed, array_shape=array_shape)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": tuple(self.chunk_shape)}}

    def update_shape(self, new_shape: tuple[int, ...]) -> Self:
        return type(self)(chunk_shape=self.chunk_shape, array_shape=new_shape)

    def all_chunk_coords(self) -> Iterator[tuple[int, ...]]:
        return itertools.product(
            *(
                range(ceildiv(s, c))
                for s, c in zip(self._array_shape, self.chunk_shape, strict=False)
            )
        )

    def get_nchunks(self) -> int:
        return reduce(
            operator.mul,
            itertools.starmap(ceildiv, zip(self._array_shape, self.chunk_shape, strict=True)),
            1,
        )

    def get_chunk_shape(self, chunk_coord: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the shape of a specific chunk.

        For RegularChunkGrid, all chunks have the same shape except possibly
        the last chunk in each dimension.
        """
        return tuple(
            int(
                min(
                    self.chunk_shape[i],
                    self._array_shape[i] - chunk_coord[i] * self.chunk_shape[i],
                )
            )
            for i in range(len(self._array_shape))
        )

    def get_chunk_start(self, chunk_coord: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the starting position of a chunk in the array.

        For RegularChunkGrid, this is simply chunk_coord * chunk_shape.
        """
        return tuple(
            coord * size for coord, size in zip(chunk_coord, self.chunk_shape, strict=False)
        )

    def array_index_to_chunk_coord(self, array_index: tuple[int, ...]) -> tuple[int, ...]:
        """
        Map an array index to chunk coordinates.

        For RegularChunkGrid, this is simply array_index // chunk_shape.
        """
        return tuple(
            0 if size == 0 else idx // size
            for idx, size in zip(array_index, self.chunk_shape, strict=False)
        )

    def array_indices_to_chunk_dim(
        self, dim: int, indices: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        """
        Vectorized mapping of array indices to chunk coordinates along one dimension.

        For RegularChunkGrid, this is simply indices // chunk_size.
        """
        chunk_size = self.chunk_shape[dim]
        if chunk_size == 0:
            return np.zeros_like(indices)
        return indices // chunk_size

    def chunks_per_dim(self, dim: int) -> int:
        """
        Get the number of chunks along a specific dimension.

        For RegularChunkGrid, this is ceildiv(array_shape[dim], chunk_shape[dim]).
        """
        return ceildiv(self._array_shape[dim], self.chunk_shape[dim])

    def get_chunk_grid_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the chunk grid (number of chunks along each dimension).

        For RegularChunkGrid, this is computed using ceildiv for each dimension.
        """
        return tuple(
            ceildiv(array_len, chunk_len)
            for array_len, chunk_len in zip(self._array_shape, self.chunk_shape, strict=False)
        )
