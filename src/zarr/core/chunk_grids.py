from __future__ import annotations

import itertools
import math
import operator
from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    ChunkCoords,
    ChunkCoordsLike,
    ShapeLike,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.core.indexing import ceildiv

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self


def _guess_chunks(
    shape: ShapeLike,
    typesize: int,
    *,
    increment_bytes: int = 256 * 1024,
    min_bytes: int = 128 * 1024,
    max_bytes: int = 64 * 1024 * 1024,
) -> ChunkCoords:
    """
    Iteratively guess an appropriate chunk layout for an array, given its shape and
    the size of each element in bytes, and size constraints expressed in bytes. This logic is
    adapted from h5py.

    Parameters
    ----------
    shape: ChunkCoords
        The chunk shape.
    typesize: int
        The size, in bytes, of each element of the chunk.
    increment_bytes: int = 256 * 1024
        The number of bytes used to increment or decrement the target chunk size in bytes.
    min_bytes: int = 128 * 1024
        The soft lower bound on the final chunk size in bytes.
    max_bytes: int = 64 * 1024 * 1024
        The hard upper bound on the final chunk size in bytes.

    Returns
    -------
    ChunkCoords

    """
    if isinstance(shape, int):
        shape = (shape,)

    ndims = len(shape)
    # require chunks to have non-zero length for all dimensions
    chunks = np.maximum(np.array(shape, dtype="=f8"), 1)

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.prod(chunks) * typesize
    target_size = increment_bytes * (2 ** np.log10(dset_size / (1024.0 * 1024)))

    if target_size > max_bytes:
        target_size = max_bytes
    elif target_size < min_bytes:
        target_size = min_bytes

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        # 2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.prod(chunks) * typesize

        if (
            chunk_bytes < target_size or abs(chunk_bytes - target_size) / target_size < 0.5
        ) and chunk_bytes < max_bytes:
            break

        if np.prod(chunks) == 1:
            break  # Element size larger than max_bytes

        chunks[idx % ndims] = math.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)


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
        return {"name": "regular", "configuration": {"chunk_shape": tuple(self.chunk_shape)}}

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
