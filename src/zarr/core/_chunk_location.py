"""Stored locations of logical chunks, separate from decoded array indexing."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, cast

import numpy as np

from zarr.codecs.sharding import MAX_UINT_64, ShardingCodec
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

if TYPE_CHECKING:
    from typing import SupportsIndex

    from zarr.types import AnyAsyncArray


async def chunk_byte_range(
    array: AnyAsyncArray, coords: tuple[SupportsIndex, ...]
) -> tuple[str, int, int] | None:
    """Resolve an encoded location for a supported layout, without payload decoding."""
    metadata = array.metadata
    shard = None
    if isinstance(metadata, ArrayV3Metadata):
        if not isinstance(metadata.chunk_grid, RegularChunkGridMetadata):
            raise NotImplementedError("chunk_byte_range requires a regular chunk grid")
        if len(metadata.codecs) == 1 and isinstance(metadata.codecs[0], ShardingCodec):
            shard = metadata.codecs[0]
            if any(isinstance(codec, ShardingCodec) for codec in shard.codecs):
                raise NotImplementedError("chunk_byte_range does not support nested sharding")
        elif any(isinstance(codec, ShardingCodec) for codec in metadata.codecs):
            raise NotImplementedError("chunk_byte_range requires sharding without outer codecs")

    if len(coords) != len(array.shape):
        raise ValueError("Expected one chunk coordinate per array dimension")
    if any(isinstance(coord, (bool, np.bool_)) for coord in coords):
        raise TypeError("Chunk coordinates must be integers, not booleans")
    normalized = tuple(operator.index(coord) for coord in coords)
    chunk_shape = shard.chunk_shape if shard is not None else array.chunks
    if any(
        coord < 0 or coord * width >= size
        for coord, width, size in zip(normalized, chunk_shape, array.shape, strict=True)
    ):
        raise IndexError(f"Chunk coordinates {coords} are out of bounds")

    if shard is None:
        key = metadata.encode_chunk_key(normalized)
        path = array.store_path / key
        try:
            size = await path.store.getsize(path.path)
        except FileNotFoundError:
            return None
        return path.path, 0, size

    metadata = cast("ArrayV3Metadata", metadata)
    grid = cast("RegularChunkGridMetadata", metadata.chunk_grid)
    per_shard = tuple(
        size // width for size, width in zip(grid.chunk_shape, chunk_shape, strict=True)
    )
    outer = tuple(coord // count for coord, count in zip(normalized, per_shard, strict=True))
    path = array.store_path / metadata.encode_chunk_key(outer)
    index = await shard._load_shard_index_maybe(path, per_shard)
    if index is None:
        return None
    local = tuple(coord % count for coord, count in zip(normalized, per_shard, strict=True))
    offset, length = (int(value) for value in index.offsets_and_lengths[local])
    if offset == MAX_UINT_64 and length == MAX_UINT_64:
        return None
    if offset == MAX_UINT_64 or length == MAX_UINT_64 or offset + length > MAX_UINT_64:
        raise ValueError("Invalid byte range in shard index")
    return path.path, offset, length
