from __future__ import annotations

from typing import Any

from zarr.core.chunk_grids.common import (
    ChunkGrid,
    _auto_partition,
    _guess_chunks,
    _guess_num_chunks_per_axis_shard,
    normalize_chunks,
)
from zarr.core.chunk_grids.regular import RegularChunkGrid
from zarr.core.common import JSON, NamedConfig, parse_named_configuration
from zarr.registry import get_chunk_grid_class, register_chunk_grid

register_chunk_grid("regular", RegularChunkGrid)


def parse_chunk_grid(
    data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any],
    *,
    array_shape: tuple[int, ...],
) -> ChunkGrid:
    """Parse a chunk grid from a dictionary, returning existing ChunkGrid instances as-is.

    Uses the chunk grid registry to look up the appropriate class by name.

    Parameters
    ----------
    data : dict[str, JSON] | ChunkGrid | NamedConfig[str, Any]
        Either a ChunkGrid instance (returned as-is) or a dictionary with
        'name' and 'configuration' keys.
    array_shape : tuple[int, ...]
        The shape of the array this chunk grid is bound to.

    Returns
    -------
    ChunkGrid

    Raises
    ------
    ValueError
        If the chunk grid name is not found in the registry.
    """
    if isinstance(data, ChunkGrid):
        return data

    name_parsed, _ = parse_named_configuration(data)
    try:
        chunk_grid_cls = get_chunk_grid_class(name_parsed)
    except KeyError as e:
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.") from e
    return chunk_grid_cls.from_dict(data, array_shape=array_shape)  # type: ignore[arg-type, call-arg]


__all__ = [
    "ChunkGrid",
    "RegularChunkGrid",
    "_auto_partition",
    "_guess_chunks",
    "_guess_num_chunks_per_axis_shard",
    "normalize_chunks",
    "parse_chunk_grid",
]
