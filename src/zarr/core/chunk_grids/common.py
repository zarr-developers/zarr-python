from __future__ import annotations

import math
import numbers
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, Union

import numpy as np
import numpy.typing as npt

import zarr
from zarr.abc.metadata import Metadata
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.array import ShardsLike
    from zarr.core.common import JSON

# Type alias for chunk edge length specification
# Can be either an integer or a run-length encoded pair [value, count]
# The pair can be a tuple or list (common in JSON/test code)
ChunkEdgeLength = int | tuple[int, int] | list[int]

# User-facing chunk specification types
# Note: ChunkGrid is defined later in this file but can be used via string literal
ChunksLike: TypeAlias = Union[
    tuple[int, ...],  # Regular chunks: (10, 10) -> RegularChunkGrid
    int,  # Uniform chunks: 10 -> RegularChunkGrid
    Sequence[
        Sequence[ChunkEdgeLength]
    ],  # Variable chunks with optional RLE -> RectilinearChunkGrid
    "ChunkGrid",  # Explicit ChunkGrid instance (forward reference)
    Literal["auto"],  # Auto-chunking -> RegularChunkGrid
]


class ChunkGrid(ABC, Metadata):
    @abstractmethod
    def to_dict(self) -> dict[str, JSON]: ...

    @abstractmethod
    def update_shape(self, new_shape: tuple[int, ...]) -> Self:
        pass

    @abstractmethod
    def all_chunk_coords(self, array_shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        pass

    @abstractmethod
    def get_nchunks(self, array_shape: tuple[int, ...]) -> int:
        pass

    @abstractmethod
    def get_chunk_shape(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the shape of a specific chunk.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.
        chunk_coord : tuple[int, ...]
            Coordinates of the chunk in the chunk grid.

        Returns
        -------
        tuple[int, ...]
            Shape of the chunk at the given coordinates.
        """

    @abstractmethod
    def get_chunk_start(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the starting position of a chunk in the array.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.
        chunk_coord : tuple[int, ...]
            Coordinates of the chunk in the chunk grid.

        Returns
        -------
        tuple[int, ...]
            Starting position (offset) of the chunk in the array.
        """

    @abstractmethod
    def array_index_to_chunk_coord(
        self, array_shape: tuple[int, ...], array_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Map an array index to the chunk coordinates that contain it.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.
        array_index : tuple[int, ...]
            Index in the array.

        Returns
        -------
        tuple[int, ...]
            Coordinates of the chunk containing the array index.
        """

    @abstractmethod
    def array_indices_to_chunk_dim(
        self, array_shape: tuple[int, ...], dim: int, indices: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        """
        Map an array of indices along one dimension to chunk coordinates (vectorized).

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.
        dim : int
            Dimension index.
        indices : np.ndarray
            Array of indices along the given dimension.

        Returns
        -------
        np.ndarray
            Array of chunk coordinates, same shape as indices.
        """

    @abstractmethod
    def chunks_per_dim(self, array_shape: tuple[int, ...], dim: int) -> int:
        """
        Get the number of chunks along a specific dimension.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.
        dim : int
            Dimension index.

        Returns
        -------
        int
            Number of chunks along the dimension.
        """

    @abstractmethod
    def get_chunk_grid_shape(self, array_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the shape of the chunk grid (number of chunks along each dimension).

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the full array.

        Returns
        -------
        tuple[int, ...]
            Number of chunks along each dimension.
        """


def _guess_chunks(
    shape: tuple[int, ...] | int,
    typesize: int,
    *,
    increment_bytes: int = 256 * 1024,
    min_bytes: int = 128 * 1024,
    max_bytes: int = 64 * 1024 * 1024,
) -> tuple[int, ...]:
    """
    Iteratively guess an appropriate chunk layout for an array, given its shape and
    the size of each element in bytes, and size constraints expressed in bytes. This logic is
    adapted from h5py.

    Parameters
    ----------
    shape : tuple[int, ...]
        The chunk shape.
    typesize : int
        The size, in bytes, of each element of the chunk.
    increment_bytes : int = 256 * 1024
        The number of bytes used to increment or decrement the target chunk size in bytes.
    min_bytes : int = 128 * 1024
        The soft lower bound on the final chunk size in bytes.
    max_bytes : int = 64 * 1024 * 1024
        The hard upper bound on the final chunk size in bytes.

    Returns
    -------
    tuple[int, ...]

    """
    if min_bytes >= max_bytes:
        raise ValueError(f"Cannot have more min_bytes ({min_bytes}) than max_bytes ({max_bytes})")
    if isinstance(shape, int):
        shape = (shape,)

    if typesize == 0:
        return shape

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


def _normalize_chunks(chunks: Any, shape: tuple[int, ...], typesize: int) -> tuple[int, ...]:
    """Convenience function to normalize the `chunks` argument for an array
    with the given `shape`."""

    # N.B., expect shape already normalized

    # handle auto-chunking
    if chunks is None or chunks is True:
        return _guess_chunks(shape, typesize)

    # handle no chunking
    if chunks is False:
        return shape

    # handle 1D convenience form
    if isinstance(chunks, numbers.Integral):
        chunks = tuple(int(chunks) for _ in shape)

    # handle dask-style chunks (iterable of iterables)
    # Note: Only regular chunks are supported here. Irregular chunks will trigger a warning
    # and use only the first chunk size per dimension. For true variable chunks,
    # use RectilinearChunkGrid.
    if all(isinstance(c, (tuple | list)) for c in chunks):
        # Check for irregular chunks and warn user
        for dim_idx, c in enumerate(chunks):
            if len(c) > 1 and not all(chunk_size == c[0] for chunk_size in c):
                warnings.warn(
                    f"Irregular chunks detected in dimension {dim_idx}: {c}. "
                    f"Only the first chunk size ({c[0]}) will be used, "
                    f"resulting in regular chunks. "
                    f"For variable chunk sizes, use RectilinearChunkGrid instead.",
                    UserWarning,
                    stacklevel=2,
                )
        # take first chunk size for each dimension
        chunks = tuple(c[0] for c in chunks)

    # handle bad dimensionality
    if len(chunks) > len(shape):
        raise ValueError("too many dimensions in chunks")

    # handle underspecified chunks
    if len(chunks) < len(shape):
        # assume chunks across remaining dimensions
        chunks += shape[len(chunks) :]

    # handle None or -1 in chunks
    if -1 in chunks or None in chunks:
        chunks = tuple(
            s if c == -1 or c is None else int(c) for s, c in zip(shape, chunks, strict=False)
        )

    if not all(isinstance(c, numbers.Integral) for c in chunks):
        raise TypeError("non integer value in chunks")

    return tuple(int(c) for c in chunks)


def _guess_num_chunks_per_axis_shard(
    chunk_shape: tuple[int, ...], item_size: int, max_bytes: int, array_shape: tuple[int, ...]
) -> int:
    """Generate the number of chunks per axis to hit a target max byte size for a shard.

    For example, for a (2,2,2) chunk size and item size 4, maximum bytes of 256 would return 2.
    In other words the shard would be a (2,2,2) grid of (2,2,2) chunks
    i.e., prod(chunk_shape) * (returned_val * len(chunk_shape)) * item_size = 256 bytes.

    Parameters
    ----------
    chunk_shape
        The shape of the (inner) chunks.
    item_size
        The item size of the data i.e., 2 for uint16.
    max_bytes
        The maximum number of bytes per shard to allow.
    array_shape
        The shape of the underlying array.

    Returns
    -------
        The number of chunks per axis.
    """
    bytes_per_chunk = np.prod(chunk_shape) * item_size
    if max_bytes < bytes_per_chunk:
        return 1
    num_axes = len(chunk_shape)
    chunks_per_shard = 1
    # First check for byte size, second check to make sure we don't go bigger than the array shape
    while (bytes_per_chunk * ((chunks_per_shard + 1) ** num_axes)) <= max_bytes and all(
        c * (chunks_per_shard + 1) <= a for c, a in zip(chunk_shape, array_shape, strict=True)
    ):
        chunks_per_shard += 1
    return chunks_per_shard


def _auto_partition(
    *,
    array_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...] | Literal["auto"],
    shard_shape: ShardsLike | None,
    item_size: int,
) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    """
    Automatically determine the shard shape and chunk shape for an array, given the shape and
    dtype of the array.
    If `shard_shape` is `None` and the chunk_shape is "auto", the chunks will be set heuristically
    based on the dtype and shape of the array.
    If `shard_shape` is "auto", then the shard shape will be set heuristically from the dtype and
    shape of the array; if the `chunk_shape` is also "auto", then the chunks will be set
    heuristically as well, given the dtype and shard shape. Otherwise, the chunks will be returned
    as-is.
    """
    if shard_shape is None:
        _shards_out: None | tuple[int, ...] = None
        if chunk_shape == "auto":
            _chunks_out = _guess_chunks(array_shape, item_size)
        else:
            _chunks_out = chunk_shape
    else:
        if chunk_shape == "auto":
            # aim for a 1MiB chunk
            _chunks_out = _guess_chunks(array_shape, item_size, max_bytes=1048576)
        else:
            _chunks_out = chunk_shape

        if shard_shape == "auto":
            warnings.warn(
                "Automatic shard shape inference is experimental and may change without notice.",
                ZarrUserWarning,
                stacklevel=2,
            )
            _shards_out = ()
            target_shard_size_bytes = zarr.config.get("array.target_shard_size_bytes", None)
            num_chunks_per_shard_axis = (
                _guess_num_chunks_per_axis_shard(
                    chunk_shape=_chunks_out,
                    item_size=item_size,
                    max_bytes=target_shard_size_bytes,
                    array_shape=array_shape,
                )
                if (has_auto_shard := (target_shard_size_bytes is not None))
                else 2
            )
            for a_shape, c_shape in zip(array_shape, _chunks_out, strict=True):
                can_shard_axis = a_shape // c_shape > 8 if not has_auto_shard else True
                if can_shard_axis:
                    _shards_out += (c_shape * num_chunks_per_shard_axis,)
                else:
                    _shards_out += (c_shape,)
        elif isinstance(shard_shape, dict):
            _shards_out = tuple(shard_shape["shape"])
        else:
            _shards_out = shard_shape

    return _shards_out, _chunks_out


def _is_nested_sequence(chunks: Any) -> bool:
    """
    Check if chunks is a nested sequence (tuple of tuples/lists).

    Returns True for inputs like [[10, 20], [5, 5]] or [(10, 20), (5, 5)].
    Returns False for flat sequences like (10, 10) or [10, 10].
    """
    # Not a sequence if it's a string, int, tuple of basic types, or ChunkGrid
    if isinstance(chunks, str | int | ChunkGrid):
        return False

    # Check if it's iterable
    if not hasattr(chunks, "__iter__"):
        return False

    # Check if first element is a sequence (but not string/bytes/int)
    try:
        first_elem = next(iter(chunks), None)
        if first_elem is None:
            return False
        return hasattr(first_elem, "__iter__") and not isinstance(first_elem, str | bytes | int)
    except (TypeError, StopIteration):
        return False


@dataclass(frozen=True)
class ResolvedChunkSpec:
    """
    Result of resolving chunk specification.

    This dataclass encapsulates the resolved chunk grid and shards
    parameters for creating a Zarr array.

    After resolution, all chunk specifications are converted to a concrete
    ChunkGrid instance (either RegularChunkGrid or RectilinearChunkGrid).
    The shards parameter is kept separate as it wraps the chunk_grid in
    a ShardingCodec.

    Attributes
    ----------
    chunk_grid : ChunkGrid
        The resolved chunk grid. Always a concrete instance after resolution.
    shards : tuple[int, ...] | None
        The shards parameter to pass to init_array/from_array.
        None if sharding is not used.
    """

    chunk_grid: ChunkGrid
    shards: tuple[int, ...] | None
