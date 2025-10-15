from __future__ import annotations

import itertools
import math
import numbers
import operator
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    ShapeLike,
    ceildiv,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from zarr.core.array import ShardsLike

from collections.abc import Sequence

# Type alias for chunk edge length specification
# Can be either an integer or a run-length encoded tuple [value, count]
ChunkEdgeLength = int | tuple[int, int]


class RectilinearChunkGridConfigurationDict(TypedDict):
    """TypedDict for rectilinear chunk grid configuration"""

    kind: Literal["inline"]
    chunk_shapes: Sequence[Sequence[ChunkEdgeLength]]


def _expand_run_length_encoding(spec: Sequence[ChunkEdgeLength]) -> tuple[int, ...]:
    """
    Expand a chunk edge length specification into a tuple of integers.

    The specification can contain:
    - integers: representing explicit edge lengths
    - tuples [value, count]: representing run-length encoded sequences

    Parameters
    ----------
    spec : Sequence[ChunkEdgeLength]
        The chunk edge length specification for one axis

    Returns
    -------
    tuple[int, ...]
        Expanded sequence of chunk edge lengths

    Examples
    --------
    >>> _expand_run_length_encoding([2, 3])
    (2, 3)
    >>> _expand_run_length_encoding([[2, 3]])
    (2, 2, 2)
    >>> _expand_run_length_encoding([1, [2, 1], 3])
    (1, 2, 3)
    >>> _expand_run_length_encoding([[1, 3], 3])
    (1, 1, 1, 3)
    """
    result: list[int] = []
    for item in spec:
        if isinstance(item, int):
            # Explicit edge length
            result.append(item)
        elif isinstance(item, (list, tuple)):
            # Run-length encoded: [value, count]
            if len(item) != 2:
                raise TypeError(
                    f"Run-length encoded items must be [int, int], got list of length {len(item)}"
                )
            value, count = item
            # Runtime validation of JSON data
            if not isinstance(value, int) or not isinstance(count, int):  # type: ignore[redundant-expr]
                raise TypeError(
                    f"Run-length encoded items must be [int, int], got [{type(value).__name__}, {type(count).__name__}]"
                )
            if count < 0:
                raise ValueError(f"Run-length count must be non-negative, got {count}")
            result.extend([value] * count)
        else:
            raise TypeError(
                f"Chunk edge length must be int or [int, int] for run-length encoding, got {type(item)}"
            )
    return tuple(result)


def _parse_chunk_shapes(
    data: Sequence[Sequence[ChunkEdgeLength]],
) -> tuple[tuple[int, ...], ...]:
    """
    Parse and expand chunk_shapes from metadata.

    Parameters
    ----------
    data : Sequence[Sequence[ChunkEdgeLength]]
        The chunk_shapes specification from metadata

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Tuple of expanded chunk edge lengths for each axis
    """
    # Runtime validation - strings are sequences but we don't want them
    # Type annotation is for static typing, this validates actual JSON data
    if isinstance(data, str) or not isinstance(data, Sequence):  # type: ignore[redundant-expr,unreachable]
        raise TypeError(f"chunk_shapes must be a sequence, got {type(data)}")

    result = []
    for i, axis_spec in enumerate(data):
        # Runtime validation for each axis spec
        if isinstance(axis_spec, str) or not isinstance(axis_spec, Sequence):  # type: ignore[redundant-expr,unreachable]
            raise TypeError(f"chunk_shapes[{i}] must be a sequence, got {type(axis_spec)}")
        expanded = _expand_run_length_encoding(axis_spec)
        result.append(expanded)

    return tuple(result)


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


def normalize_chunks(chunks: Any, shape: tuple[int, ...], typesize: int) -> tuple[int, ...]:
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
    if all(isinstance(c, (tuple | list)) for c in chunks):
        # take first chunk size for each dimension
        chunks = tuple(
            c[0] for c in chunks
        )  # TODO: check/error/warn for irregular chunks (e.g. if c[0] != c[1:-1])

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


@dataclass(frozen=True)
class ChunkGrid(Metadata):
    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkGrid) -> ChunkGrid:
        if isinstance(data, ChunkGrid):
            return data

        name_parsed, _ = parse_named_configuration(data)
        if name_parsed == "regular":
            return RegularChunkGrid._from_dict(data)
        elif name_parsed == "rectilinear":
            return RectilinearChunkGrid._from_dict(data)
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.")

    @abstractmethod
    def all_chunk_coords(self, array_shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        pass

    @abstractmethod
    def get_nchunks(self, array_shape: tuple[int, ...]) -> int:
        pass


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: tuple[int, ...]

    def __init__(self, *, chunk_shape: ShapeLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def _from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": tuple(self.chunk_shape)}}

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


@dataclass(frozen=True)
class RectilinearChunkGrid(ChunkGrid):
    """
    A rectilinear chunk grid where chunk sizes vary along each axis.

    Attributes
    ----------
    chunk_shapes : tuple[tuple[int, ...], ...]
        For each axis, a tuple of chunk edge lengths along that axis.
        The sum of edge lengths must equal the array shape along that axis.
    """

    chunk_shapes: tuple[tuple[int, ...], ...]

    def __init__(self, *, chunk_shapes: Sequence[Sequence[int]]) -> None:
        """
        Initialize a RectilinearChunkGrid.

        Parameters
        ----------
        chunk_shapes : Sequence[Sequence[int]]
            For each axis, a sequence of chunk edge lengths.
        """
        # Convert to nested tuples and validate
        parsed_shapes: list[tuple[int, ...]] = []
        for i, axis_chunks in enumerate(chunk_shapes):
            if not isinstance(axis_chunks, Sequence):
                raise TypeError(f"chunk_shapes[{i}] must be a sequence, got {type(axis_chunks)}")
            # Validate all are positive integers
            axis_tuple = tuple(axis_chunks)
            for j, size in enumerate(axis_tuple):
                if not isinstance(size, int):
                    raise TypeError(
                        f"chunk_shapes[{i}][{j}] must be an int, got {type(size).__name__}"
                    )
                if size <= 0:
                    raise ValueError(f"chunk_shapes[{i}][{j}] must be positive, got {size}")
            parsed_shapes.append(axis_tuple)

        object.__setattr__(self, "chunk_shapes", tuple(parsed_shapes))

    @classmethod
    def _from_dict(cls, data: dict[str, JSON]) -> Self:
        """
        Parse a RectilinearChunkGrid from metadata dict.

        Parameters
        ----------
        data : dict[str, JSON]
            Metadata dictionary with 'name' and 'configuration' keys

        Returns
        -------
        Self
            A RectilinearChunkGrid instance
        """
        _, configuration = parse_named_configuration(data, "rectilinear")

        if not isinstance(configuration, dict):
            raise TypeError(f"configuration must be a dict, got {type(configuration)}")

        # Validate kind field
        kind = configuration.get("kind")
        if kind != "inline":
            raise ValueError(f"Only 'inline' kind is supported, got {kind!r}")

        # Parse chunk_shapes with run-length encoding support
        chunk_shapes_raw = configuration.get("chunk_shapes")
        if chunk_shapes_raw is None:
            raise ValueError("configuration must contain 'chunk_shapes'")

        # Type ignore: JSON data validated at runtime by _parse_chunk_shapes
        chunk_shapes_expanded = _parse_chunk_shapes(chunk_shapes_raw)  # type: ignore[arg-type]

        return cls(chunk_shapes=chunk_shapes_expanded)

    def to_dict(self) -> dict[str, JSON]:
        """
        Convert to metadata dict format.

        Returns
        -------
        dict[str, JSON]
            Metadata dictionary with 'name' and 'configuration' keys
        """
        # Convert to list for JSON serialization
        chunk_shapes_list = [list(axis_chunks) for axis_chunks in self.chunk_shapes]

        return {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": chunk_shapes_list,
            },
        }

    def all_chunk_coords(self, array_shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        """
        Generate all chunk coordinates for the given array shape.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array

        Yields
        ------
        tuple[int, ...]
            Chunk coordinates

        Raises
        ------
        ValueError
            If array_shape doesn't match chunk_shapes
        """
        if len(array_shape) != len(self.chunk_shapes):
            raise ValueError(
                f"array_shape has {len(array_shape)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        # Validate that chunk sizes sum to array shape
        for axis, (arr_size, axis_chunks) in enumerate(
            zip(array_shape, self.chunk_shapes, strict=False)
        ):
            chunk_sum = sum(axis_chunks)
            if chunk_sum != arr_size:
                raise ValueError(
                    f"Sum of chunk sizes along axis {axis} is {chunk_sum} "
                    f"but array shape is {arr_size}"
                )

        # Generate coordinates
        # For each axis, we have len(axis_chunks) chunks
        nchunks_per_axis = [len(axis_chunks) for axis_chunks in self.chunk_shapes]
        return itertools.product(*(range(n) for n in nchunks_per_axis))

    def get_nchunks(self, array_shape: tuple[int, ...]) -> int:
        """
        Get the total number of chunks for the given array shape.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array

        Returns
        -------
        int
            Total number of chunks

        Raises
        ------
        ValueError
            If array_shape doesn't match chunk_shapes
        """
        if len(array_shape) != len(self.chunk_shapes):
            raise ValueError(
                f"array_shape has {len(array_shape)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        # Validate that chunk sizes sum to array shape
        for axis, (arr_size, axis_chunks) in enumerate(
            zip(array_shape, self.chunk_shapes, strict=False)
        ):
            chunk_sum = sum(axis_chunks)
            if chunk_sum != arr_size:
                raise ValueError(
                    f"Sum of chunk sizes along axis {axis} is {chunk_sum} "
                    f"but array shape is {arr_size}"
                )

        # Total chunks is the product of number of chunks per axis
        return reduce(operator.mul, (len(axis_chunks) for axis_chunks in self.chunk_shapes), 1)


def _auto_partition(
    *,
    array_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...] | Literal["auto"],
    shard_shape: ShardsLike | None,
    item_size: int,
) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    """
    Automatically determine the shard shape and chunk shape for an array, given the shape and dtype of the array.
    If `shard_shape` is `None` and the chunk_shape is "auto", the chunks will be set heuristically based
    on the dtype and shape of the array.
    If `shard_shape` is "auto", then the shard shape will be set heuristically from the dtype and shape
    of the array; if the `chunk_shape` is also "auto", then the chunks will be set heuristically as well,
    given the dtype and shard shape. Otherwise, the chunks will be returned as-is.
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
            _chunks_out = _guess_chunks(array_shape, item_size, max_bytes=1024)
        else:
            _chunks_out = chunk_shape

        if shard_shape == "auto":
            warnings.warn(
                "Automatic shard shape inference is experimental and may change without notice.",
                ZarrUserWarning,
                stacklevel=2,
            )
            _shards_out = ()
            for a_shape, c_shape in zip(array_shape, _chunks_out, strict=True):
                # TODO: make a better heuristic than this.
                # for each axis, if there are more than 8 chunks along that axis, then put
                # 2 chunks in each shard for that axis.
                if a_shape // c_shape > 8:
                    _shards_out += (c_shape * 2,)
                else:
                    _shards_out += (c_shape,)
        elif isinstance(shard_shape, dict):
            _shards_out = tuple(shard_shape["shape"])
        else:
            _shards_out = shard_shape

    return _shards_out, _chunks_out
