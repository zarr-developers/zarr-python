from __future__ import annotations

import bisect
import itertools
import math
import numbers
import operator
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, Union

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


# Type alias for chunk edge length specification
# Can be either an integer or a run-length encoded tuple [value, count]
ChunkEdgeLength = int | tuple[int, int]

# User-facing chunk specification types
# Note: ChunkGrid is defined later in this file but can be used via string literal
ChunksLike: TypeAlias = Union[
    tuple[int, ...],  # Regular chunks: (10, 10) → RegularChunkGrid
    int,  # Uniform chunks: 10 → RegularChunkGrid
    Sequence[Sequence[int]],  # Variable chunks: [[10,20],[5,5]] → RectilinearChunkGrid
    "ChunkGrid",  # Explicit ChunkGrid instance (forward reference)
    Literal["auto"],  # Auto-chunking → RegularChunkGrid
]


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
        elif isinstance(item, list | tuple):
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


def _compress_run_length_encoding(chunks: tuple[int, ...]) -> list[int | list[int]]:
    """
    Compress a sequence of chunk sizes to RLE format where beneficial.

    This function automatically detects runs of identical values and compresses them
    using the [value, count] format. Single values or short runs are kept as-is.

    Parameters
    ----------
    chunks : tuple[int, ...]
        Sequence of chunk sizes along one dimension

    Returns
    -------
    list[int | list[int]]
        Compressed representation using RLE where beneficial

    Examples
    --------
    >>> _compress_run_length_encoding((10, 10, 10, 10, 10, 10))
    [[10, 6]]
    >>> _compress_run_length_encoding((10, 20, 30))
    [10, 20, 30]
    >>> _compress_run_length_encoding((10, 10, 10, 20, 20, 30))
    [[10, 3], [20, 2], 30]
    >>> _compress_run_length_encoding((5, 5, 10, 10, 10, 10, 15))
    [[5, 2], [10, 4], 15]
    """
    if not chunks:
        return []

    result: list[int | list[int]] = []
    current_value = chunks[0]
    current_count = 1

    for value in chunks[1:]:
        if value == current_value:
            current_count += 1
        else:
            # Decide whether to use RLE or explicit value
            # Use RLE if count >= 3 to save space (tradeoff: [v,c] vs v,v,v)
            if current_count >= 3:
                result.append([current_value, current_count])
            elif current_count == 2:
                # For count=2, RLE doesn't save space, but use it for consistency
                result.append([current_value, current_count])
            else:
                result.append(current_value)

            current_value = value
            current_count = 1

    # Handle the last run
    if current_count >= 3 or current_count == 2:
        result.append([current_value, current_count])
    else:
        result.append(current_value)

    return result


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

    def get_chunk_shape(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the shape of a specific chunk.

        For RegularChunkGrid, all chunks have the same shape except possibly
        the last chunk in each dimension.
        """
        return tuple(
            int(min(self.chunk_shape[i], array_shape[i] - chunk_coord[i] * self.chunk_shape[i]))
            for i in range(len(array_shape))
        )

    def get_chunk_start(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the starting position of a chunk in the array.

        For RegularChunkGrid, this is simply chunk_coord * chunk_shape.
        """
        return tuple(
            coord * size for coord, size in zip(chunk_coord, self.chunk_shape, strict=False)
        )

    def array_index_to_chunk_coord(
        self, array_shape: tuple[int, ...], array_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Map an array index to chunk coordinates.

        For RegularChunkGrid, this is simply array_index // chunk_shape.
        """
        return tuple(
            0 if size == 0 else idx // size
            for idx, size in zip(array_index, self.chunk_shape, strict=False)
        )

    def chunks_per_dim(self, array_shape: tuple[int, ...], dim: int) -> int:
        """
        Get the number of chunks along a specific dimension.

        For RegularChunkGrid, this is ceildiv(array_shape[dim], chunk_shape[dim]).
        """
        return ceildiv(array_shape[dim], self.chunk_shape[dim])

    def get_chunk_grid_shape(self, array_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the shape of the chunk grid (number of chunks along each dimension).

        For RegularChunkGrid, this is computed using ceildiv for each dimension.
        """
        return tuple(
            ceildiv(array_len, chunk_len)
            for array_len, chunk_len in zip(array_shape, self.chunk_shape, strict=False)
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
        Convert to metadata dict format with automatic RLE compression.

        This method automatically compresses chunk shapes using run-length encoding
        where beneficial (runs of 2 or more identical values). This reduces metadata
        size for arrays with many uniform chunks.

        Returns
        -------
        dict[str, JSON]
            Metadata dictionary with 'name' and 'configuration' keys

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10, 10, 10, 10], [5, 5, 5, 5, 5]])
        >>> grid.to_dict()['configuration']['chunk_shapes']
        [[[10, 6]], [[5, 5]]]
        """
        # Compress each dimension using RLE where beneficial
        chunk_shapes_compressed = [
            _compress_run_length_encoding(axis_chunks) for axis_chunks in self.chunk_shapes
        ]

        return {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": chunk_shapes_compressed,
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

    def _validate_array_shape(self, array_shape: tuple[int, ...]) -> None:
        """
        Validate that array_shape is compatible with chunk_shapes.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes
        """
        if len(array_shape) != len(self.chunk_shapes):
            raise ValueError(
                f"array_shape has {len(array_shape)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        for axis, (arr_size, axis_chunks) in enumerate(
            zip(array_shape, self.chunk_shapes, strict=False)
        ):
            chunk_sum = sum(axis_chunks)
            if chunk_sum != arr_size:
                raise ValueError(
                    f"Sum of chunk sizes along axis {axis} is {chunk_sum} "
                    f"but array shape is {arr_size}"
                )

    @cached_property
    def _cumulative_sizes(self) -> tuple[tuple[int, ...], ...]:
        """
        Compute cumulative sizes for each axis.

        Returns a tuple of tuples where each inner tuple contains cumulative
        chunk sizes for an axis. Used for efficient chunk boundary calculations.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Cumulative sizes for each axis

        Examples
        --------
        For chunk_shapes = [[2, 3, 1], [4, 2]]:
        Returns ((0, 2, 5, 6), (0, 4, 6))
        """
        result = []
        for axis_chunks in self.chunk_shapes:
            cumsum = [0]
            for size in axis_chunks:
                cumsum.append(cumsum[-1] + size)
            result.append(tuple(cumsum))
        return tuple(result)

    def get_chunk_start(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the starting position (offset) of a chunk in the array.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        chunk_coord : tuple[int, ...]
            Chunk coordinates (indices into the chunk grid)

        Returns
        -------
        tuple[int, ...]
            Starting index of the chunk in the array

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes
        IndexError
            If chunk_coord is out of bounds

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        >>> grid.get_chunk_start((6, 6), (0, 0))
        (0, 0)
        >>> grid.get_chunk_start((6, 6), (1, 1))
        (2, 3)
        """
        self._validate_array_shape(array_shape)

        if len(chunk_coord) != len(self.chunk_shapes):
            raise IndexError(
                f"chunk_coord has {len(chunk_coord)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        # Validate chunk coordinates are in bounds
        for axis, (coord, axis_chunks) in enumerate(
            zip(chunk_coord, self.chunk_shapes, strict=False)
        ):
            if not (0 <= coord < len(axis_chunks)):
                raise IndexError(
                    f"chunk_coord[{axis}] = {coord} is out of bounds [0, {len(axis_chunks)})"
                )

        # Use cumulative sizes to get start position
        return tuple(self._cumulative_sizes[axis][coord] for axis, coord in enumerate(chunk_coord))

    def get_chunk_shape(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Get the shape of a specific chunk.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        chunk_coord : tuple[int, ...]
            Chunk coordinates (indices into the chunk grid)

        Returns
        -------
        tuple[int, ...]
            Shape of the chunk

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes
        IndexError
            If chunk_coord is out of bounds

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        >>> grid.get_chunk_shape((6, 6), (0, 0))
        (2, 4)
        >>> grid.get_chunk_shape((6, 6), (1, 0))
        (3, 4)
        """
        self._validate_array_shape(array_shape)

        if len(chunk_coord) != len(self.chunk_shapes):
            raise IndexError(
                f"chunk_coord has {len(chunk_coord)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        # Validate chunk coordinates are in bounds
        for axis, (coord, axis_chunks) in enumerate(
            zip(chunk_coord, self.chunk_shapes, strict=False)
        ):
            if not (0 <= coord < len(axis_chunks)):
                raise IndexError(
                    f"chunk_coord[{axis}] = {coord} is out of bounds [0, {len(axis_chunks)})"
                )

        # Get shape directly from chunk_shapes
        return tuple(
            axis_chunks[coord]
            for axis_chunks, coord in zip(self.chunk_shapes, chunk_coord, strict=False)
        )

    def get_chunk_slice(
        self, array_shape: tuple[int, ...], chunk_coord: tuple[int, ...]
    ) -> tuple[slice, ...]:
        """
        Get the slice for indexing into an array for a specific chunk.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        chunk_coord : tuple[int, ...]
            Chunk coordinates (indices into the chunk grid)

        Returns
        -------
        tuple[slice, ...]
            Slice tuple for indexing the array

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes
        IndexError
            If chunk_coord is out of bounds

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        >>> grid.get_chunk_slice((6, 6), (0, 0))
        (slice(0, 2, None), slice(0, 3, None))
        >>> grid.get_chunk_slice((6, 6), (1, 1))
        (slice(2, 4, None), slice(3, 6, None))
        """
        start = self.get_chunk_start(array_shape, chunk_coord)
        shape = self.get_chunk_shape(array_shape, chunk_coord)

        return tuple(slice(s, s + length) for s, length in zip(start, shape, strict=False))

    def get_chunk_grid_shape(self, array_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the shape of the chunk grid (number of chunks per axis).

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array

        Returns
        -------
        tuple[int, ...]
            Number of chunks along each axis

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        >>> grid.get_chunk_grid_shape((6, 6))
        (3, 2)
        """
        self._validate_array_shape(array_shape)

        return tuple(len(axis_chunks) for axis_chunks in self.chunk_shapes)

    def array_index_to_chunk_coord(
        self, array_shape: tuple[int, ...], array_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Find which chunk contains a given array index.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        array_index : tuple[int, ...]
            Index into the array

        Returns
        -------
        tuple[int, ...]
            Chunk coordinates containing the array index

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes
        IndexError
            If array_index is out of bounds

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        >>> grid.array_index_to_chunk_coord((6, 6), (0, 0))
        (0, 0)
        >>> grid.array_index_to_chunk_coord((6, 6), (2, 0))
        (1, 0)
        >>> grid.array_index_to_chunk_coord((6, 6), (5, 5))
        (2, 1)
        """
        self._validate_array_shape(array_shape)

        if len(array_index) != len(array_shape):
            raise IndexError(
                f"array_index has {len(array_index)} dimensions but "
                f"array_shape has {len(array_shape)} dimensions"
            )

        # Validate array index is in bounds
        for axis, (idx, size) in enumerate(zip(array_index, array_shape, strict=False)):
            if not (0 <= idx < size):
                raise IndexError(f"array_index[{axis}] = {idx} is out of bounds [0, {size})")

        # Use binary search in cumulative sizes to find chunk coordinate
        result = []
        for axis, idx in enumerate(array_index):
            cumsum = self._cumulative_sizes[axis]
            # bisect_right gives us the chunk index + 1, so subtract 1
            chunk_idx = bisect.bisect_right(cumsum, idx) - 1
            result.append(chunk_idx)

        return tuple(result)

    def chunks_in_selection(
        self, array_shape: tuple[int, ...], selection: tuple[slice, ...]
    ) -> Iterator[tuple[int, ...]]:
        """
        Get all chunks that intersect with a given selection.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        selection : tuple[slice, ...]
            Selection (slices) into the array

        Yields
        ------
        tuple[int, ...]
            Chunk coordinates that intersect with the selection

        Raises
        ------
        ValueError
            If array_shape is incompatible with chunk_shapes or selection is invalid

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        >>> selection = (slice(1, 5), slice(2, 5))
        >>> list(grid.chunks_in_selection((6, 6), selection))
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        """
        self._validate_array_shape(array_shape)

        if len(selection) != len(array_shape):
            raise ValueError(
                f"selection has {len(selection)} dimensions but "
                f"array_shape has {len(array_shape)} dimensions"
            )

        # Normalize slices and find chunk ranges for each axis
        chunk_ranges = []
        for axis, (sel, size) in enumerate(zip(selection, array_shape, strict=False)):
            if not isinstance(sel, slice):
                raise TypeError(f"selection[{axis}] must be a slice, got {type(sel)}")

            # Normalize slice with array size
            start, stop, step = sel.indices(size)

            if step != 1:
                raise ValueError(f"selection[{axis}] has step={step}, only step=1 is supported")

            if start >= stop:
                # Empty selection
                return

            # Find first and last chunk that intersect with [start, stop)
            start_chunk = self.array_index_to_chunk_coord(
                array_shape, tuple(start if i == axis else 0 for i in range(len(array_shape)))
            )[axis]

            # stop-1 is the last index we need
            end_chunk = self.array_index_to_chunk_coord(
                array_shape, tuple(stop - 1 if i == axis else 0 for i in range(len(array_shape)))
            )[axis]

            chunk_ranges.append(range(start_chunk, end_chunk + 1))

        # Generate all combinations of chunk coordinates
        yield from itertools.product(*chunk_ranges)

    def chunks_per_dim(self, array_shape: tuple[int, ...], dim: int) -> int:
        """
        Get the number of chunks along a specific dimension.

        Parameters
        ----------
        array_shape : tuple[int, ...]
            Shape of the array
        dim : int
            Dimension index

        Returns
        -------
        int
            Number of chunks along the dimension

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[10, 20], [5, 5, 5]])
        >>> grid.chunks_per_dim((30, 15), 0)  # 2 chunks along axis 0
        2
        >>> grid.chunks_per_dim((30, 15), 1)  # 3 chunks along axis 1
        3
        """
        self._validate_array_shape(array_shape)
        return len(self.chunk_shapes[dim])


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


def _normalize_rectilinear_chunks(
    chunks: Sequence[Sequence[int | Sequence[int]]], shape: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """
    Normalize and validate variable chunks for RectilinearChunkGrid.

    Supports both explicit chunk sizes and run-length encoding (RLE).
    RLE format: [[value, count]] expands to 'count' repetitions of 'value'.

    Parameters
    ----------
    chunks : Sequence[Sequence[int | Sequence[int]]]
        Nested sequence where each element is a sequence of chunk sizes along that dimension.
        Each chunk size can be:
        - An integer: explicit chunk size
        - A sequence [value, count]: RLE format (expands to 'count' chunks of size 'value')
    shape : tuple[int, ...]
        The shape of the array.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Normalized chunk shapes as tuple of tuples.

    Raises
    ------
    ValueError
        If chunks don't match shape or sum incorrectly.
    TypeError
        If chunk specification format is invalid.

    Examples
    --------
    >>> _normalize_rectilinear_chunks([[10, 20, 30], [25, 25]], (60, 50))
    ((10, 20, 30), (25, 25))
    >>> _normalize_rectilinear_chunks([[[10, 6]], [[10, 5]]], (60, 50))
    ((10, 10, 10, 10, 10, 10), (10, 10, 10, 10, 10))
    """
    # Expand RLE for each dimension
    try:
        chunk_shapes = tuple(
            _expand_run_length_encoding(dim)  # type: ignore[arg-type]
            for dim in chunks
        )
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Invalid variable chunks: {chunks}. Expected nested sequence of integers "
            f"or RLE format [[value, count]]."
        ) from e

    # Validate dimensionality
    if len(chunk_shapes) != len(shape):
        raise ValueError(
            f"Variable chunks dimensionality ({len(chunk_shapes)}) "
            f"must match array shape dimensionality ({len(shape)})"
        )

    # Validate that chunks sum to shape for each dimension
    for i, (dim_chunks, dim_size) in enumerate(zip(chunk_shapes, shape, strict=False)):
        chunk_sum = sum(dim_chunks)
        if chunk_sum != dim_size:
            raise ValueError(
                f"Variable chunks along dimension {i} sum to {chunk_sum} "
                f"but array shape is {dim_size}. Chunks must sum exactly to shape."
            )

    return chunk_shapes


def parse_chunk_grid(
    chunks: tuple[int, ...] | Sequence[Sequence[int]] | ChunkGrid | Literal["auto"] | int,
    *,
    shape: ShapeLike,
    item_size: int = 1,
    zarr_format: int | None = None,
) -> ChunkGrid:
    """
    Parse a chunks parameter into a ChunkGrid instance.

    This function handles multiple input formats for the chunks parameter and always
    returns a concrete ChunkGrid instance:
    - ChunkGrid instances: Returned as-is
    - Nested sequences (e.g., [[10, 20], [5, 5]]): Converted to RectilinearChunkGrid (Zarr v3 only)
    - Nested sequences with RLE (e.g., [[[10, 6]], [[10, 5]]]): Expanded and converted to RectilinearChunkGrid
    - Regular tuples/ints (e.g., (10, 10) or 10): Converted to RegularChunkGrid
    - Literal "auto": Computed using auto-chunking heuristics and converted to RegularChunkGrid

    Parameters
    ----------
    chunks : tuple[int, ...] | Sequence[Sequence[int]] | ChunkGrid | Literal["auto"] | int
        The chunks parameter to parse. Can be:
        - A ChunkGrid instance
        - A nested sequence for variable-sized chunks (supports RLE format)
        - A tuple of integers for uniform chunks
        - A single integer (for 1D arrays or uniform chunks across all dimensions)
        - The literal "auto"

        RLE (Run-Length Encoding) format: [[value, count]] expands to 'count' repetitions of 'value'.
        Example: [[[10, 6]]] creates 6 chunks of size 10 each.
    shape : ShapeLike
        The shape of the array. Required to create RegularChunkGrid for "auto" or tuple inputs.
    item_size : int, default=1
        The size of each array element in bytes. Used for auto-chunking heuristics.
    zarr_format : {2, 3, None}, optional
        The Zarr format version. Required for validating nested sequences
        (which are only supported in Zarr v3).

    Returns
    -------
    ChunkGrid
        A concrete ChunkGrid instance (either RegularChunkGrid or RectilinearChunkGrid).

    Raises
    ------
    ValueError
        If nested sequences are used with zarr_format=2, or if variable chunks don't sum to shape.
    TypeError
        If the chunks parameter cannot be parsed.

    Examples
    --------
    >>> # ChunkGrid instance
    >>> from zarr.core.chunk_grids import RegularChunkGrid
    >>> grid = RegularChunkGrid(chunk_shape=(10, 10))
    >>> result = parse_chunk_grid(grid, shape=(100, 100))
    >>> result is grid
    True

    >>> # Nested sequence for RectilinearChunkGrid
    >>> result = parse_chunk_grid([[10, 20, 30], [5, 5]], shape=(60, 10), zarr_format=3)
    >>> type(result).__name__
    'RectilinearChunkGrid'
    >>> result.chunk_shapes
    ((10, 20, 30), (5, 5))

    >>> # RLE format for RectilinearChunkGrid
    >>> result = parse_chunk_grid([[[10, 6]], [[10, 5]]], shape=(60, 50), zarr_format=3)
    >>> type(result).__name__
    'RectilinearChunkGrid'
    >>> result.chunk_shapes
    ((10, 10, 10, 10, 10, 10), (10, 10, 10, 10, 10))

    >>> # Regular tuple
    >>> result = parse_chunk_grid((10, 10), shape=(100, 100))
    >>> type(result).__name__
    'RegularChunkGrid'
    >>> result.chunk_shape
    (10, 10)

    >>> # Literal "auto"
    >>> result = parse_chunk_grid("auto", shape=(100, 100), item_size=4)
    >>> type(result).__name__
    'RegularChunkGrid'
    >>> isinstance(result.chunk_shape, tuple)
    True

    >>> # Single int
    >>> result = parse_chunk_grid(10, shape=(100, 100))
    >>> result.chunk_shape
    (10, 10)
    """

    # Case 1: Already a ChunkGrid instance
    if isinstance(chunks, ChunkGrid):
        return chunks

    # Parse shape to ensure it's a tuple
    shape_parsed = parse_shapelike(shape)

    # Case 2: String "auto" -> RegularChunkGrid
    if isinstance(chunks, str):
        # chunks can only be "auto" based on type annotation
        # _normalize_chunks expects None or True for auto-chunking, not "auto"
        chunk_shape = _normalize_chunks(None, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)

    # Case 3: Single int -> RegularChunkGrid
    if isinstance(chunks, int):
        chunk_shape = _normalize_chunks(chunks, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)

    # Case 4: Tuple or sequence - determine if regular or variable chunks
    if _is_nested_sequence(chunks):
        # Variable chunks (nested sequence) -> RectilinearChunkGrid
        if zarr_format == 2:
            raise ValueError(
                "Variable chunks (nested sequences) are only supported in Zarr format 3. "
                "Use zarr_format=3 or provide a regular tuple for chunks."
            )

        # Normalize and validate variable chunks
        chunk_shapes = _normalize_rectilinear_chunks(chunks, shape_parsed)  # type: ignore[arg-type]
        return RectilinearChunkGrid(chunk_shapes=chunk_shapes)
    else:
        # Regular tuple of ints -> RegularChunkGrid
        chunk_shape = _normalize_chunks(chunks, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)


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


def _validate_zarr_format_compatibility(
    chunks: Any,
    shards: Any,
    zarr_format: int,
) -> None:
    """
    Validate that chunk specification is compatible with Zarr format.

    Parameters
    ----------
    chunks : Any
        The chunks specification.
    shards : Any
        The shards specification.
    zarr_format : {2, 3}
        The Zarr format version.

    Raises
    ------
    ValueError
        If the specification is not compatible with the Zarr format.
    """
    if zarr_format == 2:
        # Zarr v2 doesn't support ChunkGrid instances
        if isinstance(chunks, ChunkGrid):
            raise ValueError(
                "ChunkGrid instances are only supported in Zarr format 3. "
                "For Zarr format 2, use a tuple of integers for chunks."
            )

        # Zarr v2 doesn't support nested sequences (variable chunks)
        if _is_nested_sequence(chunks):
            raise ValueError(
                "Variable chunks (nested sequences) are only supported in Zarr format 3. "
                "Use zarr_format=3 or provide a regular tuple for chunks."
            )

        # Zarr v2 doesn't support sharding
        if shards is not None:
            raise ValueError(
                f"Sharding is only supported in Zarr format 3. "
                f"Got zarr_format={zarr_format} with shards={shards}."
            )


def _validate_sharding_compatibility(
    chunks: Any,
    shards: Any,
) -> None:
    """
    Validate that chunk specification is compatible with sharding.

    Parameters
    ----------
    chunks : Any
        The chunks specification.
    shards : Any
        The shards specification.

    Raises
    ------
    ValueError
        If the chunk specification is not compatible with sharding.
    """
    if shards is not None:
        # ChunkGrid instances can't be used with sharding
        if isinstance(chunks, ChunkGrid):
            raise ValueError(
                "Cannot use ChunkGrid instances with sharding. "
                "When shards parameter is provided, chunks must be a tuple of integers or 'auto'."
            )

        # Variable chunks (nested sequences) can't be used with sharding
        if _is_nested_sequence(chunks):
            raise ValueError(
                "Cannot use variable chunks (nested sequences) with sharding. "
                "Sharding requires uniform chunk sizes."
            )


def _validate_data_compatibility(
    chunk_grid: ChunkGrid | None,
    has_data: bool,
) -> None:
    """
    Validate that chunk grid is compatible with creating from data.

    Parameters
    ----------
    chunk_grid : ChunkGrid | None
        The chunk grid.
    has_data : bool
        Whether the array is being created from existing data.

    Raises
    ------
    ValueError
        If the chunk grid is not compatible with from_array.
    """
    if has_data and chunk_grid is not None and isinstance(chunk_grid, RectilinearChunkGrid):
        # RectilinearChunkGrid doesn't work with from_array
        raise ValueError(
            "Cannot use RectilinearChunkGrid (variable-sized chunks) when creating array from data. "
            "The from_array function requires uniform chunk sizes. "
            "Use regular chunks instead, or create an empty array first and write data separately."
        )


def resolve_chunk_spec(
    *,
    chunks: tuple[int, ...] | Sequence[Sequence[int]] | ChunkGrid | Literal["auto"] | int,
    shards: ShardsLike | None,
    shape: tuple[int, ...],
    dtype_itemsize: int,
    zarr_format: int,
    has_data: bool = False,
) -> ResolvedChunkSpec:
    """
    Resolve chunk specification into a ChunkGrid and shards parameters.

    This function centralizes all chunk grid creation logic and error handling.
    It converts any chunk specification format into a concrete ChunkGrid instance
    and validates compatibility with:
    - Zarr format version (v2 vs v3)
    - Sharding requirements
    - Data source requirements (from_array vs init_array)

    Parameters
    ----------
    chunks : tuple[int, ...] | Sequence[Sequence[int]] | ChunkGrid | Literal["auto"] | int
        The chunks specification from the user. Can be:
        - A ChunkGrid instance (Zarr v3 only)
        - A nested sequence for variable-sized chunks (Zarr v3 only)
        - A tuple of integers for uniform chunks
        - A single integer (applied to all dimensions)
        - The literal "auto"
    shards : ShardsLike | None
        The shards specification from the user. When provided, chunks represents
        the inner chunk size within each shard, and shards represents the outer shard size.
    shape : tuple[int, ...]
        The array shape. Required for auto-chunking and validation.
    dtype_itemsize : int
        The item size of the dtype in bytes. Used for auto-chunking heuristics.
    zarr_format : {2, 3}
        The Zarr format version.
    has_data : bool, default=False
        Whether the array is being created from existing data. If True,
        RectilinearChunkGrid (variable chunks) will raise an error since
        from_array requires uniform chunks.

    Returns
    -------
    ResolvedChunkSpec
        A dataclass containing the resolved chunk_grid and shards.
        The chunk_grid is always a concrete ChunkGrid instance.

    Raises
    ------
    ValueError
        If the chunk specification is invalid for the given zarr_format,
        or if incompatible options are specified (e.g., RectilinearChunkGrid + shards,
        ChunkGrid + Zarr v2, variable chunks + sharding).
    TypeError
        If the chunks parameter has an invalid type.

    Examples
    --------
    >>> # Regular chunks, no sharding
    >>> spec = resolve_chunk_spec(
    ...     chunks=(10, 10),
    ...     shards=None,
    ...     shape=(100, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> spec.chunk_grid.chunk_shape
    (10, 10)
    >>> spec.shards is None
    True

    >>> # Sharding enabled
    >>> spec = resolve_chunk_spec(
    ...     chunks=(5, 5),
    ...     shards=(20, 20),
    ...     shape=(100, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> spec.chunk_grid.chunk_shape  # Inner chunks
    (5, 5)
    >>> spec.shards  # Outer shards
    (20, 20)

    >>> # Variable chunks (RectilinearChunkGrid)
    >>> spec = resolve_chunk_spec(
    ...     chunks=[[10, 20, 30], [25, 25, 25, 25]],
    ...     shards=None,
    ...     shape=(60, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> isinstance(spec.chunk_grid, RectilinearChunkGrid)
    True

    >>> # Error: variable chunks with Zarr v2
    >>> try:
    ...     resolve_chunk_spec(
    ...         chunks=[[10, 20], [5, 5]],
    ...         shards=None,
    ...         shape=(30, 10),
    ...         dtype_itemsize=4,
    ...         zarr_format=2
    ...     )
    ... except ValueError as e:
    ...     print(str(e))
    Variable chunks (nested sequences) are only supported in Zarr format 3...
    """
    # Step 1: Validate Zarr format compatibility
    _validate_zarr_format_compatibility(chunks, shards, zarr_format)

    # Step 2: Validate sharding compatibility
    _validate_sharding_compatibility(chunks, shards)

    # Step 3: Resolve the chunk specification to a ChunkGrid
    if shards is not None:
        # Sharding enabled: create ChunkGrid for inner chunks
        # Parse the inner chunks specification (must be regular, not variable)
        if isinstance(chunks, tuple):
            # Already normalized tuple
            inner_chunk_grid = RegularChunkGrid(chunk_shape=chunks)
        elif chunks == "auto":
            # Auto-chunk for inner chunks - use smaller target (1MB default for sharding)
            inner_chunks = _guess_chunks(shape, dtype_itemsize, max_bytes=1024 * 1024)
            inner_chunk_grid = RegularChunkGrid(chunk_shape=inner_chunks)
        elif isinstance(chunks, int):
            # Convert single int to tuple for all dimensions
            inner_chunks = _normalize_chunks(chunks, shape, dtype_itemsize)
            inner_chunk_grid = RegularChunkGrid(chunk_shape=inner_chunks)
        else:
            # This should have been caught by _validate_sharding_compatibility
            # but be defensive
            raise TypeError(
                f"Invalid chunks type when sharding is enabled: {type(chunks)}. "
                "Expected tuple, int, or 'auto'."
            )

        # Normalize shards to tuple[int, ...] for ResolvedChunkSpec
        shards_param: tuple[int, ...] | None
        if isinstance(shards, tuple):
            shards_param = shards
        elif isinstance(shards, dict):
            # ShardsConfigParam - extract the shape
            shards_param = shards.get("shape")
        else:
            # shards == "auto" or other cases
            # For "auto" shards, we pass None and let init_array handle it
            shards_param = None

        return ResolvedChunkSpec(
            chunk_grid=inner_chunk_grid,
            shards=shards_param,
        )
    else:
        # No sharding - use parse_chunk_grid to handle ChunkGrid, nested sequences, etc.
        chunk_grid = parse_chunk_grid(
            chunks, shape=shape, item_size=dtype_itemsize, zarr_format=zarr_format
        )

        # Step 4: Validate data compatibility
        _validate_data_compatibility(chunk_grid, has_data)

        # Step 5: Return the chunk_grid
        return ResolvedChunkSpec(
            chunk_grid=chunk_grid,
            shards=None,
        )
