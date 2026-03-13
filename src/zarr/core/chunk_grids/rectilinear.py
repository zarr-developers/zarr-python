from __future__ import annotations

import bisect
import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Literal, Self, TypedDict

import numpy as np
import numpy.typing as npt

from zarr.core.chunk_grids.common import ChunkEdgeLength, ChunkGrid
from zarr.core.common import (
    JSON,
    parse_named_configuration,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class RectilinearChunkGridConfigurationDict(TypedDict):
    """TypedDict for rectilinear chunk grid configuration"""

    kind: Literal["inline"]
    chunk_shapes: Sequence[Sequence[ChunkEdgeLength]]


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
                f"Chunk edge length must be int or [int, int] for run-length encoding, "
                f"got {type(item)}"
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
        if chunk_sum < dim_size:
            raise ValueError(
                f"Variable chunks along dimension {i} sum to {chunk_sum} "
                f"but array shape is {dim_size}. "
                f"Chunks must sum to be greater than or equal to the shape."
            )
        if sum(dim_chunks[:-1]) >= dim_size:
            raise ValueError(
                f"Dimension {i} has more chunks than needed. "
                f"The last chunk(s) would contain no valid data. "
                f"Remove the extra chunk(s) or increase the array shape."
            )

    return chunk_shapes


@dataclass(frozen=True)
class RectilinearChunkGrid(ChunkGrid):
    """
    A rectilinear chunk grid where chunk sizes vary along each axis.

    .. warning::
        This is an experimental feature and may change in future releases.
        Expected to stabilize in Zarr version 3.3.

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
    def from_dict(cls, data: dict[str, JSON]) -> Self:
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

    def update_shape(self, new_shape: tuple[int, ...]) -> Self:
        """
        Update the RectilinearChunkGrid to accommodate a new array shape.

        When resizing an array, this method adjusts the chunk grid to match the new shape.
        For dimensions that grow, a new chunk is added with size equal to the size difference.
        For dimensions that shrink, chunks are truncated or removed to fit the new shape.

        Parameters
        ----------
        new_shape : tuple[int, ...]
            The new shape of the array. Must have the same number of dimensions as the
            chunk grid.

        Returns
        -------
        Self
            A new RectilinearChunkGrid instance with updated chunk shapes

        Raises
        ------
        ValueError
            If the number of dimensions in new_shape doesn't match the number of dimensions
            in the chunk grid

        Examples
        --------
        >>> grid = RectilinearChunkGrid(chunk_shapes=[[10, 20], [15, 15]])
        >>> grid.update_shape((50, 40))  # Grow both dimensions
        RectilinearChunkGrid(chunk_shapes=((10, 20, 20), (15, 15, 10)))

        >>> grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
        >>> grid.update_shape((25, 30))  # Shrink first dimension
        RectilinearChunkGrid(chunk_shapes=((10, 20), (25, 25)))

        Notes
        -----
        This method is automatically called when an array is resized. The chunk size
        strategy for growing dimensions adds a single new chunk with size equal to
        the growth amount. This may not be optimal for all use cases, and users may
        want to manually adjust chunk shapes after resizing.
        """

        if len(new_shape) != len(self.chunk_shapes):
            raise ValueError(
                f"new_shape has {len(new_shape)} dimensions but "
                f"chunk_shapes has {len(self.chunk_shapes)} dimensions"
            )

        new_chunk_shapes: list[tuple[int, ...]] = []
        for dim in range(len(new_shape)):
            old_dim_length = sum(self.chunk_shapes[dim])
            new_dim_chunks: tuple[int, ...]
            if new_shape[dim] == 0:
                new_dim_chunks = ()
            elif new_shape[dim] == old_dim_length:
                new_dim_chunks = self.chunk_shapes[dim]  # no changes

            elif new_shape[dim] > old_dim_length:
                new_dim_chunks = (*self.chunk_shapes[dim], new_shape[dim] - old_dim_length)
            else:
                # drop chunk sizes that are not inside the shape anymore
                total = 0
                i = 0
                for c in self.chunk_shapes[dim]:
                    i += 1
                    total += c
                    if total >= new_shape[dim]:
                        break
                # keep the last chunk (it may be too long)
                new_dim_chunks = self.chunk_shapes[dim][:i]

            new_chunk_shapes.append(new_dim_chunks)

        return replace(self, chunk_shapes=tuple(new_chunk_shapes))

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

        # check array_shape is compatible with chunk grid
        self._validate_array_shape(array_shape)

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
        # check array_shape is compatible with chunk grid
        self._validate_array_shape(array_shape)

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
            if chunk_sum < arr_size:
                raise ValueError(
                    f"Sum of chunk sizes along axis {axis} is {chunk_sum} "
                    f"but array shape is {arr_size}. This is invalid for the "
                    "RectilinearChunkGrid."
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

        # Validate chunk coordinates are in bounds
        for axis, (coord, axis_chunks) in enumerate(
            zip(chunk_coord, self.chunk_shapes, strict=False)
        ):
            if len(axis_chunks) == 0:
                continue  # skip validation for zero-size dimensions
            if not (0 <= coord < len(axis_chunks)):
                raise IndexError(
                    f"chunk_coord[{axis}] = {coord} is out of bounds [0, {len(axis_chunks)})"
                )

        # Use cumulative sizes to get start position
        return tuple(
            0 if len(self.chunk_shapes[axis]) == 0 else self._cumulative_sizes[axis][coord]
            for axis, coord in enumerate(chunk_coord)
        )

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

        # Validate chunk coordinates are in bounds
        for axis, (coord, axis_chunks) in enumerate(
            zip(chunk_coord, self.chunk_shapes, strict=False)
        ):
            if len(axis_chunks) == 0:
                continue  # skip validation for zero-size dimensions
            if not (0 <= coord < len(axis_chunks)):
                raise IndexError(
                    f"chunk_coord[{axis}] = {coord} is out of bounds [0, {len(axis_chunks)})"
                )

        # Get shape directly from chunk_shapes
        return tuple(
            0 if len(axis_chunks) == 0 else axis_chunks[coord]
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

        # Validate array index is in bounds
        for axis, (idx, size) in enumerate(zip(array_index, array_shape, strict=False)):
            if size == 0:
                continue  # skip validation for zero-size dimensions
            if not (0 <= idx < size):
                raise IndexError(f"array_index[{axis}] = {idx} is out of bounds [0, {size})")

        # Use binary search in cumulative sizes to find chunk coordinate
        result = []
        for axis, idx in enumerate(array_index):
            if array_shape[axis] == 0:
                result.append(0)  # no chunks along zero-size dimensions
                continue
            cumsum = self._cumulative_sizes[axis]
            # bisect_right gives us the chunk index + 1, so subtract 1
            chunk_idx = bisect.bisect_right(cumsum, idx) - 1
            result.append(chunk_idx)

        return tuple(result)

    def array_indices_to_chunk_dim(
        self, array_shape: tuple[int, ...], dim: int, indices: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.intp]:
        """
        Vectorized mapping of array indices to chunk coordinates along one dimension.

        For RectilinearChunkGrid, uses np.searchsorted on cumulative sizes.
        """
        cumsum = np.asarray(self._cumulative_sizes[dim])
        return np.searchsorted(cumsum, indices, side="right").astype(np.intp) - 1

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
