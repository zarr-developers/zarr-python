from __future__ import annotations

import bisect
import itertools
import math
import numbers
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

import zarr
from zarr.core.common import (
    JSON,
    NamedConfig,
    ShapeLike,
    ceildiv,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.array import ShardsLike


# ---------------------------------------------------------------------------
# Per-dimension grid types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedDimension:
    """Uniform chunk size. Boundary chunks contain less data but are
    encoded at full size by the codec pipeline."""

    size: int  # chunk edge length (>= 0)
    extent: int  # array dimension length

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"FixedDimension size must be >= 0, got {self.size}")
        if self.extent < 0:
            raise ValueError(f"FixedDimension extent must be >= 0, got {self.extent}")

    @property
    def nchunks(self) -> int:
        if self.size == 0:
            return 1 if self.extent == 0 else 0
        return ceildiv(self.extent, self.size)

    def index_to_chunk(self, idx: int) -> int:
        if self.size == 0:
            return 0
        return idx // self.size

    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.size

    def chunk_size(self, chunk_ix: int) -> int:
        """Buffer size for codec processing — always uniform."""
        return self.size

    def data_size(self, chunk_ix: int) -> int:
        """Valid data region within the buffer — clipped at extent."""
        if self.size == 0:
            return 0
        return max(0, min(self.size, self.extent - chunk_ix * self.size))

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        if self.size == 0:
            return np.zeros_like(indices)
        return indices // self.size


@dataclass(frozen=True)
class VaryingDimension:
    """Explicit per-chunk sizes. No padding — each edge length is
    both the codec size and the data size."""

    edges: tuple[int, ...]  # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]  # prefix sums for O(log n) lookup

    def __init__(self, edges: Sequence[int]) -> None:
        edges_tuple = tuple(edges)
        if not edges_tuple:
            raise ValueError("VaryingDimension edges must not be empty")
        if any(e <= 0 for e in edges_tuple):
            raise ValueError(f"All edge lengths must be > 0, got {edges_tuple}")
        cumulative = tuple(itertools.accumulate(edges_tuple))
        object.__setattr__(self, "edges", edges_tuple)
        object.__setattr__(self, "cumulative", cumulative)

    @property
    def nchunks(self) -> int:
        return len(self.edges)

    @property
    def extent(self) -> int:
        return self.cumulative[-1]

    def index_to_chunk(self, idx: int) -> int:
        return bisect.bisect_right(self.cumulative, idx)

    def chunk_offset(self, chunk_ix: int) -> int:
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0

    def chunk_size(self, chunk_ix: int) -> int:
        """Buffer size for codec processing."""
        return self.edges[chunk_ix]

    def data_size(self, chunk_ix: int) -> int:
        """Valid data region — same as chunk_size for varying dims."""
        return self.edges[chunk_ix]

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return np.searchsorted(self.cumulative, indices, side="right")


@runtime_checkable
class DimensionGrid(Protocol):
    """Structural interface shared by FixedDimension and VaryingDimension."""

    @property
    def nchunks(self) -> int: ...
    @property
    def extent(self) -> int: ...
    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...
    def chunk_size(self, chunk_ix: int) -> int: ...
    def data_size(self, chunk_ix: int) -> int: ...
    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]: ...


# ---------------------------------------------------------------------------
# ChunkSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    """Specification of a single chunk's location and size.

    ``slices`` gives the valid data region in array coordinates.
    ``codec_shape`` gives the buffer shape for codec processing.
    For interior chunks these are equal. For boundary chunks of a regular
    grid, ``codec_shape`` is the full declared chunk size while ``shape``
    is clipped. For rectilinear grids, ``shape == codec_shape`` always.
    """

    slices: tuple[slice, ...]
    codec_shape: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.slices)

    @property
    def is_boundary(self) -> bool:
        return self.shape != self.codec_shape


# ---------------------------------------------------------------------------
# RLE helpers (ported from #3534)
# ---------------------------------------------------------------------------


def _expand_rle(data: Sequence[list[int] | int]) -> list[int]:
    """Expand a mixed array of bare integers and RLE pairs.

    Per the rectilinear chunk grid spec, each element can be:
    - a bare integer (an explicit edge length)
    - a two-element array ``[value, count]`` (run-length encoded)
    """
    result: list[int] = []
    for item in data:
        if isinstance(item, int):
            result.append(item)
        elif len(item) == 2:
            size, count = item
            result.extend([size] * count)
        else:
            raise ValueError(f"RLE entries must be an integer or [size, count], got {item}")
    return result


def _compress_rle(sizes: Sequence[int]) -> list[list[int] | int]:
    """Compress chunk sizes to mixed RLE format per the rectilinear spec.

    Runs of length > 1 are emitted as ``[value, count]`` pairs; runs of
    length 1 are emitted as bare integers::

        [10, 10, 10, 5] -> [[10, 3], 5]
    """
    if not sizes:
        return []
    result: list[list[int] | int] = []
    current = sizes[0]
    count = 1
    for s in sizes[1:]:
        if s == current:
            count += 1
        else:
            result.append([current, count] if count > 1 else current)
            current = s
            count = 1
    result.append([current, count] if count > 1 else current)
    return result


def _validate_rectilinear_kind(configuration: dict[str, JSON]) -> None:
    """Validate the ``kind`` field of a rectilinear chunk grid configuration.

    The spec requires ``kind: "inline"``.
    """
    kind = configuration.get("kind")
    if kind is None:
        raise ValueError(
            "Rectilinear chunk grid configuration requires a 'kind' field. "
            "Only 'inline' is currently supported."
        )
    if kind != "inline":
        raise ValueError(
            f"Unsupported rectilinear chunk grid kind: {kind!r}. "
            f"Only 'inline' is currently supported."
        )


def _decode_dim_spec(dim_spec: JSON, array_extent: int | None = None) -> list[int]:
    """Decode a single dimension's chunk edge specification per the rectilinear spec.

    Per the spec, each element of ``chunk_shapes`` can be:
    - a bare integer ``m``: repeat ``m`` until the sum >= array extent
    - an array of bare integers and/or ``[value, count]`` RLE pairs

    Parameters
    ----------
    dim_spec
        The raw JSON value for one dimension's chunk edges.
    array_extent
        Array length along this dimension. Required when *dim_spec* is a bare
        integer (to know how many repetitions).
    """
    if isinstance(dim_spec, int):
        if array_extent is None:
            raise ValueError("Integer chunk_shapes shorthand requires array shape to expand.")
        if dim_spec <= 0:
            raise ValueError(f"Integer chunk edge length must be > 0, got {dim_spec}")
        n = ceildiv(array_extent, dim_spec)
        return [dim_spec] * n
    if isinstance(dim_spec, list):
        # Check if the list contains any sub-lists (RLE pairs) or is all bare ints
        has_sublists = any(isinstance(e, list) for e in dim_spec)
        if has_sublists:
            return _expand_rle(dim_spec)
        else:
            # All bare integers — explicit edge lengths
            return [int(e) for e in dim_spec]
    raise ValueError(f"Invalid chunk_shapes entry: {dim_spec}")


# ---------------------------------------------------------------------------
# Unified ChunkGrid
# ---------------------------------------------------------------------------

# Type alias for what users can pass as chunks to create_array
ChunksLike = tuple[int, ...] | list[list[int] | int] | int


def _is_rectilinear_chunks(chunks: Any) -> bool:
    """Check if chunks is a nested sequence (e.g. [[10, 20], [5, 5]]).

    Returns True for inputs like [[10, 20], [5, 5]] or [(10, 20), (5, 5)].
    Returns False for flat sequences like (10, 10) or [10, 10].
    """
    if isinstance(chunks, (str, int, ChunkGrid)):
        return False
    if not hasattr(chunks, "__iter__"):
        return False
    try:
        first_elem = next(iter(chunks), None)
        if first_elem is None:
            return False
        return hasattr(first_elem, "__iter__") and not isinstance(first_elem, (str, bytes, int))
    except (TypeError, StopIteration):
        return False


@dataclass(frozen=True)
class ChunkGrid:
    """
    Unified chunk grid supporting both regular and rectilinear chunking.

    A chunk grid is a concrete arrangement of chunks for a specific array.
    It stores the extent (array dimension length) per dimension, enabling
    ``grid[coords]`` to return a ``ChunkSpec`` without external parameters.

    Internally represents each dimension as either FixedDimension (uniform chunks)
    or VaryingDimension (per-chunk edge lengths with prefix sums).
    """

    dimensions: tuple[DimensionGrid, ...]
    _is_regular: bool

    def __init__(self, *, dimensions: tuple[DimensionGrid, ...]) -> None:
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(
            self, "_is_regular", all(isinstance(d, FixedDimension) for d in dimensions)
        )

    @classmethod
    def from_regular(cls, array_shape: ShapeLike, chunk_shape: ShapeLike) -> ChunkGrid:
        """Create a ChunkGrid where all dimensions are fixed (regular)."""
        shape_parsed = parse_shapelike(array_shape)
        chunks_parsed = parse_shapelike(chunk_shape)
        if len(shape_parsed) != len(chunks_parsed):
            raise ValueError(
                f"array_shape and chunk_shape must have same ndim, "
                f"got {len(shape_parsed)} vs {len(chunks_parsed)}"
            )
        dims = tuple(
            FixedDimension(size=c, extent=s)
            for s, c in zip(shape_parsed, chunks_parsed, strict=True)
        )
        return cls(dimensions=dims)

    @classmethod
    def from_rectilinear(cls, chunk_shapes: Sequence[Sequence[int]]) -> ChunkGrid:
        """Create a ChunkGrid with per-dimension edge lists.

        Each element of chunk_shapes is a sequence of chunk sizes for that dimension.
        If all sizes in a dimension are identical, it's stored as FixedDimension.
        The extent of each dimension is ``sum(edges)``.
        """
        dims: list[DimensionGrid] = []
        for edges in chunk_shapes:
            edges_list = list(edges)
            if not edges_list:
                raise ValueError("Each dimension must have at least one chunk")
            extent = sum(edges_list)
            if all(e == edges_list[0] for e in edges_list):
                dims.append(FixedDimension(size=edges_list[0], extent=extent))
            else:
                dims.append(VaryingDimension(edges_list))
        return cls(dimensions=tuple(dims))

    # -- Properties --

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    @property
    def is_regular(self) -> bool:
        return self._is_regular

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of chunks per dimension."""
        return tuple(d.nchunks for d in self.dimensions)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the uniform chunk shape. Raises if grid is not regular."""
        if not self.is_regular:
            raise ValueError(
                "chunk_shape is only available for regular chunk grids. "
                "Use grid[coords] for per-chunk sizes."
            )
        return tuple(
            d.size
            for d in self.dimensions
            if isinstance(d, FixedDimension)  # guaranteed by is_regular
        )

    # -- Collection interface --

    def __getitem__(self, coords: int | tuple[int, ...]) -> ChunkSpec | None:
        """Return the ChunkSpec for a chunk at the given grid position, or None if OOB."""
        if isinstance(coords, int):
            coords = (coords,)
        if len(coords) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} coordinate(s) for a {self.ndim}-d chunk grid, "
                f"got {len(coords)}."
            )
        slices: list[slice] = []
        codec_shape: list[int] = []
        for dim, ix in zip(self.dimensions, coords, strict=True):
            if ix < 0 or ix >= dim.nchunks:
                return None
            offset = dim.chunk_offset(ix)
            slices.append(slice(offset, offset + dim.data_size(ix)))
            codec_shape.append(dim.chunk_size(ix))
        return ChunkSpec(tuple(slices), tuple(codec_shape))

    def __iter__(self) -> Iterator[ChunkSpec]:
        """Iterate all chunks, yielding ChunkSpec for each."""
        for coords in itertools.product(*(range(d.nchunks) for d in self.dimensions)):
            spec = self[coords]
            if spec is not None:
                yield spec

    def all_chunk_coords(self) -> Iterator[tuple[int, ...]]:
        return itertools.product(*(range(d.nchunks) for d in self.dimensions))

    def get_nchunks(self) -> int:
        return reduce(operator.mul, (d.nchunks for d in self.dimensions), 1)

    # -- Resize --

    def update_shape(self, new_shape: tuple[int, ...]) -> ChunkGrid:
        """Return a new ChunkGrid adjusted for *new_shape*.

        For regular (FixedDimension) axes the extent is simply re-bound.
        For varying (VaryingDimension) axes:
        * **grow**: a new chunk whose size equals the growth is appended.
        * **shrink**: trailing chunks that lie entirely beyond *new_shape* are
          dropped; the last retained chunk is the one whose cumulative offset
          first reaches or exceeds the new extent.
        * **no change**: the dimension is kept as-is.

        Raises
        ------
        ValueError
            If *new_shape* has the wrong number of dimensions.
        """
        if len(new_shape) != self.ndim:
            raise ValueError(
                f"new_shape has {len(new_shape)} dimensions but "
                f"chunk grid has {self.ndim} dimensions"
            )
        dims: list[DimensionGrid] = []
        for dim, new_extent in zip(self.dimensions, new_shape, strict=True):
            if isinstance(dim, FixedDimension):
                dims.append(FixedDimension(size=dim.size, extent=new_extent))
            elif isinstance(dim, VaryingDimension):
                old_extent = dim.extent
                if new_extent == old_extent:
                    dims.append(dim)
                elif new_extent > old_extent:
                    expanded_edges = list(dim.edges) + [new_extent - old_extent]
                    dims.append(VaryingDimension(expanded_edges))
                else:
                    # Shrink: keep chunks whose cumulative offset covers new_extent
                    shrunk_edges: list[int] = []
                    total = 0
                    for edge in dim.edges:
                        shrunk_edges.append(edge)
                        total += edge
                        if total >= new_extent:
                            break
                    dims.append(VaryingDimension(shrunk_edges))
            else:
                raise TypeError(f"Unexpected dimension type: {type(dim)}")
        return ChunkGrid(dimensions=tuple(dims))

    # ChunkGrid does not serialize itself. The format choice ("regular" vs
    # "rectilinear") belongs to the metadata layer. Use serialize_chunk_grid()
    # for output and parse_chunk_grid() for input.


def parse_chunk_grid(
    data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any],
    array_shape: tuple[int, ...],
) -> ChunkGrid:
    """Create a ChunkGrid from a metadata dict or existing grid, binding array shape.

    This is the primary entry point for constructing a ChunkGrid from serialized
    metadata. It always produces a grid with correct extent values.
    """
    if isinstance(data, ChunkGrid):
        # Re-bind extent if array_shape differs from what's stored
        dims: list[DimensionGrid] = []
        for dim, extent in zip(data.dimensions, array_shape, strict=True):
            if isinstance(dim, FixedDimension):
                dims.append(FixedDimension(size=dim.size, extent=extent))
            elif isinstance(dim, VaryingDimension):
                # VaryingDimension has intrinsic extent (sum of edges).
                # After resize/shrink the last chunk may extend past the array
                # boundary, so extent >= array_shape is valid (like regular grids).
                if dim.extent < extent:
                    raise ValueError(
                        f"VaryingDimension extent {dim.extent} is less than "
                        f"array shape extent {extent} for dimension {len(dims)}"
                    )
                dims.append(dim)
            else:
                raise TypeError(f"Unexpected dimension type: {type(dim)}")
        return ChunkGrid(dimensions=tuple(dims))

    name_parsed, configuration_parsed = parse_named_configuration(data)

    if name_parsed == "regular":
        chunk_shape_raw = configuration_parsed.get("chunk_shape")
        if chunk_shape_raw is None:
            raise ValueError("Regular chunk grid requires 'chunk_shape' configuration")
        if not isinstance(chunk_shape_raw, Sequence):
            raise TypeError(f"chunk_shape must be a sequence, got {type(chunk_shape_raw)}")
        return ChunkGrid.from_regular(array_shape, cast("Sequence[int]", chunk_shape_raw))

    if name_parsed == "rectilinear":
        _validate_rectilinear_kind(configuration_parsed)
        chunk_shapes_raw = configuration_parsed.get("chunk_shapes")
        if chunk_shapes_raw is None:
            raise ValueError("Rectilinear chunk grid requires 'chunk_shapes' configuration")
        if not isinstance(chunk_shapes_raw, Sequence):
            raise TypeError(f"chunk_shapes must be a sequence, got {type(chunk_shapes_raw)}")
        if len(chunk_shapes_raw) != len(array_shape):
            raise ValueError(
                f"chunk_shapes has {len(chunk_shapes_raw)} dimensions but array shape "
                f"has {len(array_shape)} dimensions"
            )
        decoded: list[list[int]] = []
        for dim_spec, extent in zip(chunk_shapes_raw, array_shape, strict=True):
            decoded.append(_decode_dim_spec(dim_spec, array_extent=extent))
        for i, (edges, extent) in enumerate(zip(decoded, array_shape, strict=True)):
            edge_sum = sum(edges)
            if edge_sum < extent:
                raise ValueError(
                    f"Rectilinear chunk edges for dimension {i} sum to {edge_sum} "
                    f"but array shape extent is {extent} (edge sum must be >= extent)"
                )
        return ChunkGrid.from_rectilinear(decoded)

    raise ValueError(f"Unknown chunk grid name: {name_parsed!r}")


def serialize_chunk_grid(grid: ChunkGrid, name: str) -> dict[str, JSON]:
    """Serialize a ChunkGrid to a metadata dict using the given format name.

    The format choice ("regular" vs "rectilinear") belongs to the metadata layer,
    not the grid itself. This function is called by ArrayV3Metadata.to_dict().
    """
    if name == "regular":
        if not grid.is_regular:
            raise ValueError(
                "Cannot serialize a non-regular chunk grid as 'regular'. Use 'rectilinear' instead."
            )
        return {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(grid.chunk_shape)},
        }

    if name == "rectilinear":
        chunk_shapes: list[Any] = []
        for dim in grid.dimensions:
            if isinstance(dim, FixedDimension):
                # Produce the most compact spec representation.
                n = dim.nchunks
                if n == 0:
                    chunk_shapes.append([])
                else:
                    last_data = dim.extent - (n - 1) * dim.size
                    if last_data == dim.size:
                        # All chunks uniform → integer shorthand
                        chunk_shapes.append(dim.size)
                    elif n == 1:
                        # Single boundary chunk → bare integer
                        chunk_shapes.append([last_data])
                    elif n == 2:
                        # One full chunk + one boundary → bare integers
                        chunk_shapes.append([dim.size, last_data])
                    else:
                        # RLE for the uniform run + bare int for boundary
                        chunk_shapes.append([[dim.size, n - 1], last_data])
            elif isinstance(dim, VaryingDimension):
                edges = list(dim.edges)
                rle = _compress_rle(edges)
                if len(rle) < len(edges):
                    chunk_shapes.append(rle)
                else:
                    chunk_shapes.append(edges)
        return {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
        }

    raise ValueError(f"Unknown chunk grid name for serialization: {name!r}")


def _infer_chunk_grid_name(
    data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any],
    grid: ChunkGrid,
) -> str:
    """Extract or infer the chunk grid serialization name from the input."""
    if isinstance(data, dict):
        name, _ = parse_named_configuration(data)
        return name
    # ChunkGrid passed directly — infer from structure
    return "regular" if grid.is_regular else "rectilinear"


# ---------------------------------------------------------------------------
# Chunk guessing / normalization (unchanged)
# ---------------------------------------------------------------------------


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
                # The previous heuristic was `a_shape // c_shape > 8` and now, with target_shard_size_bytes, we only check that the shard size is less than the array size.
                can_shard_axis = a_shape // c_shape > 8 if not has_auto_shard else True
                if can_shard_axis:
                    _shards_out += (c_shape * num_chunks_per_shard_axis,)
                else:
                    _shards_out += (c_shape,)
        elif isinstance(shard_shape, dict):
            _shards_out = tuple(shard_shape["shape"])
        else:
            _shards_out = cast("tuple[int, ...]", shard_shape)

    return _shards_out, _chunks_out
