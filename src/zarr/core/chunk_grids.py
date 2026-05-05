from __future__ import annotations

import bisect
import itertools
import math
import numbers
import operator
import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    NewType,
    Protocol,
    TypeGuard,
    cast,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

import zarr
from zarr.core.common import (
    ShapeLike,
    ceildiv,
    parse_shapelike,
)
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from zarr.core.array import ShardsLike
    from zarr.core.metadata import ArrayMetadata

SHARDED_INNER_CHUNK_MAX_BYTES: int = 1048576
"""Target ceiling in bytes for the auto-chunking heuristic when sharding is active (1 MiB).

Applied when `chunks` is left to auto-chunking (`None` or `"auto"`) and `shards`
is not `None`. Explicit chunk sizes are not affected by this value.
"""

ChunksTuple = NewType("ChunksTuple", tuple[np.ndarray[tuple[int], np.dtype[np.int64]], ...])
"""Normalized chunk specification: one 1D int64 array of chunk sizes per dimension.

Produced exclusively by `normalize_chunks_nd` and `guess_chunks`.
Consumers should use this type to ensure they receive validated,
canonical chunk specifications rather than raw user input.
"""


class ChunkLayout(NamedTuple):
    """Result of resolving user `chunks`/`shards` into grid metadata inputs.

    outer_chunks
        Chunk sizes for the chunk grid metadata.  When sharding is active
        these are the shard sizes; otherwise they are the user's chunk sizes.
    inner
        Recursive sub-structure inside each chunk.  `None` means the chunk is
        opaque (no sharding).  When present, `inner.outer_chunks` gives the
        sub-chunk sizes passed to `ShardingCodec`, and `inner.inner` gives
        the next level of nesting (for nested sharding), or `None`.
    """

    outer_chunks: ChunksTuple
    inner: ChunkLayout | None = None


@dataclass(frozen=True)
class FixedDimension:
    """Uniform chunk size. Boundary chunks contain less data but are
    encoded at full size by the codec pipeline."""

    size: int  # chunk edge length (>= 0)
    extent: int  # array dimension length
    nchunks: int = field(init=False, repr=False)
    ngridcells: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"FixedDimension size must be >= 0, got {self.size}")
        if self.extent < 0:
            raise ValueError(f"FixedDimension extent must be >= 0, got {self.extent}")
        if self.size == 0:
            n = 0
        else:
            n = ceildiv(self.extent, self.size)
        object.__setattr__(self, "nchunks", n)
        object.__setattr__(self, "ngridcells", n)

    def index_to_chunk(self, idx: int) -> int:
        if idx < 0:
            raise IndexError(f"Negative index {idx} is not allowed")
        if idx >= self.extent:
            raise IndexError(f"Index {idx} is out of bounds for extent {self.extent}")
        if self.size == 0:
            return 0
        return idx // self.size

    def chunk_offset(self, chunk_ix: int) -> int:
        """Byte-aligned start position of chunk *chunk_ix* in array coordinates.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, nchunks)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        return chunk_ix * self.size

    def chunk_size(self, chunk_ix: int) -> int:
        """Buffer size for codec processing — always uniform.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, nchunks)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        return self.size

    def data_size(self, chunk_ix: int) -> int:
        """Valid data region within the buffer — clipped at extent.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, nchunks)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        if self.size == 0:
            return 0
        return max(0, min(self.size, self.extent - chunk_ix * self.size))

    @property
    def _unique_edge_lengths(self) -> Iterable[int]:
        """Distinct chunk edge lengths for this dimension.

        Used by shard validation to check that every unique edge length
        is divisible by the inner chunk size. O(1) for fixed dimensions
        since there is only one edge length.
        """
        return (self.size,)

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        if self.size == 0:
            return np.zeros_like(indices)
        return indices // self.size

    def with_extent(self, new_extent: int) -> FixedDimension:
        """Re-bind to *new_extent* without modifying edges.

        Used when constructing a grid from existing metadata where edges
        are already correct. Raises on
        ``VaryingDimension`` if edges don't cover the new extent.
        """
        return FixedDimension(size=self.size, extent=new_extent)

    def resize(self, new_extent: int) -> FixedDimension:
        """Adapt for a user-initiated array resize, growing edges if needed.

        For ``FixedDimension`` this is identical to ``with_extent`` since
        regular grids don't store explicit edges.
        """
        return FixedDimension(size=self.size, extent=new_extent)

    @property
    def _size_repr(self) -> str:
        return str(self.size)


@dataclass(frozen=True)
class VaryingDimension:
    """Explicit per-chunk sizes. The last chunk may extend past the array
    extent (``extent < sum(edges)``), in which case ``data_size`` clips to
    the valid region while ``chunk_size`` returns the full edge length for
    codec processing. This underflow is allowed to match how regular grids
    handle boundary chunks, and to support shrinking an array without
    rewriting chunk edges (the spec allows trailing edges beyond the extent)."""

    edges: tuple[int, ...]  # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]  # prefix sums for O(log n) lookup
    extent: int  # array dimension length (may be < sum(edges) after resize)
    nchunks: int = field(init=False, repr=False)  # cached at construction
    ngridcells: int = field(init=False, repr=False)  # cached at construction

    # TODO(perf): for long dimensions (O(million chunks)):
    # - with_extent/resize recompute cumulative sums and nchunks from scratch;
    #   add a fast path that reuses the existing cumulative tuple.
    # - Consider storing cumulative as ndarray so bisect calls can use
    #   np.searchsorted. Scalar lookups (chunk_offset, index_to_chunk)
    #   would need benchmarking to confirm no regression.
    def __init__(self, edges: Sequence[int], extent: int) -> None:
        edges_tuple = tuple(edges)
        if not edges_tuple:
            raise ValueError("VaryingDimension edges must not be empty")
        if any(e <= 0 for e in edges_tuple):
            raise ValueError(f"All edge lengths must be > 0, got {edges_tuple}")
        cumulative = tuple(itertools.accumulate(edges_tuple))
        if extent < 0:
            raise ValueError(f"VaryingDimension extent must be >= 0, got {extent}")
        if extent > cumulative[-1]:
            raise ValueError(
                f"VaryingDimension extent {extent} exceeds sum of edges {cumulative[-1]}"
            )
        object.__setattr__(self, "edges", edges_tuple)
        object.__setattr__(self, "cumulative", cumulative)
        object.__setattr__(self, "extent", extent)
        # Cache nchunks: number of chunks that overlap [0, extent)
        if extent == 0:
            n = 0
        else:
            n = bisect.bisect_left(cumulative, extent) + 1
        object.__setattr__(self, "nchunks", n)
        object.__setattr__(self, "ngridcells", len(edges_tuple))

    def index_to_chunk(self, idx: int) -> int:
        if idx < 0 or idx >= self.extent:
            raise IndexError(f"Index {idx} out of bounds for dimension with extent {self.extent}")
        return bisect.bisect_right(self.cumulative, idx)

    def chunk_offset(self, chunk_ix: int) -> int:
        """Start position of chunk *chunk_ix* in array coordinates.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, ngridcells)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0

    def chunk_size(self, chunk_ix: int) -> int:
        """Buffer size for codec processing.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, ngridcells)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        return self.edges[chunk_ix]

    def data_size(self, chunk_ix: int) -> int:
        """Valid data region within the buffer — clipped at extent.

        Does not validate *chunk_ix* — callers must ensure it is in
        ``[0, ngridcells)``. Use ``ChunkGrid.__getitem__`` for safe access.
        """
        offset = self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0
        return max(0, min(self.edges[chunk_ix], self.extent - offset))

    @property
    def _unique_edge_lengths(self) -> Iterable[int]:
        """Distinct chunk edge lengths for this dimension (lazily deduplicated).

        Used by shard validation to check that every unique edge length
        is divisible by the inner chunk size. Lazy deduplication avoids
        materializing all edges for dimensions with many repeated sizes.
        """
        seen: set[int] = set()
        for e in self.edges:
            if e not in seen:
                seen.add(e)
                yield e

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return np.searchsorted(self.cumulative, indices, side="right")

    def with_extent(self, new_extent: int) -> VaryingDimension:
        """Re-bind to *new_extent* without modifying edges.

        Used when constructing a grid from existing metadata where edges
        are already correct. Raises if the
        existing edges don't cover *new_extent*.
        """
        edge_sum = self.cumulative[-1]
        if edge_sum < new_extent:
            raise ValueError(
                f"VaryingDimension edge sum {edge_sum} is less than new extent {new_extent}"
            )
        return VaryingDimension(self.edges, extent=new_extent)

    def resize(self, new_extent: int) -> VaryingDimension:
        """Adapt for a user-initiated array resize, growing edges if needed.

        Unlike ``with_extent``, this never fails — if *new_extent* exceeds
        the current edge sum, a new chunk is appended to cover the gap.
        Shrinking preserves all edges (the spec allows trailing edges
        beyond the array extent).
        """
        if new_extent == self.extent:
            return self
        elif new_extent > self.cumulative[-1]:
            expanded_edges = list(self.edges) + [new_extent - self.cumulative[-1]]
            return VaryingDimension(expanded_edges, extent=new_extent)
        else:
            return VaryingDimension(self.edges, extent=new_extent)

    @property
    def _size_repr(self) -> str:
        return repr(tuple(self.edges))


@runtime_checkable
class DimensionGrid(Protocol):
    """Structural interface shared by FixedDimension and VaryingDimension."""

    @property
    def nchunks(self) -> int: ...
    @property
    def ngridcells(self) -> int: ...
    @property
    def extent(self) -> int: ...
    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...
    def chunk_size(self, chunk_ix: int) -> int: ...
    def data_size(self, chunk_ix: int) -> int: ...
    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]: ...
    @property
    def _unique_edge_lengths(self) -> Iterable[int]: ...
    def with_extent(self, new_extent: int) -> DimensionGrid: ...
    def resize(self, new_extent: int) -> DimensionGrid: ...
    @property
    def _size_repr(self) -> str: ...


@dataclass(frozen=True)
class ChunkSpec:
    """Specification of a single chunk's location and size.

    ``slices`` gives the valid data region in array coordinates.
    ``codec_shape`` gives the buffer shape for codec processing.
    For interior chunks these are equal. For boundary chunks of a regular
    grid, ``codec_shape`` is the full declared chunk size while ``shape``
    is clipped. For rectilinear grids, ``shape == codec_shape`` unless the
    last chunk extends past the array extent.
    """

    slices: tuple[slice, ...]
    codec_shape: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.slices)

    @property
    def is_boundary(self) -> bool:
        return self.shape != self.codec_shape


# A single dimension's rectilinear chunk spec: bare int (uniform shorthand),
# list of ints (explicit edges), or mixed RLE (e.g. [[10, 3], 5]).


def _is_rectilinear_chunks(chunks: Any) -> TypeGuard[Sequence[Sequence[int]]]:
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

    _dimensions: tuple[DimensionGrid, ...]
    _is_regular: bool

    def __init__(self, *, dimensions: tuple[DimensionGrid, ...]) -> None:
        object.__setattr__(self, "_dimensions", dimensions)
        object.__setattr__(
            self, "_is_regular", all(isinstance(d, FixedDimension) for d in dimensions)
        )

    def __repr__(self) -> str:
        sizes = ", ".join(d._size_repr for d in self._dimensions)
        shape = tuple(d.extent for d in self._dimensions)
        return f"ChunkGrid(chunk_sizes=({sizes}), array_shape={shape})"

    @classmethod
    def from_metadata(cls, metadata: ArrayMetadata) -> ChunkGrid:
        """Construct a ChunkGrid from array metadata.

        For v2 metadata, builds from shape and chunks.
        For v3 metadata, dispatches on the chunk grid type.
        """
        from zarr.core.metadata import ArrayV2Metadata
        from zarr.core.metadata.v3 import RectilinearChunkGridMetadata, RegularChunkGridMetadata

        if isinstance(metadata, ArrayV2Metadata):
            return cls.from_sizes(metadata.shape, tuple(metadata.chunks))
        chunk_grid_meta = metadata.chunk_grid
        if isinstance(chunk_grid_meta, RegularChunkGridMetadata):
            return cls.from_sizes(metadata.shape, tuple(chunk_grid_meta.chunk_shape))
        elif isinstance(chunk_grid_meta, RectilinearChunkGridMetadata):
            return cls.from_sizes(metadata.shape, chunk_grid_meta.chunk_shapes)
        else:
            raise TypeError(f"Unknown chunk grid metadata type: {type(chunk_grid_meta)}")

    @classmethod
    def from_sizes(
        cls,
        array_shape: ShapeLike,
        chunk_sizes: Sequence[int | Sequence[int]],
    ) -> ChunkGrid:
        """Create a ChunkGrid from per-dimension chunk size specifications.

        Parameters
        ----------
        array_shape
            The array shape (one extent per dimension).
        chunk_sizes
            Per-dimension chunk sizes. Each element is either:

            - An ``int`` — regular (fixed) chunk size for that dimension.
            - A ``Sequence[int]`` — explicit per-chunk edge lengths. If all
              edges are identical and cover the extent, the dimension is
              stored as ``FixedDimension``; otherwise as ``VaryingDimension``.
        """
        extents = parse_shapelike(array_shape)
        if len(extents) != len(chunk_sizes):
            raise ValueError(
                f"array_shape has {len(extents)} dimensions but chunk_sizes "
                f"has {len(chunk_sizes)} dimensions"
            )
        dims: list[DimensionGrid] = []
        for dim_spec, extent in zip(chunk_sizes, extents, strict=True):
            if isinstance(dim_spec, int):
                dims.append(FixedDimension(size=dim_spec, extent=extent))
            else:
                edges_list = list(dim_spec)
                if not edges_list:
                    raise ValueError("Each dimension must have at least one chunk")
                edge_sum = sum(edges_list)
                if (
                    edges_list[0] > 0
                    and all(e == edges_list[0] for e in edges_list)
                    and (extent == edge_sum or len(edges_list) == ceildiv(extent, edges_list[0]))
                ):
                    dims.append(FixedDimension(size=edges_list[0], extent=extent))
                else:
                    dims.append(VaryingDimension(edges_list, extent=extent))
        return cls(dimensions=tuple(dims))

    # -- Properties --

    @property
    def ndim(self) -> int:
        return len(self._dimensions)

    @property
    def is_regular(self) -> bool:
        return self._is_regular

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Number of chunks per dimension."""
        return tuple(d.nchunks for d in self._dimensions)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the uniform chunk shape. Raises if grid is not regular."""
        if not self.is_regular:
            raise ValueError(
                "chunk_shape is only available for regular chunk grids. "
                "Use grid[coords] for per-chunk sizes."
            )
        return tuple(d.size for d in self._dimensions if isinstance(d, FixedDimension))

    @property
    def chunk_sizes(self) -> tuple[tuple[int, ...], ...]:
        """Per-dimension chunk sizes, including the final boundary chunk.

        Returns the actual data size of each chunk (clipped at the array
        extent), matching the dask ``Array.chunks`` convention.  Works for
        both regular and rectilinear grids.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            One inner tuple per dimension, each containing the data size
            of every chunk along that dimension.
        """
        return tuple(tuple(d.data_size(i) for i in range(d.nchunks)) for d in self._dimensions)

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
        for dim, ix in zip(self._dimensions, coords, strict=True):
            if ix < 0 or ix >= dim.nchunks:
                return None
            offset = dim.chunk_offset(ix)
            slices.append(slice(offset, offset + dim.data_size(ix), 1))
            codec_shape.append(dim.chunk_size(ix))
        return ChunkSpec(tuple(slices), tuple(codec_shape))

    def __iter__(self) -> Iterator[ChunkSpec]:
        """Iterate all chunks, yielding ChunkSpec for each."""
        for coords in itertools.product(*(range(d.nchunks) for d in self._dimensions)):
            spec = self[coords]
            if spec is not None:
                yield spec

    def all_chunk_coords(
        self,
        *,
        origin: Sequence[int] | None = None,
        selection_shape: Sequence[int] | None = None,
    ) -> Iterator[tuple[int, ...]]:
        """Iterate over chunk coordinates, optionally restricted to a subregion.

        Parameters
        ----------
        origin : Sequence[int] | None
            The first chunk coordinate to return. Defaults to the grid origin.
        selection_shape : Sequence[int] | None
            The number of chunks per dimension to iterate. Defaults to the
            remaining extent from origin.
        """
        if origin is None:
            origin_parsed = (0,) * self.ndim
        else:
            origin_parsed = tuple(origin)
        if selection_shape is None:
            selection_shape_parsed = tuple(
                g - o for o, g in zip(origin_parsed, self.grid_shape, strict=True)
            )
        else:
            selection_shape_parsed = tuple(selection_shape)
        ranges = tuple(
            range(o, o + s) for o, s in zip(origin_parsed, selection_shape_parsed, strict=True)
        )
        return itertools.product(*ranges)

    def iter_chunk_regions(
        self,
        *,
        origin: Sequence[int] | None = None,
        selection_shape: Sequence[int] | None = None,
    ) -> Iterator[tuple[slice, ...]]:
        """Iterate over the data regions (slices) spanned by each chunk.

        Parameters
        ----------
        origin : Sequence[int] | None
            The first chunk coordinate to return. Defaults to the grid origin.
        selection_shape : Sequence[int] | None
            The number of chunks per dimension to iterate. Defaults to the
            remaining extent from origin.
        """
        for coords in self.all_chunk_coords(origin=origin, selection_shape=selection_shape):
            spec = self[coords]
            if spec is not None:
                yield spec.slices

    def get_nchunks(self) -> int:
        return reduce(operator.mul, (d.nchunks for d in self._dimensions), 1)

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
        dims = tuple(
            dim.resize(new_extent)
            for dim, new_extent in zip(self._dimensions, new_shape, strict=True)
        )
        return ChunkGrid(dimensions=dims)


def _guess_regular_chunks(
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


def normalize_chunks_1d(
    chunks: int | Iterable[object], span: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """
    Normalize a one-dimensional chunk specification into a 1D int64 array of
    chunk sizes that cover the span.

    `-1` means "one chunk covering the entire span."
    For an integer chunk size, all chunks are uniform — the last chunk may
    overhang the span. The actual data extent of each chunk is determined
    by the chunk grid at runtime, not by this function.
    """
    if chunks == -1:
        return np.array([span], dtype=np.int64)
    if isinstance(chunks, int):
        if chunks <= 0:
            raise ValueError(f"Chunk size must be positive, got {chunks}")
        if span == 0:
            return np.array([chunks], dtype=np.int64)
        n = ceildiv(span, chunks)
        return np.full(n, chunks, dtype=np.int64)
    else:
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Chunk specification must not be empty")
        non_int = [
            (idx, c) for idx, c in enumerate(chunk_list) if not isinstance(c, numbers.Integral)
        ]
        if non_int:
            non_int_idxs, non_int_vals = [*zip(*non_int, strict=False)]
            raise TypeError(
                f"Each chunk size must be an integer; got non-integer element(s) {non_int_vals!r} "
                f"at indices {non_int_idxs!r}. Chunk sizes must be declared as a flat sequence of "
                f"positive integers (e.g. [3, 3, 1])."
            )
        ints: list[int] = [int(c) for c in chunk_list]  # type: ignore[call-overload]
        if any(c <= 0 for c in ints):
            raise ValueError(f"All chunk sizes must be positive, got {ints}")
        if sum(ints) != span:
            raise ValueError(f"Chunk sizes {ints} do not sum to span {span}")
        return np.asarray(ints, dtype=np.int64)


def normalize_chunks_nd(
    chunks: Any,
    shape: tuple[int, ...],
) -> ChunksTuple:
    """
    Normalize a chunk specification into a `ChunksTuple`.

    This is a mechanical transformation — no heuristics, no guessing.
    Handles `False` ("no chunking"), scalar ints, `-1` sentinels (one chunk
    per dimension covering the full span), and explicit per-dimension lists
    of chunk sizes (regular or rectilinear).

    For auto-chunking, use `guess_chunks` which returns a
    `ChunksTuple` directly. `chunks=None` and `chunks=True` are rejected
    here — the caller is responsible for choosing between explicit sizes
    and auto-chunking.
    """
    if chunks is None or chunks is True:
        raise ValueError(
            f"normalize_chunks_nd does not accept {chunks!r}. Use guess_chunks() for auto-chunking."
        )

    # handle no chunking
    if chunks is False:
        return ChunksTuple(tuple(np.array([s], dtype=np.int64) for s in shape))

    # handle 1D convenience form. bool is excluded above so this only catches actual ints.
    if isinstance(chunks, numbers.Integral):
        chunks = tuple(int(chunks) for _ in shape)

    # handle bad dimensionality
    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks has {len(chunks)} dimensions but shape has {len(shape)} dimensions"
        )

    return ChunksTuple(
        tuple(normalize_chunks_1d(c, span=s) for c, s in zip(chunks, shape, strict=True))
    )


def guess_chunks(
    shape: tuple[int, ...], typesize: int, *, max_bytes: int | None = None
) -> ChunksTuple:
    """
    Heuristically determine chunk sizes for an array.

    This is the policy function — it makes opinionated choices about
    chunk sizes based on array shape and element size, and returns a
    normalized `ChunksTuple`.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    typesize : int
        Size of one element in bytes.
    max_bytes : int or None
        Target maximum chunk size in bytes. If None, uses the default
        heuristic from `_guess_regular_chunks`.
    """
    if max_bytes is not None:
        flat = _guess_regular_chunks(shape, typesize, max_bytes=max_bytes)
    else:
        flat = _guess_regular_chunks(shape, typesize)
    return normalize_chunks_nd(flat, shape)


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


def resolve_outer_and_inner_chunks(
    *,
    array_shape: tuple[int, ...],
    chunks: ChunksTuple,
    shard_shape: ShardsLike | None,
    item_size: int,
) -> ChunkLayout:
    """Resolve user `chunks`/`shards` into outer and inner chunk specs.

    Parameters
    ----------
    array_shape
        The array shape.
    chunks
        Normalized chunk specification (the user's `chunks=`).
    shard_shape
        Raw shard specification (the user's `shards=`).
        `None` means no sharding, `"auto"` triggers heuristic inference,
        a nested sequence is treated as rectilinear shard boundaries,
        and anything else is used as a regular shard shape.
    item_size
        Element size in bytes.

    Returns
    -------
    ChunkLayout
        `outer_chunks` is the `ChunksTuple` for chunk grid
        metadata.  `inner` holds the sub-chunk structure for
        `ShardingCodec`, or is `None` when sharding is not active.
    """
    if shard_shape is None:
        return ChunkLayout(outer_chunks=chunks)

    # Rectilinear shards: normalize the nested sequence directly.
    if _is_rectilinear_chunks(shard_shape):
        outer = normalize_chunks_nd(shard_shape, array_shape)
        return ChunkLayout(outer_chunks=outer, inner=ChunkLayout(outer_chunks=chunks))

    # Extract the flat chunk shape (first size per dimension) for arithmetic.
    chunk_shape_flat = tuple(int(dim[0]) for dim in chunks)

    if shard_shape == "auto":
        warnings.warn(
            "Automatic shard shape inference is experimental and may change without notice.",
            ZarrUserWarning,
            stacklevel=2,
        )
        _shards_out: tuple[int, ...] = ()
        target_shard_size_bytes = zarr.config.get("array.target_shard_size_bytes", None)
        num_chunks_per_shard_axis = (
            _guess_num_chunks_per_axis_shard(
                chunk_shape=chunk_shape_flat,
                item_size=item_size,
                max_bytes=target_shard_size_bytes,
                array_shape=array_shape,
            )
            if (has_auto_shard := (target_shard_size_bytes is not None))
            else 2
        )
        for a_shape, c_shape in zip(array_shape, chunk_shape_flat, strict=True):
            can_shard_axis = a_shape // c_shape > 8 if not has_auto_shard else True
            if can_shard_axis:
                _shards_out += (c_shape * num_chunks_per_shard_axis,)
            else:
                _shards_out += (c_shape,)
        shard_flat = _shards_out
    elif isinstance(shard_shape, dict):
        shard_flat = tuple(shard_shape["shape"])
    else:
        shard_flat = cast("tuple[int, ...]", shard_shape)

    outer = normalize_chunks_nd(shard_flat, array_shape)
    return ChunkLayout(outer_chunks=outer, inner=ChunkLayout(outer_chunks=chunks))
