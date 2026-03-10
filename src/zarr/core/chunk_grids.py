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
    from typing import Self

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


@dataclass(frozen=True)
class TiledDimension:
    """Periodic chunk pattern repeated N times, with an optional trailing remainder.

    Exploits periodicity for O(1) chunk_offset/chunk_size and O(log pattern_len)
    index_to_chunk, regardless of total chunk count. Memory is O(pattern_len)
    instead of O(n_chunks).

    Example: 30 years of monthly chunks (days per month):
        TiledDimension(pattern=(31,28,31,30,31,30,31,31,30,31,30,31), repeats=30)
    """

    pattern: tuple[int, ...]  # one period's edge lengths (all > 0)
    repeats: int  # number of full repetitions (>= 1)
    remainder: tuple[int, ...]  # trailing partial period (all > 0, may be empty)

    # Precomputed
    _pattern_cumulative: tuple[int, ...]  # prefix sums within one period
    _period_extent: int  # sum(pattern)
    _pattern_nchunks: int  # len(pattern)
    _remainder_cumulative: tuple[int, ...]  # prefix sums of remainder
    _total_nchunks: int
    _total_extent: int

    def __init__(
        self,
        pattern: Sequence[int],
        repeats: int = 1,
        remainder: Sequence[int] = (),
    ) -> None:
        pattern_t = tuple(pattern)
        remainder_t = tuple(remainder)
        if not pattern_t:
            raise ValueError("TiledDimension pattern must not be empty")
        if repeats < 1:
            raise ValueError(f"TiledDimension repeats must be >= 1, got {repeats}")
        if any(e <= 0 for e in pattern_t):
            raise ValueError(f"All pattern edge lengths must be > 0, got {pattern_t}")
        if any(e <= 0 for e in remainder_t):
            raise ValueError(f"All remainder edge lengths must be > 0, got {remainder_t}")

        pattern_cum = tuple(itertools.accumulate(pattern_t))
        period_extent = pattern_cum[-1]
        remainder_cum = tuple(itertools.accumulate(remainder_t)) if remainder_t else ()
        total_nchunks = len(pattern_t) * repeats + len(remainder_t)
        total_extent = period_extent * repeats + (remainder_cum[-1] if remainder_cum else 0)

        object.__setattr__(self, "pattern", pattern_t)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "remainder", remainder_t)
        object.__setattr__(self, "_pattern_cumulative", pattern_cum)
        object.__setattr__(self, "_period_extent", period_extent)
        object.__setattr__(self, "_pattern_nchunks", len(pattern_t))
        object.__setattr__(self, "_remainder_cumulative", remainder_cum)
        object.__setattr__(self, "_total_nchunks", total_nchunks)
        object.__setattr__(self, "_total_extent", total_extent)

    @property
    def nchunks(self) -> int:
        return self._total_nchunks

    @property
    def extent(self) -> int:
        return self._total_extent

    def chunk_offset(self, chunk_ix: int) -> int:
        period, offset = divmod(chunk_ix, self._pattern_nchunks)
        if period < self.repeats:
            base = period * self._period_extent
            return base + (self._pattern_cumulative[offset - 1] if offset > 0 else 0)
        # In the remainder
        rem_ix = chunk_ix - self.repeats * self._pattern_nchunks
        return self.repeats * self._period_extent + (
            self._remainder_cumulative[rem_ix - 1] if rem_ix > 0 else 0
        )

    def chunk_size(self, chunk_ix: int) -> int:
        """Buffer size for codec processing."""
        period, offset = divmod(chunk_ix, self._pattern_nchunks)
        if period < self.repeats:
            return self.pattern[offset]
        return self.remainder[chunk_ix - self.repeats * self._pattern_nchunks]

    def data_size(self, chunk_ix: int) -> int:
        """Valid data region — same as chunk_size for tiled dims."""
        return self.chunk_size(chunk_ix)

    def index_to_chunk(self, idx: int) -> int:
        period, within = divmod(idx, self._period_extent)
        if period < self.repeats:
            local = bisect.bisect_right(self._pattern_cumulative, within)
            return period * self._pattern_nchunks + local
        # In the remainder region
        rem_idx = idx - self.repeats * self._period_extent
        local = bisect.bisect_right(self._remainder_cumulative, rem_idx)
        return self.repeats * self._pattern_nchunks + local

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        periods, withins = np.divmod(indices, self._period_extent)
        result = np.empty_like(indices)

        # Chunks in the repeating region
        in_repeat = periods < self.repeats
        if np.any(in_repeat):
            local = np.searchsorted(self._pattern_cumulative, withins[in_repeat], side="right")
            result[in_repeat] = periods[in_repeat] * self._pattern_nchunks + local

        # Chunks in the remainder region
        in_remainder = ~in_repeat
        if np.any(in_remainder) and self._remainder_cumulative:
            rem_indices = indices[in_remainder] - self.repeats * self._period_extent
            local = np.searchsorted(self._remainder_cumulative, rem_indices, side="right")
            result[in_remainder] = self.repeats * self._pattern_nchunks + local

        return result

    @property
    def edges(self) -> tuple[int, ...]:
        """Expand to full edge list (for compatibility with VaryingDimension)."""
        return self.pattern * self.repeats + self.remainder


@runtime_checkable
class DimensionGrid(Protocol):
    """Structural interface shared by FixedDimension, VaryingDimension, and TiledDimension."""

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


def _expand_rle(data: list[list[int]]) -> list[int]:
    """Expand run-length encoded chunk sizes: [[size, count], ...] -> [size, size, ...]"""
    result: list[int] = []
    for item in data:
        if len(item) != 2:
            raise ValueError(f"RLE entries must be [size, count], got {item}")
        size, count = item
        result.extend([size] * count)
    return result


def _compress_rle(sizes: Sequence[int]) -> list[list[int]]:
    """Compress chunk sizes to RLE: [10,10,10,20,20] -> [[10,3],[20,2]]"""
    if not sizes:
        return []
    result: list[list[int]] = []
    current = sizes[0]
    count = 1
    for s in sizes[1:]:
        if s == current:
            count += 1
        else:
            result.append([current, count])
            current = s
            count = 1
    result.append([current, count])
    return result


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------


def _detect_tile_pattern(
    edges: Sequence[int],
) -> tuple[tuple[int, ...], int, tuple[int, ...]] | None:
    """Detect the shortest repeating tile pattern in an edge list.

    Returns (pattern, repeats, remainder) if a tile pattern saves space over
    the flat representation, otherwise None.

    A pattern must repeat at least 2 times to qualify.
    """
    n = len(edges)
    if n < 4:
        return None

    # Try pattern lengths from 2 up to n//2
    for plen in range(2, n // 2 + 1):
        pattern = tuple(edges[:plen])
        full_repeats = n // plen
        if full_repeats < 2:
            break
        # Check all full repetitions match
        match = True
        for r in range(1, full_repeats):
            start = r * plen
            if tuple(edges[start : start + plen]) != pattern:
                match = False
                break
        if not match:
            continue
        remainder = tuple(edges[full_repeats * plen :])
        # Only use tile if it's more compact: pattern + remainder < flat list
        tile_cost = plen + len(remainder) + 2  # +2 for repeats field + overhead
        if tile_cost < n:
            return pattern, full_repeats, remainder
    return None


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

    Internally represents each dimension as FixedDimension (uniform chunks),
    VaryingDimension (per-chunk edge lengths with prefix sums), or
    TiledDimension (periodic pattern repeated N times).
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

    def __getitem__(self, coords: tuple[int, ...]) -> ChunkSpec | None:
        """Return the ChunkSpec for a chunk at the given grid position, or None if OOB."""
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

    # -- Serialization --

    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any]) -> ChunkGrid:
        if isinstance(data, ChunkGrid):
            if isinstance(data, RegularChunkGrid):
                return ChunkGrid.from_regular(
                    tuple(d.extent for d in data.dimensions),
                    data.chunk_shape,
                )
            return data

        name_parsed, configuration_parsed = parse_named_configuration(data)

        if name_parsed == "regular":
            chunk_shape_raw = configuration_parsed.get("chunk_shape")
            if chunk_shape_raw is None:
                raise ValueError("Regular chunk grid requires 'chunk_shape' configuration")
            if not isinstance(chunk_shape_raw, Sequence):
                raise TypeError(f"chunk_shape must be a sequence, got {type(chunk_shape_raw)}")
            # Without array shape, return a RegularChunkGrid that preserves
            # chunk_shape but raises on extent-dependent operations.
            # Use parse_chunk_grid() when array shape is available.
            return RegularChunkGrid(chunk_shape=tuple(int(cast("int", s)) for s in chunk_shape_raw))

        if name_parsed == "rectilinear":
            chunk_shapes_raw = configuration_parsed.get("chunk_shapes")
            if chunk_shapes_raw is None:
                raise ValueError("Rectilinear chunk grid requires 'chunk_shapes' configuration")
            if not isinstance(chunk_shapes_raw, Sequence):
                raise TypeError(f"chunk_shapes must be a sequence, got {type(chunk_shapes_raw)}")
            dims_list: list[DimensionGrid] = []
            for dim_spec in chunk_shapes_raw:
                parsed = _parse_dim_spec(dim_spec)
                dims_list.append(_build_dimension(parsed))
            return cls(dimensions=tuple(dims_list))

        raise ValueError(f"Unknown chunk grid name: {name_parsed!r}")

    # ChunkGrid does not serialize itself. The format choice ("regular" vs
    # "rectilinear") belongs to the metadata layer. Use serialize_chunk_grid().


def _parse_dim_spec(dim_spec: Any) -> list[int] | TiledDimension:
    """Parse a single dimension's chunk_shapes entry.

    Returns either a flat edge list or a TiledDimension (for tile-encoded entries).
    Handles: flat list of ints, RLE ([[size, count], ...]), and tile dicts.
    """
    if isinstance(dim_spec, dict):
        # Tile encoding: {"tile": [...], "repeat": N, "remainder": [...]}
        tile_pattern = dim_spec.get("tile")
        if tile_pattern is None:
            raise ValueError(f"Tile-encoded dim_spec must have 'tile' key, got {dim_spec}")
        repeat = dim_spec.get("repeat", 1)
        remainder = dim_spec.get("remainder", [])
        return TiledDimension(
            pattern=tile_pattern,
            repeats=repeat,
            remainder=remainder,
        )
    if isinstance(dim_spec, list) and dim_spec and isinstance(dim_spec[0], list):
        return _expand_rle(dim_spec)
    if isinstance(dim_spec, list):
        return dim_spec
    raise ValueError(f"Invalid chunk_shapes entry: {dim_spec}")


def _build_dimension(dim_spec_parsed: list[int] | TiledDimension) -> DimensionGrid:
    """Build a DimensionGrid from a parsed dim spec."""
    if isinstance(dim_spec_parsed, TiledDimension):
        return dim_spec_parsed
    edges = dim_spec_parsed
    if all(e == edges[0] for e in edges):
        return FixedDimension(size=edges[0], extent=sum(edges))
    return VaryingDimension(edges)


def parse_chunk_grid(
    data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any],
    array_shape: tuple[int, ...],
) -> ChunkGrid:
    """Create a ChunkGrid from a metadata dict, injecting array shape as extent.

    This is the primary entry point for constructing a ChunkGrid from serialized
    metadata. Unlike ``ChunkGrid.from_dict``, this always produces a grid with
    correct extent values.
    """
    if isinstance(data, ChunkGrid):
        # Re-bind extent if array_shape differs from what's stored
        dims: list[DimensionGrid] = []
        for dim, extent in zip(data.dimensions, array_shape, strict=True):
            if isinstance(dim, FixedDimension):
                dims.append(FixedDimension(size=dim.size, extent=extent))
            else:
                # VaryingDimension/TiledDimension have intrinsic extent — validate
                if dim.extent != extent:
                    dim_type = type(dim).__name__
                    raise ValueError(
                        f"{dim_type} extent {dim.extent} does not match "
                        f"array shape extent {extent} for dimension {len(dims)}"
                    )
                dims.append(dim)
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
        dims_built: list[DimensionGrid] = []
        for i, dim_spec in enumerate(chunk_shapes_raw):
            parsed = _parse_dim_spec(dim_spec)
            dim = _build_dimension(parsed)
            if dim.extent != array_shape[i]:
                raise ValueError(
                    f"Rectilinear chunk edges for dimension {i} sum to {dim.extent} "
                    f"but array shape extent is {array_shape[i]}"
                )
            dims_built.append(dim)
        return ChunkGrid(dimensions=tuple(dims_built))

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
                # Produce RLE directly without allocating a full edge list.
                n = dim.nchunks
                if n == 0:
                    chunk_shapes.append([])
                else:
                    last_data = dim.extent - (n - 1) * dim.size
                    if last_data == dim.size:
                        chunk_shapes.append([[dim.size, n]])
                    else:
                        rle: list[list[int]] = []
                        if n > 1:
                            rle.append([dim.size, n - 1])
                        rle.append([last_data, 1])
                        chunk_shapes.append(rle)
            elif isinstance(dim, TiledDimension):
                tile_dict: dict[str, Any] = {
                    "tile": list(dim.pattern),
                    "repeat": dim.repeats,
                }
                if dim.remainder:
                    tile_dict["remainder"] = list(dim.remainder)
                chunk_shapes.append(tile_dict)
            elif isinstance(dim, VaryingDimension):
                edges = list(dim.edges)
                # Try tile compression first (more compact for periodic patterns)
                tile_result = _detect_tile_pattern(edges)
                if tile_result is not None:
                    pattern, repeats, remainder = tile_result
                    tile_dict_v: dict[str, Any] = {
                        "tile": list(pattern),
                        "repeat": repeats,
                    }
                    if remainder:
                        tile_dict_v["remainder"] = list(remainder)
                    chunk_shapes.append(tile_dict_v)
                else:
                    rle = _compress_rle(edges)
                    if sum(count for _, count in rle) == len(edges) and len(rle) < len(edges):
                        chunk_shapes.append(rle)
                    else:
                        chunk_shapes.append(edges)
        return {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": chunk_shapes},
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
# Backwards-compatible alias
# ---------------------------------------------------------------------------


class RegularChunkGrid(ChunkGrid):
    """Backwards-compatible wrapper. Prefer ChunkGrid.from_regular() for new code."""

    _chunk_shape: tuple[int, ...]

    def __init__(self, *, chunk_shape: ShapeLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        # Without array shape, use extent=0 as placeholder
        dims = tuple(FixedDimension(size=s, extent=0) for s in chunk_shape_parsed)
        super().__init__(dimensions=dims)
        object.__setattr__(self, "_chunk_shape", chunk_shape_parsed)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the stored chunk shape (extent may be 0 as placeholder)."""
        return self._chunk_shape

    @classmethod
    def _from_dict(cls, data: dict[str, JSON] | NamedConfig[str, Any]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def _raise_no_extent(self) -> None:
        raise ValueError(
            "RegularChunkGrid does not have array shape information. "
            "Use ChunkGrid.from_regular(array_shape, chunk_shape) or "
            "parse_chunk_grid() to create a grid with extent."
        )

    @property
    def shape(self) -> tuple[int, ...]:
        self._raise_no_extent()
        raise AssertionError  # unreachable, for mypy

    def all_chunk_coords(self) -> Iterator[tuple[int, ...]]:
        self._raise_no_extent()
        raise AssertionError  # unreachable, for mypy

    def get_nchunks(self) -> int:
        self._raise_no_extent()
        raise AssertionError  # unreachable, for mypy


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
            _shards_out = shard_shape

    return _shards_out, _chunks_out
