"""Compact chunk grids and the narrow planner protocol.

``DimensionGridLike`` describes only the per-axis operations required by
``plan_chunks``.  The concrete compact grids below also retain enough metadata
to describe chunk data regions and codec buffer regions without importing
Zarr's array implementation.
"""

from __future__ import annotations

import bisect
import itertools
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import numpy.typing as npt


class DimensionGridLike(Protocol):
    """The per-dimension chunk-mapping surface consumed by chunk resolution."""

    def index_to_chunk(self, idx: int) -> int:
        """Map a global source index to the index of the chunk that contains it.

        Implementers must raise `IndexError` when `idx` lies outside `[0, extent)`.
        """
        ...

    def chunk_offset(self, chunk_ix: int) -> int:
        """The global source coordinate at which chunk `chunk_ix` begins."""
        ...

    def chunk_size(self, chunk_ix: int) -> int:
        """The declared length of chunk `chunk_ix`, i.e. its codec buffer size along this axis."""
        ...

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_to_chunk`: map global source indices to chunk indices.

        Implementers must raise `IndexError` if any index lies outside `[0, extent)`.
        """
        ...


def _bounded_indices(indices: npt.NDArray[np.intp], extent: int) -> npt.NDArray[np.intp]:
    """Normalize a vector lookup and enforce the scalar grid bounds."""
    arr = np.asarray(indices, dtype=np.intp)
    if arr.size > 0 and (int(arr.min()) < 0 or int(arr.max()) >= extent):
        raise IndexError(
            f"indices must lie in [0, {extent}); got [{int(arr.min())}, {int(arr.max())}]"
        )
    return arr


@dataclass(frozen=True)
class FixedDimension:
    """Uniform chunk size with a boundary chunk clipped to the axis extent."""

    size: int
    extent: int
    nchunks: int = field(init=False, repr=False)
    ngridcells: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"FixedDimension size must be >= 0, got {self.size}")
        if self.extent < 0:
            raise ValueError(f"FixedDimension extent must be >= 0, got {self.extent}")
        if self.size == 0 and self.extent > 0:
            raise ValueError(
                "FixedDimension size must be > 0 when extent is nonzero; "
                f"got size {self.size} and extent {self.extent}"
            )
        nchunks = 0 if self.size == 0 else (self.extent + self.size - 1) // self.size
        object.__setattr__(self, "nchunks", nchunks)
        object.__setattr__(self, "ngridcells", nchunks)

    def index_to_chunk(self, idx: int) -> int:
        """Map a global source index to its chunk index (`idx // size`).

        Raises `IndexError` when `idx` lies outside `[0, extent)`.
        """
        if idx < 0 or idx >= self.extent:
            raise IndexError(f"index {idx} is out of bounds for extent {self.extent}")
        return 0 if self.size == 0 else idx // self.size

    def chunk_offset(self, chunk_ix: int) -> int:
        """The global source coordinate where chunk `chunk_ix` begins (`chunk_ix * size`).

        Not bounds-checked: chunk indices past the last chunk extrapolate linearly.
        """
        return chunk_ix * self.size

    def chunk_size(self, chunk_ix: int) -> int:
        """The declared chunk length, `size` for every chunk.

        The boundary chunk is not clipped here; use `data_size` for the valid data length.
        """
        return self.size

    def data_size(self, chunk_ix: int) -> int:
        """The number of valid data elements in chunk `chunk_ix`, clipped to `extent`.

        Interior chunks report `size`; the boundary chunk reports the remainder, and chunk
        indices at or past `nchunks` report 0.
        """
        if self.size == 0:
            return 0
        return max(0, min(self.size, self.extent - chunk_ix * self.size))

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_to_chunk` over an array of global source indices.

        Raises `IndexError` if any index lies outside `[0, extent)`.
        """
        arr = _bounded_indices(indices, self.extent)
        if self.size == 0:
            return np.zeros_like(arr)
        return arr // self.size

    def with_extent(self, new_extent: int) -> FixedDimension:
        """Return a copy with the same chunk size and the axis extent set to `new_extent`."""
        return FixedDimension(size=self.size, extent=new_extent)

    def resize(self, new_extent: int) -> FixedDimension:
        """Return a copy resized to `new_extent`; the fixed chunk size covers any new extent."""
        return FixedDimension(size=self.size, extent=new_extent)

    @property
    def size_repr(self) -> str:
        """The chunk size rendered as a scalar for `ChunkGrid.__repr__`."""
        return str(self.size)


@dataclass(frozen=True, init=False)
class VaryingDimension:
    """Explicit chunk edge lengths, with trailing data clipped to ``extent``."""

    edges: tuple[int, ...]
    cumulative: tuple[int, ...]
    extent: int
    nchunks: int = field(init=False, repr=False)
    ngridcells: int = field(init=False, repr=False)

    def __init__(self, edges: Sequence[int], extent: int) -> None:
        edges_tuple = tuple(edges)
        if not edges_tuple:
            raise ValueError("VaryingDimension edges must not be empty")
        if any(edge <= 0 for edge in edges_tuple):
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
        nchunks = 0 if extent == 0 else bisect.bisect_left(cumulative, extent) + 1
        object.__setattr__(self, "nchunks", nchunks)
        object.__setattr__(self, "ngridcells", len(edges_tuple))

    def index_to_chunk(self, idx: int) -> int:
        """Map a global source index to the chunk whose edge interval contains it.

        Raises `IndexError` when `idx` lies outside `[0, extent)`.
        """
        if idx < 0 or idx >= self.extent:
            raise IndexError(f"index {idx} is out of bounds for extent {self.extent}")
        return bisect.bisect_right(self.cumulative, idx)

    def chunk_offset(self, chunk_ix: int) -> int:
        """The global source coordinate where chunk `chunk_ix` begins (sum of prior edges)."""
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0

    def chunk_size(self, chunk_ix: int) -> int:
        """The declared edge length of chunk `chunk_ix`.

        Trailing chunks are not clipped to `extent` here; use `data_size` for that.
        """
        return self.edges[chunk_ix]

    def data_size(self, chunk_ix: int) -> int:
        """The number of valid data elements in chunk `chunk_ix`, clipped to `extent`.

        Grid cells that lie entirely at or past `extent` report 0.
        """
        offset = self.chunk_offset(chunk_ix)
        return max(0, min(self.edges[chunk_ix], self.extent - offset))

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_to_chunk` over an array of global source indices.

        Raises `IndexError` if any index lies outside `[0, extent)`.
        """
        arr = _bounded_indices(indices, self.extent)
        return np.searchsorted(self.cumulative, arr, side="right")

    def with_extent(self, new_extent: int) -> VaryingDimension:
        """Return a copy with the same edges re-clipped to `new_extent`.

        The existing edges must already cover the new extent; raises `ValueError` when
        `new_extent` exceeds the sum of edges. Use `resize` to grow past the edges.
        """
        if self.cumulative[-1] < new_extent:
            raise ValueError(
                f"VaryingDimension edge sum {self.cumulative[-1]} is less than new extent "
                f"{new_extent}"
            )
        return VaryingDimension(self.edges, extent=new_extent)

    def resize(self, new_extent: int) -> VaryingDimension:
        """Return a copy resized to `new_extent`.

        Shrinking (or growing within the existing edges) keeps the edges and re-clips them;
        growing past the sum of edges appends one new trailing edge covering the remainder.
        """
        if new_extent == self.extent:
            return self
        if new_extent > self.cumulative[-1]:
            return VaryingDimension((*self.edges, new_extent - self.cumulative[-1]), new_extent)
        return VaryingDimension(self.edges, extent=new_extent)

    @property
    def size_repr(self) -> str:
        """The edge lengths rendered as a tuple for `ChunkGrid.__repr__`."""
        return repr(self.edges)


@runtime_checkable
class DimensionGrid(Protocol):
    """Structural interface shared by the compact dimension grids."""

    @property
    def nchunks(self) -> int:
        """The number of chunks holding data within `extent`."""
        ...

    @property
    def ngridcells(self) -> int:
        """The number of declared grid cells; may exceed `nchunks` when trailing cells are empty."""
        ...

    @property
    def extent(self) -> int:
        """The axis length in global source coordinates."""
        ...

    def index_to_chunk(self, idx: int) -> int:
        """Map a global source index to the chunk index that contains it.

        Implementers must raise `IndexError` when `idx` lies outside `[0, extent)`.
        """
        ...

    def chunk_offset(self, chunk_ix: int) -> int:
        """The global source coordinate at which chunk `chunk_ix` begins."""
        ...

    def chunk_size(self, chunk_ix: int) -> int:
        """The declared (codec buffer) length of chunk `chunk_ix`, never clipped to `extent`."""
        ...

    def data_size(self, chunk_ix: int) -> int:
        """The valid data length of chunk `chunk_ix`, clipped to `extent` at the boundary."""
        ...

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_to_chunk`; must raise `IndexError` for indices outside `[0, extent)`."""
        ...

    def with_extent(self, new_extent: int) -> DimensionGrid:
        """Return a grid with the existing chunk layout re-clipped to `new_extent`.

        Implementers must not invent new grid cells: raise `ValueError` when the declared
        layout cannot cover `new_extent`.
        """
        ...

    def resize(self, new_extent: int) -> DimensionGrid:
        """Return a grid covering `new_extent`, extending the chunk layout when it must grow."""
        ...

    @property
    def size_repr(self) -> str:
        """A compact rendering of the chunk sizes, used by `ChunkGrid.__repr__`."""
        ...


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk's valid data region and its full codec buffer shape."""

    slices: tuple[slice, ...]
    codec_shape: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the valid data region described by `slices`.

        Smaller than `codec_shape` on boundary chunks, where the array extent clips the chunk.
        """
        return tuple(chunk_slice.stop - chunk_slice.start for chunk_slice in self.slices)

    @property
    def is_boundary(self) -> bool:
        """Whether the valid data region is smaller than the full codec buffer on any axis."""
        return self.shape != self.codec_shape


@dataclass(frozen=True)
class ChunkGrid:
    """A concrete regular or rectilinear arrangement of chunks for one array."""

    dimensions: tuple[DimensionGrid, ...]
    _is_regular: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_is_regular",
            all(isinstance(dimension, FixedDimension) for dimension in self.dimensions),
        )

    def __repr__(self) -> str:
        sizes = ", ".join(dimension.size_repr for dimension in self.dimensions)
        shape = tuple(dimension.extent for dimension in self.dimensions)
        return f"ChunkGrid(chunk_sizes=({sizes}), array_shape={shape})"

    @classmethod
    def from_sizes(
        cls, array_shape: Sequence[int], chunk_sizes: Sequence[int | Sequence[int]]
    ) -> ChunkGrid:
        """Build a grid from an array shape and one chunk-size spec per dimension.

        An `int` entry gives a fixed chunk size along that axis; a sequence of ints gives
        explicit per-chunk edge lengths. A uniform sequence consistent with the axis extent
        collapses to a fixed dimension, so the result may report `is_regular`.

        Parameters
        ----------
        array_shape : Sequence[int]
            The array extent along each dimension, in global source coordinates.
        chunk_sizes : Sequence[int | Sequence[int]]
            Per-dimension chunk layout: a single size or explicit edge lengths.
        """
        extents = _shape_tuple(array_shape)
        if len(extents) != len(chunk_sizes):
            raise ValueError(
                f"array_shape has {len(extents)} dimensions but chunk_sizes has "
                f"{len(chunk_sizes)} dimensions"
            )
        dimensions: list[DimensionGrid] = []
        for dimension_spec, extent in zip(chunk_sizes, extents, strict=True):
            if isinstance(dimension_spec, int):
                dimensions.append(FixedDimension(size=dimension_spec, extent=extent))
            else:
                edges = tuple(dimension_spec)
                if not edges:
                    raise ValueError("Each dimension must have at least one chunk")
                if (
                    edges[0] > 0
                    and all(edge == edges[0] for edge in edges)
                    and (extent == sum(edges) or len(edges) == (extent + edges[0] - 1) // edges[0])
                ):
                    dimensions.append(FixedDimension(size=edges[0], extent=extent))
                else:
                    dimensions.append(VaryingDimension(edges, extent=extent))
        return cls(dimensions=tuple(dimensions))

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.dimensions)

    @property
    def is_regular(self) -> bool:
        """Whether every dimension uses a single fixed chunk size.

        False when any axis carries explicit (rectilinear) per-chunk edge lengths.
        """
        return self._is_regular

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The number of data-bearing chunks along each dimension."""
        return tuple(dimension.nchunks for dimension in self.dimensions)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """The uniform declared chunk shape of a regular grid.

        Raises `ValueError` for rectilinear grids, which have no single chunk shape;
        use `grid[coords]` for per-chunk sizes instead.
        """
        if not self.is_regular:
            raise ValueError(
                "chunk_shape is only available for regular chunk grids. "
                "Use grid[coords] for per-chunk sizes."
            )
        return tuple(
            dimension.size for dimension in self.dimensions if isinstance(dimension, FixedDimension)
        )

    @property
    def chunk_sizes(self) -> tuple[tuple[int, ...], ...]:
        """Per-dimension tuples of each chunk's valid data length.

        Boundary chunks report their clipped extent, not the declared codec size.
        """
        return tuple(
            tuple(dimension.data_size(index) for index in range(dimension.nchunks))
            for dimension in self.dimensions
        )

    def __getitem__(self, coords: int | tuple[int, ...]) -> ChunkSpec | None:
        """Look up the `ChunkSpec` at the given chunk coordinates (grid cells, not indices).

        Returns `None` when any coordinate falls outside the grid; raises `ValueError`
        when the number of coordinates does not match `ndim`. The spec's slices are in
        global source coordinates.
        """
        if isinstance(coords, int):
            coords = (coords,)
        if len(coords) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} coordinate(s) for a {self.ndim}-d chunk grid, "
                f"got {len(coords)}."
            )
        slices: list[slice] = []
        codec_shape: list[int] = []
        for dimension, index in zip(self.dimensions, coords, strict=True):
            if index < 0 or index >= dimension.nchunks:
                return None
            offset = dimension.chunk_offset(index)
            slices.append(slice(offset, offset + dimension.data_size(index), 1))
            codec_shape.append(dimension.chunk_size(index))
        return ChunkSpec(tuple(slices), tuple(codec_shape))

    def __iter__(self) -> Iterator[ChunkSpec]:
        """Yield a `ChunkSpec` for every data-bearing chunk in row-major (C) order."""
        for coords in itertools.product(
            *(range(dimension.nchunks) for dimension in self.dimensions)
        ):
            spec = self[coords]
            if spec is not None:
                yield spec

    def all_chunk_coords(
        self,
        *,
        origin: Sequence[int] | None = None,
        selection_shape: Sequence[int] | None = None,
    ) -> Iterator[tuple[int, ...]]:
        """Iterate chunk coordinates over a rectangular grid region in row-major (C) order.

        `origin` defaults to the grid origin and `selection_shape` to the rest of the grid.
        The region is not bounds-checked: an oversized region yields coordinates outside
        the grid, which `__getitem__` resolves to `None`.
        """
        origin_parsed = (0,) * self.ndim if origin is None else tuple(origin)
        selection_shape_parsed = (
            tuple(
                grid_size - coordinate
                for coordinate, grid_size in zip(origin_parsed, self.grid_shape, strict=True)
            )
            if selection_shape is None
            else tuple(selection_shape)
        )
        return itertools.product(
            *(
                range(coordinate, coordinate + size)
                for coordinate, size in zip(origin_parsed, selection_shape_parsed, strict=True)
            )
        )

    def iter_chunk_regions(
        self,
        *,
        origin: Sequence[int] | None = None,
        selection_shape: Sequence[int] | None = None,
    ) -> Iterator[tuple[slice, ...]]:
        """Yield each chunk's valid-data slices, in global source coordinates.

        Covers the same region as `all_chunk_coords`, silently skipping coordinates
        that fall outside the grid.
        """
        for coords in self.all_chunk_coords(origin=origin, selection_shape=selection_shape):
            spec = self[coords]
            if spec is not None:
                yield spec.slices

    def get_nchunks(self) -> int:
        """The total number of data-bearing chunks: the product of `grid_shape` (1 if 0-d)."""
        return reduce(operator.mul, (dimension.nchunks for dimension in self.dimensions), 1)

    def update_shape(self, new_shape: tuple[int, ...]) -> ChunkGrid:
        """Return a grid resized to `new_shape` by resizing each dimension.

        Fixed axes keep their chunk size; rectilinear axes gain one trailing edge when
        grown past their declared edges. Raises `ValueError` when `new_shape` does not
        have `ndim` entries.
        """
        if len(new_shape) != self.ndim:
            raise ValueError(
                f"new_shape has {len(new_shape)} dimensions but chunk grid has {self.ndim} dimensions"
            )
        return ChunkGrid(
            dimensions=tuple(
                dimension.resize(new_extent)
                for dimension, new_extent in zip(self.dimensions, new_shape, strict=True)
            )
        )


class EdgeDimensionGrid:
    """An explicitly edge-based grid for coordinate-origin examples and planners."""

    __slots__ = ("_offsets", "sizes")

    sizes: tuple[int, ...]

    def __init__(self, sizes: Sequence[int]) -> None:
        """Build a one-axis grid from explicit per-chunk sizes.

        Every size must be positive; raises `ValueError` otherwise. A zero-length axis
        is spelled as an empty sequence (no chunks), not as a zero size.

        Parameters
        ----------
        sizes : Sequence[int]
            The length of each chunk along the axis, in order.
        """
        normalized = tuple(int(size) for size in sizes)
        for index, size in enumerate(normalized):
            if size <= 0:
                raise ValueError(
                    f"chunk sizes must be positive; got {size} at position {index} of {normalized}. "
                    "A zero-length axis is spelled as no chunks at all: EdgeDimensionGrid(())"
                )
        self.sizes = normalized
        offsets: np.ndarray[Any, np.dtype[np.intp]] = np.zeros(len(normalized) + 1, dtype=np.intp)
        if normalized:
            np.cumsum(np.asarray(normalized, dtype=np.intp), out=offsets[1:])
        self._offsets = offsets

    @property
    def num_chunks(self) -> int:
        """The number of chunks along the axis."""
        return len(self.sizes)

    @property
    def extent(self) -> int:
        """The axis length in global source coordinates: the sum of all chunk sizes."""
        return int(self._offsets[-1])

    def __repr__(self) -> str:
        return f"EdgeDimensionGrid(sizes={self.sizes})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeDimensionGrid):
            return NotImplemented
        return self.sizes == other.sizes

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.sizes))

    def index_to_chunk(self, idx: int) -> int:
        """Map a global source index to the chunk whose interval contains it.

        Raises `IndexError` when `idx` lies outside `[0, extent)`.
        """
        if idx < 0 or idx >= self.extent:
            raise IndexError(f"index {idx} is out of bounds for an axis of extent {self.extent}")
        return int(np.searchsorted(self._offsets, idx, side="right")) - 1

    def chunk_offset(self, chunk_ix: int) -> int:
        """The global source coordinate where chunk `chunk_ix` begins.

        Raises `IndexError` when `chunk_ix` lies outside `[0, num_chunks)`.
        """
        if chunk_ix < 0 or chunk_ix >= len(self.sizes):
            raise IndexError(
                f"chunk index {chunk_ix} is out of bounds for {len(self.sizes)} chunks"
            )
        return int(self._offsets[chunk_ix])

    def chunk_size(self, chunk_ix: int) -> int:
        """The length of chunk `chunk_ix`; every chunk holds data, so no boundary clipping applies.

        Raises `IndexError` when `chunk_ix` lies outside `[0, num_chunks)`.
        """
        if chunk_ix < 0 or chunk_ix >= len(self.sizes):
            raise IndexError(
                f"chunk index {chunk_ix} is out of bounds for {len(self.sizes)} chunks"
            )
        return self.sizes[chunk_ix]

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_to_chunk` over an array of global source indices.

        Raises `IndexError` if any index lies outside `[0, extent)`.
        """
        arr = _bounded_indices(indices, self.extent)
        return (np.searchsorted(self._offsets, arr, side="right") - 1).astype(np.intp)


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(size) for size in shape)
    if any(size < 0 for size in result):
        raise ValueError(f"shape entries must be non-negative; got {result}")
    return result


def _entry_kind(entry: Any) -> str:
    if isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
        return "int"
    if isinstance(entry, (str, bytes)):
        return "neither"
    try:
        iter(cast("Iterable[Any]", entry))
    except TypeError:
        return "neither"
    return "sequence"


def dimension_grids_from_chunks(
    chunks: Sequence[int] | Sequence[Sequence[int]], shape: Sequence[int]
) -> tuple[DimensionGrid, ...]:
    """Build compact dimensions from regular sizes or explicit per-axis edges."""
    shape_t = _shape_tuple(shape)
    entries: tuple[Any, ...] = tuple(chunks)
    if len(entries) != len(shape_t):
        raise ValueError(
            f"chunks must have one entry per dimension; got {len(entries)} entries for shape {shape_t}"
        )

    conventions = (
        "chunks must be either a uniform chunk shape (one integer per dimension) "
        "or per-axis chunk sizes (one sequence of integers per dimension)"
    )
    kinds = [_entry_kind(entry) for entry in entries]
    neither = [(axis, entries[axis]) for axis, kind in enumerate(kinds) if kind == "neither"]
    if neither:
        described = ", ".join(f"{entry!r} at dimension {axis}" for axis, entry in neither)
        verb = "is" if len(neither) == 1 else "are"
        raise ValueError(f"{conventions}; {described} {verb} neither")

    integer_count = sum(kind == "int" for kind in kinds)
    if entries and integer_count == len(entries):
        dimensions: list[DimensionGrid] = []
        for entry, extent in zip(entries, shape_t, strict=True):
            size = int(entry)
            if size <= 0:
                raise ValueError(f"chunk shape entries must be positive; got {size}")
            dimensions.append(FixedDimension(size=size, extent=extent))
        return tuple(dimensions)
    if integer_count:
        raise ValueError(f"{conventions}, not a mixture; got {entries!r}")

    dimensions = []
    for axis, (entry, extent) in enumerate(zip(entries, shape_t, strict=True)):
        elements: tuple[Any, ...] = tuple(cast("Iterable[Any]", entry))
        if any(
            not isinstance(element, (int, np.integer)) or isinstance(element, bool)
            for element in elements
        ):
            raise ValueError(
                f"per-axis chunk sizes must be integers; dimension {axis} has {entry!r}"
            )
        edges = tuple(int(element) for element in elements)
        total = sum(edges)
        if total != extent:
            raise ValueError(
                f"per-axis chunk sizes for dimension {axis} sum to {total}, but the array extent is {extent}"
            )
        if extent == 0 and all(edge == 0 for edge in edges):
            dimensions.append(FixedDimension(size=0, extent=0))
            continue
        if any(edge <= 0 for edge in edges):
            raise ValueError(f"chunk sizes must be positive; got {edges}")
        dimensions.append(VaryingDimension(edges=edges, extent=extent))
    return tuple(dimensions)
