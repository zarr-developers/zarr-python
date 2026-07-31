"""Per-dimension chunk grids: the Protocol and a concrete implementation.

`chunk_resolution` needs only a narrow slice of a chunk grid: the per-dimension
mapping between storage indices and chunk coordinates, passed as one
`DimensionGridLike` per storage dimension. Rather than import zarr's concrete
grid types, we type against this Protocol; zarr's per-dimension grids satisfy
it structurally, so no zarr import is needed here.

`EdgeDimensionGrid` is a concrete implementation for callers that do not have a
zarr grid to hand — it is built from an explicit tuple of per-chunk sizes along
one axis, so a clipped ("edge") boundary chunk is expressed directly rather than
derived from a uniform chunk shape. `dimension_grids_from_chunks` builds one per
axis from either chunk convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt


class DimensionGridLike(Protocol):
    """The per-dimension chunk-mapping surface consumed by chunk resolution."""

    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...
    def chunk_size(self, chunk_ix: int) -> int: ...
    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]: ...


class EdgeDimensionGrid:
    """A one-dimensional chunk grid described by an explicit tuple of chunk sizes.

    The sizes follow the dask `Array.chunks` convention for a single axis: one
    entry per chunk giving that chunk's extent, with a possibly-smaller final
    ("edge") chunk. Chunk offsets are the exclusive prefix sums, so lookups are
    a binary search rather than a division — the grid does not have to be
    regular.

    Parameters
    ----------
    sizes
        Per-chunk extents along this axis. Every entry must be positive. An
        empty sequence describes a zero-length axis.

    Examples
    --------
    >>> grid = EdgeDimensionGrid((3, 3, 1))
    >>> grid.index_to_chunk(4)
    1
    >>> grid.chunk_offset(2), grid.chunk_size(2)
    (6, 1)
    """

    __slots__ = ("_offsets", "sizes")

    sizes: tuple[int, ...]

    def __init__(self, sizes: Sequence[int]) -> None:
        normalized = tuple(int(s) for s in sizes)
        for i, s in enumerate(normalized):
            if s <= 0:
                raise ValueError(
                    f"chunk sizes must be positive; got {s} at position {i} of {normalized}"
                )
        self.sizes = normalized
        # Exclusive prefix sums, length len(sizes) + 1: `_offsets[c]` is the
        # first index of chunk `c`, and `_offsets[-1]` the total extent.
        offsets: np.ndarray[Any, np.dtype[np.intp]] = np.zeros(len(normalized) + 1, dtype=np.intp)
        if len(normalized) > 0:
            np.cumsum(np.asarray(normalized, dtype=np.intp), out=offsets[1:])
        self._offsets = offsets

    @property
    def num_chunks(self) -> int:
        """The number of chunks along this axis."""
        return len(self.sizes)

    @property
    def extent(self) -> int:
        """The total number of indices this axis spans."""
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
        """Return the index of the chunk holding index `idx`."""
        if idx < 0 or idx >= self.extent:
            raise IndexError(f"index {idx} is out of bounds for an axis of extent {self.extent}")
        return int(np.searchsorted(self._offsets, idx, side="right")) - 1

    def chunk_offset(self, chunk_ix: int) -> int:
        """Return the first index belonging to chunk `chunk_ix`."""
        if chunk_ix < 0 or chunk_ix >= len(self.sizes):
            raise IndexError(
                f"chunk index {chunk_ix} is out of bounds for {len(self.sizes)} chunks"
            )
        return int(self._offsets[chunk_ix])

    def chunk_size(self, chunk_ix: int) -> int:
        """Return the extent of chunk `chunk_ix`."""
        if chunk_ix < 0 or chunk_ix >= len(self.sizes):
            raise IndexError(
                f"chunk index {chunk_ix} is out of bounds for {len(self.sizes)} chunks"
            )
        return self.sizes[chunk_ix]

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Return the chunk holding each index in `indices`, elementwise."""
        arr = np.asarray(indices, dtype=np.intp)
        if arr.size > 0 and (int(arr.min()) < 0 or int(arr.max()) >= self.extent):
            raise IndexError(
                f"indices must lie in [0, {self.extent}); got [{int(arr.min())}, {int(arr.max())}]"
            )
        return (np.searchsorted(self._offsets, arr, side="right") - 1).astype(np.intp)


def _uniform_chunk_sizes(extent: int, chunk: int) -> tuple[int, ...]:
    """Expand a single uniform chunk length into per-chunk sizes, tail clipped."""
    if chunk <= 0:
        raise ValueError(f"chunk shape entries must be positive; got {chunk}")
    n_full, remainder = divmod(extent, chunk)
    sizes = (chunk,) * n_full
    if remainder > 0:
        sizes = (*sizes, remainder)
    return sizes


def _entry_kind(entry: Any) -> str:
    """Classify one `chunks` entry as `"int"`, `"sequence"`, or `"neither"`."""
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
    chunks: Sequence[int] | Sequence[Sequence[int]],
    shape: Sequence[int],
) -> tuple[EdgeDimensionGrid, ...]:
    """Build one `EdgeDimensionGrid` per axis from either chunk convention.

    Two conventions are accepted, disambiguated by element type:

    - a **uniform chunk shape** — one integer per axis, the chunk length along
      that axis. The trailing chunk is clipped to the array extent, matching how
      a zarr regular chunk grid covers a shape that is not a multiple of the
      chunk shape.
    - **dask-convention per-axis sizes** — one sequence of per-chunk extents per
      axis, e.g. `((3, 3, 1), (4,))`. Each sequence must sum to the
      corresponding entry of `shape`.

    Parameters
    ----------
    chunks
        A uniform chunk shape or dask-convention per-axis chunk sizes.
    shape
        The array shape the grids cover.

    Returns
    -------
    tuple of EdgeDimensionGrid
        One grid per axis, in axis order.

    Raises
    ------
    ValueError
        If `chunks` has a different length than `shape`, mixes the two
        conventions, or declares per-axis sizes that do not sum to `shape`.

    Examples
    --------
    >>> grids = dimension_grids_from_chunks((3, 4), (7, 4))
    >>> (grids[0].sizes, grids[1].sizes)
    ((3, 3, 1), (4,))
    """
    shape_t = tuple(int(s) for s in shape)
    entries: tuple[Any, ...] = tuple(chunks)
    if len(entries) != len(shape_t):
        raise ValueError(
            f"chunks must have one entry per dimension; got {len(entries)} "
            f"entries for shape {shape_t}"
        )

    _CONVENTIONS = (
        "chunks must be either a uniform chunk shape (one integer per "
        "dimension) or per-axis chunk sizes (one sequence of integers per "
        "dimension)"
    )
    # Classify every entry first, so the diagnosis names what is actually wrong.
    # An entry that belongs to *neither* convention (a float, a string, None) is
    # a different mistake from a well-formed mixture of the two, and saying
    # "not a mixture" about `(2.5, 2, 2)` sends the reader looking for the wrong
    # thing. `sum`/explicit loops rather than `all`/`any` over a comprehension:
    # the latter narrows `entry` to Never on the fall-through branch.
    kinds = [_entry_kind(entry) for entry in entries]
    neither = [(axis, entries[axis]) for axis, kind in enumerate(kinds) if kind == "neither"]
    if len(neither) > 0:
        described = ", ".join(f"{entry!r} at dimension {axis}" for axis, entry in neither)
        verb = "is" if len(neither) == 1 else "are"
        raise ValueError(f"{_CONVENTIONS}; {described} {verb} neither")

    n_int = sum(1 for kind in kinds if kind == "int")
    if len(entries) > 0 and n_int == len(entries):
        return tuple(
            EdgeDimensionGrid(_uniform_chunk_sizes(extent, int(entry)))
            for entry, extent in zip(entries, shape_t, strict=True)
        )
    if n_int > 0:
        raise ValueError(f"{_CONVENTIONS}, not a mixture; got {entries!r}")

    grids: list[EdgeDimensionGrid] = []
    for axis, (entry, extent) in enumerate(zip(entries, shape_t, strict=True)):
        # Every remaining entry is a sequence; its *elements* may still not be
        # integers, which is a third distinct mistake.
        # `int()` would silently truncate a float, so check the element type
        # rather than relying on the conversion to fail.
        elements: tuple[Any, ...] = tuple(cast("Iterable[Any]", entry))
        if any(not isinstance(e, (int, np.integer)) or isinstance(e, bool) for e in elements):
            raise ValueError(
                f"per-axis chunk sizes must be integers; dimension {axis} has {entry!r}"
            )
        sizes = tuple(int(e) for e in elements)
        total = sum(sizes)
        if total != extent:
            raise ValueError(
                f"per-axis chunk sizes for dimension {axis} sum to {total}, "
                f"but the array extent is {extent}"
            )
        grids.append(EdgeDimensionGrid(sizes))
    return tuple(grids)
