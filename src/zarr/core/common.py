from __future__ import annotations

import asyncio
import bisect
import itertools
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    NotRequired,
    TypedDict,
    cast,
    overload,
)

import numpy as np
from typing_extensions import ReadOnly

from zarr.core.config import config as zarr_config
from zarr.core.json_parse import convert, parse_field
from zarr.errors import ZarrRuntimeWarning

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from typing import Self

    import numpy.typing as npt

    from zarr.core.metadata.v3 import ChunkGridMetadata


ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"
ZMETADATA_V2_JSON = ".zmetadata"

BytesLike = bytes | bytearray | memoryview
ShapeLike = Iterable[int | np.integer[Any]] | int | np.integer[Any]
# Per-dimension chunk specs may mix a bare int (uniform chunk size, the
# rectilinear spec's step-size shorthand) with explicit edge-length sequences.
# A stored chunk grid (`ChunkGridMetadata`) is also accepted and is used
# verbatim, under the tolerant stored-metadata validation rules (e.g. trailing
# edges beyond the array extent, as left behind by a shrinking resize).
type ChunksLike = ShapeLike | Iterable[int | Iterable[int]] | ChunkGridMetadata
# For backwards compatibility
ChunkCoords = tuple[int, ...]
ZarrFormat = Literal[2, 3]
NodeType = Literal["array", "group"]
JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
MemoryOrder = Literal["C", "F"]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]
ANY_ACCESS_MODE: Final = "r", "r+", "a", "w", "w-"
DimensionNamesLike = Iterable[str | None] | None
DimensionNames = DimensionNamesLike  # for backwards compatibility


class NamedConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is an optional mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: NotRequired[ReadOnly[TConfig]]
    """The configuration of the object. Not required."""


class NamedRequiredConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is a mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: ReadOnly[TConfig]
    """The configuration of the object."""


def product(tup: tuple[int, ...]) -> int:
    return math.prod(tup)


def ceildiv(a: float, b: float) -> int:
    """Ceiling of ``a / b`` using floating-point division; zero when ``a`` is zero."""
    if a == 0:
        return 0
    return math.ceil(a / b)


def ceildiv_int(a: int, b: int) -> int:
    """Ceiling of integer division using exact Python integer arithmetic."""
    return -(-int(a) // int(b))


def concurrent_iter[T: tuple[Any, ...], V](
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
) -> list[asyncio.Task[V]]:
    """Launch `func(*item)` for each item concurrently, returning the tasks.

    When `limit` is set, no more than `limit` calls are in flight at once.
    Tasks are returned in input order; callers that want completion order
    should wrap the result in `asyncio.as_completed`.

    Every task is scheduled (via `ensure_future`) before this function
    returns, not on first iteration of the result. That matters for callers
    that await the returned tasks one at a time — without eager scheduling,
    each coroutine would only start when individually awaited, serializing
    the work and defeating the semaphore. It also makes the return type
    honest (real `Task`s support `.cancel()`, `.done()`, callbacks) rather
    than bare coroutines.

    See https://docs.python.org/3/library/asyncio-task.html#coroutines:
    "Note that simply calling a coroutine will not schedule it to be executed:"
    """
    if limit is None:
        return [asyncio.ensure_future(func(*item)) for item in items]

    sem = asyncio.Semaphore(limit)

    async def run(item: T) -> V:
        async with sem:
            return await func(*item)

    return [asyncio.ensure_future(run(item)) for item in items]


async def concurrent_map[T: tuple[Any, ...], V](
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
) -> list[V]:
    return await asyncio.gather(*concurrent_iter(items, func, limit))


def enum_names[E: Enum](enum: type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum[E: Enum](data: object, cls: type[E]) -> E:
    if isinstance(data, cls):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data)}")
    if data in enum_names(cls):
        return cls(data)
    raise ValueError(f"Value must be one of {list(enum_names(cls))!r}. Got {data} instead.")


def parse_name(data: JSON, expected: str | None = None) -> str:
    try:
        data = cast("str", convert(data, str))
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Expected a string, got an instance of {type(data)}.") from exc
    if expected is None or data == expected:
        return data
    raise ValueError(f"Expected '{expected}'. Got {data} instead.")


def parse_configuration(data: JSON) -> JSON:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return data


@overload
def parse_named_configuration(
    data: JSON | NamedConfig[str, Any], expected_name: str | None = None
) -> tuple[str, dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
) -> tuple[str, dict[str, JSON] | None]: ...


def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
) -> tuple[str, JSON | None]:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    if "name" not in data:
        raise ValueError(f"Named configuration does not have a 'name' key. Got {data}.")
    name_parsed = parse_name(data["name"], expected_name)
    if "configuration" in data:
        configuration_parsed = parse_configuration(data["configuration"])
    elif require_configuration:
        raise ValueError(f"Named configuration does not have a 'configuration' key. Got {data}.")
    else:
        configuration_parsed = None
    return name_parsed, configuration_parsed


def parse_shapelike(data: ShapeLike) -> tuple[int, ...]:
    """
    Parse a shape-like input into an explicit shape.
    """
    if isinstance(data, int | np.integer):
        if data < 0:
            raise ValueError(f"Expected a non-negative integer. Got {data} instead")
        return (int(data),)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = f"Expected an integer or an iterable of integers. Got {data} instead."
        raise TypeError(msg) from e

    if not all(isinstance(v, int | np.integer) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {data} instead."
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = f"Expected all values to be non-negative. Got {data} instead."
        raise ValueError(msg)

    # cast NumPy scalars to plain python ints
    return tuple(int(x) for x in data_tuple)


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data


def parse_order(data: Any) -> Literal["C", "F"]:
    return cast("Literal['C', 'F']", parse_field(data, Literal["C", "F"], "order"))


def parse_bool(data: Any) -> bool:
    return cast("bool", convert(data, bool))


def parse_int(data: Any) -> int:
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    raise ValueError(f"Expected int, got {data} instead.")


def _warn_write_empty_chunks_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `write_empty_chunks` keyword argument is deprecated and will be removed in future versions. "
        "To control whether empty chunks are written to storage, either use the `config` keyword "
        "argument, as in `config={'write_empty_chunks': True}`,"
        "or change the global 'array.write_empty_chunks' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _warn_order_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `order` keyword argument has no effect for Zarr format 3 arrays. "
        "To control the memory layout of the array, either use the `config` keyword "
        "argument, as in `config={'order': 'C'}`,"
        "or change the global 'array.order' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _default_zarr_format() -> ZarrFormat:
    """Return the default zarr_format."""
    return cast("ZarrFormat", int(zarr_config.get("default_zarr_format", 3)))


def _subject(name: str, axis: int | None) -> str:
    """`name` as the subject of an error message, prefixed by the dimension `axis`."""
    return name[0].upper() + name[1:] if axis is None else f"Dimension {axis}: {name}"


def _parse_positive_int(value: object, name: str, axis: int | None) -> int:
    """`value` as an `int` of at least 1. A `bool` is read as the `int` it equals; any
    other type, a NumPy integer or a float (even an integral one: stored documents with
    integral floats are read by `zarr.core.metadata.repair`), is rejected."""
    subject = _subject(name, axis)
    if not isinstance(value, int):
        raise TypeError(f"{subject} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{subject} must be >= 1, got {value!r}")
    return int(value)


def parse_chunk_edge(size: object, axis: int | None = None) -> int:
    """Check that `size` is a chunk edge length: an `int` of at least 1 (a `bool` is
    read as the `int` it equals).

    This is the one rule for chunk edge lengths in metadata: bare chunk sizes, explicit
    edges and run-length encoded sizes. `axis`, when given, is named in the error.
    """
    return _parse_positive_int(size, "chunk edge length", axis)


def parse_chunk_shape(data: object) -> tuple[int, ...]:
    """Check a regular chunk shape: an iterable, other than a string or a mapping, of one
    chunk edge length per axis (see `parse_chunk_edge`)."""
    match data:
        case str() | Mapping():
            pass
        case Iterable():
            return tuple(parse_chunk_edge(size, axis) for axis, size in enumerate(data))
    raise TypeError(f"A chunk shape must be an iterable of chunk edge lengths, got {data!r}")


class RunLengthEdges(Sequence[int]):
    """An immutable sequence of chunk edge lengths, stored run-length encoded.

    The sequence is held as `(size, count)` runs, with adjacent runs of the same size
    merged, so construction, lookups and prefix sums cost time and memory in the number
    of runs, not in the number of edges: `RunLengthEdges([(1, 2**40)])` is one run. It
    reads like the tuple of edges it stands for (`len`, indexing, iteration, and `==`
    against a `tuple`), but only iteration and the comparison with a `tuple` visit every
    edge.

    Sizes are not checked here: what a valid edge length is, and how to report an invalid
    one, is up to the caller (see `parse_chunk_edge`). Each count must be at least 1.

    Instances hash by their runs, so an instance does not hash like the `tuple` it
    compares equal to.
    """

    __slots__ = ("_index_stops", "_lookup_tables", "_offset_stops", "counts", "sizes")

    sizes: tuple[int, ...]
    """The edge length of each run."""
    counts: tuple[int, ...]
    """The number of edges in each run."""

    def __init__(self, runs: Iterable[tuple[int, int]] = ()) -> None:
        sizes: list[int] = []
        counts: list[int] = []
        for size, count in runs:
            if count < 1:
                raise ValueError(f"Run counts must be >= 1, got {count!r}")
            if sizes and sizes[-1] == size:
                counts[-1] += count
            else:
                sizes.append(size)
                counts.append(count)
        self.sizes = tuple(sizes)
        self.counts = tuple(counts)
        # Per run, the number of edges and the sum of edges up to and including it.
        self._index_stops = tuple(itertools.accumulate(counts))
        self._offset_stops = tuple(
            itertools.accumulate(size * count for size, count in zip(sizes, counts, strict=True))
        )
        self._lookup_tables: tuple[npt.NDArray[np.intp], ...] | None = None

    @classmethod
    def from_edges(cls, edges: Iterable[int]) -> Self:
        """Run-length encode `edges`, one entry per edge. An instance of this class is
        returned as it is."""
        if isinstance(edges, cls):
            return edges
        return cls((size, sum(1 for _ in group)) for size, group in itertools.groupby(edges))

    @property
    def runs(self) -> tuple[tuple[int, int], ...]:
        """The `(size, count)` runs, no two adjacent runs sharing a size."""
        return tuple(zip(self.sizes, self.counts, strict=True))

    @property
    def num_edges(self) -> int:
        """The number of edges. Unlike `len`, this is not limited to `sys.maxsize`."""
        return self._index_stops[-1] if self._index_stops else 0

    @property
    def total(self) -> int:
        """The sum of all edges."""
        return self._offset_stops[-1] if self._offset_stops else 0

    def __reduce__(self) -> tuple[type[RunLengthEdges], tuple[tuple[tuple[int, int], ...]]]:
        return type(self), (self.runs,)

    def __len__(self) -> int:
        return self.num_edges

    def __iter__(self) -> Iterator[int]:
        return itertools.chain.from_iterable(map(itertools.repeat, self.sizes, self.counts))

    def __contains__(self, value: object) -> bool:
        return value in self.sizes

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> RunLengthEdges: ...
    def __getitem__(self, index: int | slice) -> int | RunLengthEdges:
        num_edges = self.num_edges
        if isinstance(index, slice):
            start, stop, step = index.indices(num_edges)
            if step != 1:
                return RunLengthEdges.from_edges(self[i] for i in range(start, stop, step))
            runs: list[tuple[int, int]] = []
            run_start = 0
            for size, run_stop in zip(self.sizes, self._index_stops, strict=True):
                count = min(stop, run_stop) - max(start, run_start)
                if count > 0:
                    runs.append((size, count))
                run_start = run_stop
            return RunLengthEdges(runs)
        position = index + num_edges if index < 0 else index
        if not 0 <= position < num_edges:
            raise IndexError(f"Edge index {index} is out of range for {num_edges} edges")
        return self.sizes[bisect.bisect_right(self._index_stops, position)]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RunLengthEdges):
            return self.sizes == other.sizes and self.counts == other.counts
        if isinstance(other, tuple):
            return len(other) == self.num_edges and all(
                a == b for a, b in zip(self, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.sizes, self.counts))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_rle()})"

    def count(self, value: object) -> int:
        return sum(count for size, count in self.runs if size == value)

    def to_rle(self) -> list[int | list[int]]:
        """The mixed run-length encoding of the rectilinear chunk grid spec: a run of
        one edge is its bare size, a longer run is `[size, count]`."""
        return [[size, count] if count > 1 else size for size, count in self.runs]

    def with_edge(self, size: int) -> RunLengthEdges:
        """A copy with one more edge of length `size` at the end."""
        return RunLengthEdges((*self.runs, (size, 1)))

    def offset_of(self, index: int) -> int:
        """The sum of the first `index` edges: where edge `index` starts, or `total`
        for `index == num_edges`."""
        if not 0 <= index <= self.num_edges:
            raise IndexError(f"Edge index {index} is out of range for {self.num_edges} edges")
        if index == self.num_edges:
            return self.total
        run = bisect.bisect_right(self._index_stops, index)
        if run == 0:
            return index * self.sizes[0]
        return self._offset_stops[run - 1] + (index - self._index_stops[run - 1]) * self.sizes[run]

    def index_at(self, position: int) -> int:
        """The index of the edge that covers `position`, a coordinate in `[0, total)`
        along the axis the edges tile."""
        if not 0 <= position < self.total:
            raise IndexError(
                f"Position {position} is out of range for edges summing to {self.total}"
            )
        run = bisect.bisect_right(self._offset_stops, position)
        if run == 0:
            return position // self.sizes[0]
        return (
            self._index_stops[run - 1] + (position - self._offset_stops[run - 1]) // self.sizes[run]
        )

    def indices_at(self, positions: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Vectorized `index_at`, without the bounds check: a negative position gives 0,
        and a position of `total` or more gives `num_edges`."""
        positions = np.asarray(positions)
        if not self.sizes:
            return np.zeros(positions.shape, dtype=np.intp)
        if self._lookup_tables is None:
            # No position reaches a run that starts past the largest intp, and a size
            # that large is never exceeded, so the tables always fit intp.
            intp_max = int(np.iinfo(np.intp).max)
            offsets = [o for o in (0, *self._offset_stops[:-1]) if o <= intp_max]
            n = len(offsets)
            self._lookup_tables = (
                np.array(offsets, dtype=np.intp),
                np.array((0, *self._index_stops[: n - 1]), dtype=np.intp),
                np.array([min(size, intp_max) for size in self.sizes[:n]], dtype=np.intp),
                np.array(min(self.num_edges, intp_max), dtype=np.intp),
            )
        offsets_arr, indices_arr, sizes_arr, last = self._lookup_tables
        run = np.searchsorted(offsets_arr[1:], positions, side="right")
        found = indices_arr[run] + (positions - offsets_arr[run]) // sizes_arr[run]
        return np.clip(found, 0, last).astype(np.intp, copy=False)


def _iter_rle_runs(data: Iterable[object], axis: int | None) -> Iterator[tuple[int, int]]:
    """The `(size, count)` runs of a mixed array of bare integers and RLE pairs, each
    checked as it is read."""
    for item in data:
        if isinstance(item, list):
            if len(item) != 2:
                subject = _subject("RLE entries", axis)
                raise ValueError(f"{subject} must be an integer or [size, count], got {item}")
            size, count = item
            repeat = _parse_positive_int(count, "RLE repeat count", axis)
            yield parse_chunk_edge(size, axis), repeat
        else:
            yield parse_chunk_edge(item, axis), 1


def parse_rle(data: Sequence[object], axis: int | None = None) -> RunLengthEdges:
    """Read a mixed array of bare integers and RLE pairs, the edges of dimension `axis`
    (named in errors, when given), without expanding it.

    Per the rectilinear chunk grid spec, each element can be:
    - a bare integer (an explicit edge length)
    - a two-element array ``[value, count]`` (run-length encoded)

    The cost is in the number of elements, whatever the repeat counts.
    """
    return RunLengthEdges(_iter_rle_runs(data, axis))


def expand_rle(data: Sequence[object], axis: int | None = None) -> list[int]:
    """Expand a mixed array of bare integers and RLE pairs, the edges of dimension
    `axis` (named in errors, when given), into one entry per edge.

    The result is as long as the sum of the repeat counts; `parse_rle` reads the same
    input without expanding it.
    """
    result: list[int] = []
    for size, count in _iter_rle_runs(data, axis):
        result.extend([size] * count)
    return result


def compress_rle(sizes: Sequence[int]) -> list[int | list[int]]:
    """Compress chunk sizes to mixed RLE format per the rectilinear spec.

    Runs of length > 1 are emitted as ``[value, count]`` pairs; runs of
    length 1 are emitted as bare integers::

        [10, 10, 10, 5] -> [[10, 3], 5]
    """
    return RunLengthEdges.from_edges(sizes).to_rle()


def validate_rectilinear_kind(kind: str | None) -> None:
    """Validate the ``kind`` field of a rectilinear chunk grid configuration.

    The rectilinear spec requires ``kind: "inline"``.
    """
    if kind is None:
        raise ValueError(
            "Rectilinear chunk grid configuration requires a 'kind' field. "
            "Only 'inline' is currently supported."
        )
    if kind != "inline":
        raise ValueError(
            f"Unsupported rectilinear chunk grid kind: {kind!r}. "
            "Only 'inline' is currently supported."
        )


def validate_rectilinear_edges(
    chunk_shapes: Sequence[int | Sequence[int]], array_shape: Sequence[int]
) -> None:
    """Validate that rectilinear chunk edges cover the array extent per dimension.

    Bare-int dimensions (regular step) always cover any extent, so they are
    skipped. Explicit edge lists must sum to at least the array extent.
    """
    for i, (dim_spec, extent) in enumerate(zip(chunk_shapes, array_shape, strict=True)):
        if isinstance(dim_spec, int):
            continue
        edge_sum = dim_spec.total if isinstance(dim_spec, RunLengthEdges) else sum(dim_spec)
        if edge_sum < extent:
            raise ValueError(
                f"Rectilinear chunk edges for dimension {i} sum to {edge_sum} "
                f"but array shape extent is {extent} (edge sum must be >= extent)"
            )
