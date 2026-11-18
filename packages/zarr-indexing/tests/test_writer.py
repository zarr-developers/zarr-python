from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform
from zarr_indexing.writer import write_into


class BasicSource:
    """Reject source reads and advanced writes, recording basic writes."""

    def __init__(self, data: np.ndarray[Any, Any]) -> None:
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.calls: list[tuple[int | slice, ...]] = []

    def __setitem__(self, key: tuple[int | slice, ...], value: Any) -> None:
        assert all(isinstance(item, (int, slice)) for item in key)
        assert all(
            not isinstance(item, slice) or item.step is None or item.step > 0 for item in key
        )
        self.calls.append(key)
        self.data[key] = value


def _assert_each_cell_holds_a_written_value(
    actual: Any, initial: Any, identifiers: Any, values: Any
) -> None:
    """Cells the selection misses are unchanged; each addressed cell holds one value written to it.

    Which of a repeated cell's values lands is unspecified, so any of them passes.
    """
    candidates: dict[int, set[Any]] = {}
    for identifier, value in zip(
        np.asarray(identifiers).flat, np.asarray(values).flat, strict=True
    ):
        candidates.setdefault(int(identifier), set()).add(value.item())
    flat_actual = np.asarray(actual).reshape(-1)
    flat_initial = np.asarray(initial).reshape(-1)
    for cell, (got, before) in enumerate(
        zip(flat_actual.tolist(), flat_initial.tolist(), strict=True)
    ):
        assert got in candidates.get(cell, {before}), (cell, got, candidates.get(cell, {before}))


@pytest.mark.parametrize(
    "case",
    ["reverse", "scalar", "newaxis", "empty", "outer", "composed", "correlated", "transpose"],
)
def test_write_coordinates(case: str) -> None:
    data = np.arange(24).reshape(4, 6)
    initial = data.copy()
    # Select flat source identifiers independently of the transform evaluator.
    identifiers = np.arange(data.size).reshape(data.shape)
    transform = IndexTransform.from_shape(data.shape)
    if case == "reverse":
        transform = transform[::-1, ::2]
        identifiers = identifiers[::-1, ::2]
    elif case == "scalar":
        transform = transform[2, 3]
        identifiers = identifiers[2, 3]
    elif case == "newaxis":
        transform = IndexTransform(
            IndexDomain.from_shape((1, 4, 6)), (DimensionMap(1), DimensionMap(2))
        )
        identifiers = identifiers[None]
    elif case == "empty":
        transform = transform[1:1, :]
        identifiers = identifiers[1:1, :]
    elif case == "outer":
        transform = transform.oindex[[3, 1, 3], [5, 0, 5]]
        identifiers = identifiers[np.ix_([3, 1, 3], [5, 0, 5])]
    elif case == "composed":
        transform = transform.oindex[[3, 1, 3], :].oindex[[2, 0], [5, 0]]
        identifiers = identifiers[[3, 1, 3]][np.ix_([2, 0], [5, 0])]
    elif case == "correlated":
        transform = IndexTransform(
            IndexDomain.from_shape((3, 2)),
            (
                ArrayMap(np.array([[3], [0], [3]], dtype=np.intp)),
                ArrayMap(np.array([[5, 1], [0, 2], [5, 1]], dtype=np.intp)),
            ),
        )
        identifiers = identifiers[np.array([[3], [0], [3]]), np.array([[5, 1], [0, 2], [5, 1]])]
    else:
        transform = IndexTransform(
            IndexDomain.from_shape((6, 4)), (DimensionMap(1), DimensionMap(0))
        )
        identifiers = identifiers.T
    values = np.arange(identifiers.size).reshape(identifiers.shape) + 100
    source = BasicSource(data)
    assert write_into(source, transform, values) is None
    _assert_each_cell_holds_a_written_value(data, initial, identifiers, values)
    if case in ("reverse", "scalar", "newaxis", "transpose"):
        assert len(source.calls) == 1
    elif case == "empty":
        assert source.calls == []


@pytest.mark.parametrize("fancy", [False, True])
def test_write_alias_and_broadcast(fancy: bool) -> None:
    data = np.arange(6)
    expected = data[::-1].copy()
    transform = IndexTransform.from_shape(data.shape)
    transform = transform.oindex[[5, 4, 3, 2, 1, 0]] if fancy else transform[::-1]
    write_into(BasicSource(data), transform, data)
    np.testing.assert_array_equal(data, expected)
    write_into(BasicSource(data), transform, np.array([[17]]))
    np.testing.assert_array_equal(data, np.full(6, 17))


def test_write_broadcast_error_before_mutation() -> None:
    source = BasicSource(np.zeros((4, 6)))
    with pytest.raises(ValueError):
        write_into(source, IndexTransform.from_shape(source.shape), [1, 2])
    assert source.calls == []


def test_write_cast_error_before_mutation() -> None:
    source = BasicSource(np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        write_into(source, IndexTransform.from_shape(source.shape), ["1", "invalid"])
    assert source.calls == []


def test_write_bounds_error_before_mutation() -> None:
    source = BasicSource(np.zeros(2))
    transform = IndexTransform(
        IndexDomain.from_shape((2,)), (ArrayMap(np.array([0, 2], dtype=np.intp)),)
    )
    with pytest.raises(IndexError, match="outside"):
        write_into(source, transform, [1, 2])
    assert source.calls == []


def test_write_readonly_error() -> None:
    data = np.zeros(2)
    data.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        write_into(data, IndexTransform.from_shape(data.shape), 1)


def test_write_constant_repeated_destination() -> None:
    source = BasicSource(np.zeros(3))
    transform = IndexTransform(IndexDomain.from_shape((2, 3)), (ConstantMap(1),))
    write_into(source, transform, [[1], [9]])
    assert source.data[[0, 2]].tolist() == [0, 0]
    assert source.data[1] in (1, 9)
    assert len(source.calls) == 6


@pytest.mark.parametrize("scalar", [False, True])
def test_write_scalar_source_and_shared_dimensions(scalar: bool) -> None:
    if scalar:
        source = BasicSource(np.array(0))
        transform = IndexTransform.from_shape(())
        expected = np.array(7)
    else:
        source = BasicSource(np.zeros((3, 3), dtype=np.int64))
        transform = IndexTransform(IndexDomain.from_shape((3,)), (DimensionMap(0), DimensionMap(0)))
        expected = np.diag([7, 7, 7])
    write_into(source, transform, 7)
    np.testing.assert_array_equal(source.data, expected)


def test_write_rank_error_before_mutation() -> None:
    source = BasicSource(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="output rank"):
        write_into(source, IndexTransform.from_shape((2,)), 1)
    assert source.calls == []


@pytest.mark.parametrize("masked_source", [False, True])
@pytest.mark.parametrize("fancy", [False, True])
def test_write_masked_values(masked_source: bool, fancy: bool) -> None:
    data: Any = np.arange(12).reshape(3, 4)
    if masked_source:
        data = np.ma.array(data, mask=np.ones(data.shape, dtype=bool))
    expected = data.copy()
    values = np.ma.array([[31, 32, 33, 34]], mask=[[False, True, False, True]])
    transform = IndexTransform.from_shape(data.shape)
    if fancy:
        transform = transform.oindex[[2, 0, 2], :]
        expected[[2, 0, 2], :] = values
    else:
        transform = transform[::-1, :]
        expected[::-1, :] = values
    write_into(data, transform, values)
    np.testing.assert_array_equal(np.ma.getdata(data), np.ma.getdata(expected))
    np.testing.assert_array_equal(np.ma.getmaskarray(data), np.ma.getmaskarray(expected))


def test_write_masked_lazy_alias() -> None:
    from zarr_indexing.lazy_array import LazyArray

    source = np.ma.array(np.arange(6), mask=[True, False, False, True, False, True])
    expected = source[::-1].copy()
    array = LazyArray(source)
    array[::-1] = array
    np.testing.assert_array_equal(source.data, expected.data)
    np.testing.assert_array_equal(source.mask, expected.mask)


@pytest.mark.parametrize("backend", ["numpy", "zarr"])
@pytest.mark.parametrize("case", ["basic", "outer", "vector", "newaxis", "empty", "alias"])
def test_lazy_write_and_assignment(backend: str, case: str) -> None:
    from zarr_indexing.lazy_array import LazyArray

    initial = np.arange(20).reshape(4, 5)
    source: Any = initial.copy()
    if backend == "zarr":
        zarr = pytest.importorskip("zarr")
        source = zarr.create_array({}, shape=initial.shape, chunks=(2, 3), dtype="int64")
        source[:] = initial
    array = LazyArray(source)
    identifiers = initial.copy()
    if case == "basic":
        view = array[::-1, ::2][1:, 1:]
        identifiers = identifiers[::-1, ::2][1:, 1:]
    elif case == "outer":
        view = array.oindex[[3, 0, 3], [4, 1]].oindex[[2, 0, 1], [1, 0]]
        identifiers = identifiers[np.ix_([3, 0, 3], [4, 1])][np.ix_([2, 0, 1], [1, 0])]
    elif case == "vector":
        view = array.oindex[[3, 0, 3], :].vindex[[2, 0, 1], [4, 4, 1]]
        identifiers = identifiers[[3, 0, 3], :][[2, 0, 1], [4, 4, 1]]
    elif case == "newaxis":
        view = array[None, 1:3, :]
        identifiers = identifiers[None, 1:3, :]
    elif case == "empty":
        view = array[2:2, :]
        identifiers = identifiers[2:2, :]
    else:
        array[::-1, :] = array
        np.testing.assert_array_equal(source[:], initial[::-1])
        return
    values = np.arange(identifiers.size).reshape(identifiers.shape) + 100
    assert view.write(values) is None
    _assert_each_cell_holds_a_written_value(source[:], initial, identifiers, values)
    # All selector assignment entry points write through to the original source.
    written = np.array(source[:])
    array.oindex[[0, 3], [1, 4]] = 77
    written[np.ix_([0, 3], [1, 4])] = 77
    np.testing.assert_array_equal(source[:], written)
    array.vindex[[0, 0, 2], [2, 2, 4]] = [10, 11, 12]
    _assert_each_cell_holds_a_written_value(
        source[:], written, initial[[0, 0, 2], [2, 2, 4]], [10, 11, 12]
    )


@pytest.mark.parametrize("backend", ["numpy", "zarr"])
def test_random_selection_chain_scatter(backend: str) -> None:
    """Map selected integer labels back to storage without consulting transforms."""
    from zarr_indexing.lazy_array import LazyArray

    rng = np.random.default_rng(58123)
    for _ in range(24):
        initial = np.arange(30).reshape(5, 6)
        source: Any = initial.copy()
        if backend == "zarr":
            zarr = pytest.importorskip("zarr")
            source = zarr.create_array({}, shape=initial.shape, chunks=(2, 3), dtype="int64")
            source[:] = initial
        view = LazyArray(source)
        identifiers = initial
        for _ in range(3):
            if rng.integers(2):
                rows = rng.integers(0, view.shape[0], size=4)
                cols = rng.integers(0, view.shape[1], size=3)
                view = view.oindex[rows, cols]
                identifiers = identifiers[np.ix_(rows, cols)]
            else:
                step = int(rng.choice([-2, -1, 1, 2]))
                view = view[::step, ::-1]
                identifiers = identifiers[::step, ::-1]
        values = rng.integers(100, 200, size=identifiers.shape)
        view.write(values)
        _assert_each_cell_holds_a_written_value(source[:], initial, identifiers, values)


class _ChunkTouches:
    """Count the storage round trips a zarr array makes, one per chunk key."""

    def __init__(self) -> None:
        self.gets = 0
        self.sets = 0

    @classmethod
    def wrap(cls, zarr: Any) -> tuple[Any, _ChunkTouches]:
        counter = cls()

        class CountingStore(zarr.storage.MemoryStore):
            async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
                counter.gets += 1
                return await super().get(key, prototype, byte_range)

            async def set(self, key: str, value: Any) -> None:
                counter.sets += 1
                await super().set(key, value)

        return CountingStore(), counter


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("case", ["outer_rows", "vector_points", "duplicates"])
def test_fancy_writes_touch_each_write_chunk_once(sharded: bool, case: str) -> None:
    """Storage round trips scale with touched write chunks, not selected elements."""
    zarr = pytest.importorskip("zarr")
    from zarr_indexing.lazy_array import LazyArray

    store, touches = _ChunkTouches.wrap(zarr)
    kwargs: dict[str, Any] = {"chunks": (2, 2), "shards": (4, 4)} if sharded else {"chunks": (4, 4)}
    source = zarr.create_array(store, shape=(12, 12), dtype="int64", **kwargs)
    source[:] = 0
    expected = np.zeros((12, 12), dtype=np.int64)
    touches.gets = touches.sets = 0
    view = LazyArray(source)
    if case == "outer_rows":
        rows = [1, 5, 11]
        values = np.arange(3 * 12).reshape(3, 12)
        view.oindex[rows, :] = values
        expected[rows, :] = values
        touched = {(r // 4, c) for r in rows for c in range(3)}
    elif case == "vector_points":
        rows, cols = [0, 7, 7, 11, 3], [0, 2, 9, 11, 5]
        values = np.arange(5) + 10
        view.vindex[rows, cols] = values
        expected[rows, cols] = values
        touched = {(r // 4, c // 4) for r, c in zip(rows, cols, strict=True)}
    else:
        view.oindex[[5, 5, 6], 2:5] = np.array([[1] * 3, [2] * 3, [3] * 3])
        expected[6, 2:5] = 3
        touched = {(1, 0), (1, 1)}
    gets, sets = touches.gets, touches.sets
    actual = source[:]
    if case == "duplicates":
        # Row 5 is written twice; either row of values may land, but one whole value per cell.
        assert set(actual[5, 2:5].tolist()) <= {1, 2}
        expected[5, 2:5] = actual[5, 2:5]
    np.testing.assert_array_equal(actual, expected)
    assert sets == len(touched), (sets, len(touched))
    # At most the hull read plus zarr's own partial-chunk read per cell.
    assert gets <= 2 * len(touched), (gets, len(touched))


def test_write_grid_prefers_write_chunk_sizes_over_chunks() -> None:
    """A source whose write grid is coarser than its read grid is written by the write grid."""
    from zarr_indexing.lazy_array import LazyArray

    class ShardedSource:
        chunks = ((2, 2, 2), (3, 3))
        write_chunk_sizes = ((6,), (3, 3))

        def __init__(self) -> None:
            self.data = np.zeros((6, 6), dtype=np.int64)
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            self.boxes: list[tuple[slice, ...]] = []

        def __getitem__(self, key: Any) -> Any:
            return self.data[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            self.boxes.append(key)
            self.data[key] = value

    source = ShardedSource()
    LazyArray(source).oindex[[0, 5], :] = 7
    expected = np.zeros((6, 6), dtype=np.int64)
    expected[[0, 5], :] = 7
    np.testing.assert_array_equal(source.data, expected)
    # One read-modify-write per touched write cell: two column shards, each
    # spanning all six rows, rather than the four inner chunks a read grid names.
    assert len(source.boxes) == 2
    assert all(box[0] == slice(0, 6) for box in source.boxes)


def test_write_only_source_falls_back_to_element_assignment() -> None:
    source = BasicSource(np.zeros((3, 4), dtype=np.int64))
    transform = IndexTransform.from_shape(source.shape).oindex[[2, 0], [1, 3]]
    write_into(source, transform, np.array([[1, 2], [3, 4]]))
    assert len(source.calls) == 4
    expected = np.zeros((3, 4), dtype=np.int64)
    expected[np.ix_([2, 0], [1, 3])] = [[1, 2], [3, 4]]
    np.testing.assert_array_equal(source.data, expected)


def test_masked_zero_rank_affine_write_keeps_payload() -> None:
    from zarr_indexing.lazy_array import LazyArray

    source = np.ma.array([1, 2, 3])
    LazyArray(source)[1].write(np.ma.array(7, mask=True))
    np.testing.assert_array_equal(source.data, [1, 7, 3])
    np.testing.assert_array_equal(np.ma.getmaskarray(source), [False, True, False])


def test_write_rank_error_precedes_value_broadcasting() -> None:
    source = np.zeros((3, 4), dtype=np.int64)
    transform = IndexTransform.from_shape((3,))
    with pytest.raises(ValueError, match="rank"):
        write_into(source, transform, np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(source, 0)


def test_empty_selection_still_validates_values() -> None:
    source = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="broadcast"):
        write_into(source, IndexTransform.from_shape((5,))[2:2], np.arange(3))
    with pytest.raises(ValueError):
        write_into(source, IndexTransform.from_shape((5,))[2:2], ["not a number"])
    np.testing.assert_array_equal(source, 0)


def test_gridless_readable_source_never_reads() -> None:
    """Without a write grid the hull could be the whole array, so elements are assigned."""
    from zarr_indexing.lazy_array import LazyArray

    class ReadableSource(BasicSource):
        def __getitem__(self, key: Any) -> Any:
            raise AssertionError("gridless writes must not read the source")

    source = ReadableSource(np.zeros((4, 4), dtype=np.int64))
    LazyArray(source).vindex[[0, 3], [3, 0]] = [1, 2]
    expected = np.zeros((4, 4), dtype=np.int64)
    expected[[0, 3], [3, 0]] = [1, 2]
    np.testing.assert_array_equal(source.data, expected)
    assert len(source.calls) == 2


def test_unfactorable_transform_falls_back_to_elements_on_a_gridded_source() -> None:
    """A diagonal reads one input axis twice, which the planner rejects; it is still written."""

    class GriddedSource(BasicSource):
        chunks = ((2, 2), (2, 2))

        def __getitem__(self, key: Any) -> Any:
            return self.data[key]

    source = GriddedSource(np.zeros((4, 4), dtype=np.int64))
    diagonal = IndexTransform(IndexDomain.from_shape((4,)), (DimensionMap(0), DimensionMap(0)))
    write_into(source, diagonal, np.arange(1, 5), write_grid=None)
    from zarr_indexing.grid import dimension_grids_from_chunks

    source.data[:] = 0
    source.calls.clear()
    write_into(
        source,
        diagonal,
        np.arange(1, 5),
        write_grid=dimension_grids_from_chunks(source.chunks, (4, 4)),
    )
    np.testing.assert_array_equal(source.data, np.diag([1, 2, 3, 4]))
    assert len(source.calls) == 4


def test_writes_bypass_the_reader_so_a_caching_reader_serves_stale_values() -> None:
    """Pins the documented hazard: the reader is not invalidated by `write`."""
    from zarr_indexing.lazy_array import LazyArray
    from zarr_indexing.reader import ReadContext, basic_reader

    class CachingReader:
        def __init__(self) -> None:
            self.cache: dict[tuple[Any, ...], Any] = {}

        def read_into(self, source: Any, context: ReadContext, out: Any, /) -> None:
            key = (context.transform.domain.shape, str(context.transform.to_json()))
            if key not in self.cache:
                basic_reader.read_into(source, context, out)
                self.cache[key] = out.copy()
            out[...] = self.cache[key]

    source = np.arange(6)
    view = LazyArray(source).with_reader(CachingReader())[1:4]
    np.testing.assert_array_equal(view.result(), [1, 2, 3])
    view.write([7, 8, 9])
    np.testing.assert_array_equal(source, [0, 7, 8, 9, 4, 5])
    np.testing.assert_array_equal(view.result(), [1, 2, 3])


class GridSource:
    """A readable, non-NumPy source with a write grid, so writes scatter cell by cell."""

    def __init__(self, data: np.ndarray[Any, Any], chunks: tuple[tuple[int, ...], ...]) -> None:
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.chunks = chunks
        self.reads = 0

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray[Any, Any]:
        assert all(isinstance(item, slice) for item in key)
        self.reads += 1
        return self.data[key]

    def __setitem__(self, key: tuple[slice, ...], value: Any) -> None:
        assert all(isinstance(item, slice) for item in key)
        self.data[key] = value


def _reverse_within_groups(
    keys: np.ndarray[Any, np.dtype[np.intp]],
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Group equal keys, listing each group's entries in reverse request order."""
    return np.asarray(np.lexsort((-np.arange(keys.size), keys)), dtype=np.intp)


def _repeated_selection(case: str, view: Any, identifiers: Any) -> tuple[Any, Any]:
    """A selection that addresses some cells twice and lists each grid cell's points unsorted."""
    # On the 4-by-4 grid below, rows 3, 1, 3, 0, 2, 1 land in the first chunk
    # row and 6, 5 in the second; the columns split 7, 5, 6 against 2, 2, 0.
    rows = [3, 1, 3, 0, 6, 2, 1, 5]
    cols = [7, 2, 5, 2, 0, 6]
    if case == "orthogonal":
        return view.oindex[rows, cols], identifiers[np.ix_(rows, cols)]
    if case == "vectorized":
        points = ([3, 1, 3, 6, 5, 1, 0, 6], [2, 0, 2, 7, 4, 0, 3, 5])
        return view.vindex[points], identifiers[points]
    # Mixed: a vectorized pair over an orthogonal gather of the rows.
    gathered = view.oindex[rows, :]
    points = ([0, 2, 4, 1, 6, 0, 7, 3], [6, 6, 1, 0, 0, 6, 5, 1])
    return gathered.vindex[points], identifiers[rows, :][points]


@pytest.mark.parametrize("grouping", ["default", "reversed"])
# The per-cell path copies blocks into plain arrays, so only NumPy sources carry masks.
@pytest.mark.parametrize("backend", ["planned", "numpy", "numpy-masked"])
@pytest.mark.parametrize("case", ["orthogonal", "vectorized", "mixed"])
def test_repeated_writes_land_one_whole_value_per_cell(
    monkeypatch: pytest.MonkeyPatch, case: str, backend: str, grouping: str
) -> None:
    """Repeated targets get one of their values whatever order the planner lists a cell's points in.

    Targets addressed once get exactly their value, so a writer that pairs a
    point with another point's value fails whichever order grouping produces.
    """
    from zarr_indexing import chunk_resolution
    from zarr_indexing.lazy_array import LazyArray

    if grouping == "reversed":
        monkeypatch.setattr(chunk_resolution, "_group_by_key", _reverse_within_groups)
    initial = np.arange(64, dtype=np.int64).reshape(8, 8)
    masked = backend == "numpy-masked"
    data: Any = initial.copy()
    if masked:
        data = np.ma.array(data, mask=np.zeros(data.shape, dtype=bool))
    source: Any = GridSource(data, ((4, 4), (4, 4))) if backend == "planned" else data
    view, identifiers = _repeated_selection(case, LazyArray(source), np.arange(64).reshape(8, 8))
    values = np.arange(identifiers.size).reshape(identifiers.shape) * 7 + 100
    if masked:
        # Mask values by parity so a cell whose data and mask came from two
        # different writes is caught: written value v is masked iff v is odd.
        values = np.ma.array(values, mask=values % 2 == 1)
    view.write(values)
    if backend == "planned":
        assert source.reads > 0
    _assert_each_cell_holds_a_written_value(np.ma.getdata(data), initial, identifiers, values)
    if masked:
        np.testing.assert_array_equal(
            np.ma.getmaskarray(data),
            np.isin(np.arange(64), identifiers).reshape(8, 8) & (np.ma.getdata(data) % 2 == 1),
        )
