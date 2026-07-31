"""Tests for `zarr_indexing.grid` grids and the `LazyArray` wrapper.

The happy-path suite is a single oracle test: every selection case is applied
both to a `LazyArray` and to the NumPy array it wraps, and the results must
match. The case list is crossed with four source flavors — an unchunked NumPy
array, NumPy with each of the two declared chunk conventions, and a zarr array
whose chunking is auto-discovered — so the chunked and unchunked resolution
strategies are held to the same answers.
"""

from __future__ import annotations

import operator
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr_indexing import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    EdgeDimensionGrid,
    IndexTransform,
    LazyArray,
    array_map_dependent_axis,
    dimension_grids_from_chunks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SHAPE = (7, 5, 4)
PART_SHAPE = (3, 2, 3)
EXPLICIT_PARTS = ((3, 3, 1), (2, 2, 1), (3, 1))


def reference() -> np.ndarray[Any, np.dtype[np.int64]]:
    """The array every source flavor holds, and the oracle for every case."""
    return np.arange(int(np.prod(SHAPE)), dtype=np.int64).reshape(SHAPE)


def outer(ref: np.ndarray[Any, Any], selections: Sequence[Any]) -> np.ndarray[Any, Any]:
    """NumPy oracle for orthogonal indexing: the outer product of per-axis selections.

    Scalar integers are basic indices — NumPy applies them first and drops the
    axis — so they are peeled off before the outer product is formed.
    """
    scalars = tuple(
        sel if isinstance(sel, (int, np.integer)) and not isinstance(sel, bool) else slice(None)
        for sel in selections
    )
    reduced = ref[scalars]
    axes = [
        np.arange(size)[sel]
        for size, sel in zip(
            reduced.shape,
            [
                s
                for s in selections
                if not (isinstance(s, (int, np.integer)) and not isinstance(s, bool))
            ],
            strict=True,
        )
    ]
    if len(axes) == 0:
        return reduced
    return reduced[np.ix_(*axes)]


# ---------------------------------------------------------------------------
# EdgeDimensionGrid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sizes", "expected_index_to_chunk", "expected_offsets", "expected_sizes"),
    [
        # A clipped edge chunk.
        ((3, 3, 1), [0, 0, 0, 1, 1, 1, 2], [0, 3, 6], [3, 3, 1]),
        # A single chunk covering the whole axis.
        ((4,), [0, 0, 0, 0], [0], [4]),
        # A size-1 axis.
        ((1,), [0], [0], [1]),
        # Irregular sizes: the grid does not have to be regular.
        ((1, 2, 1), [0, 1, 1, 2], [0, 1, 3], [1, 2, 1]),
    ],
)
def test_edge_dimension_grid(
    sizes: tuple[int, ...],
    expected_index_to_chunk: list[int],
    expected_offsets: list[int],
    expected_sizes: list[int],
) -> None:
    """All four `DimensionGridLike` methods agree with hand-computed values."""
    grid = EdgeDimensionGrid(sizes)
    extent = sum(sizes)

    assert grid.num_chunks == len(sizes)
    assert grid.extent == extent
    assert [grid.index_to_chunk(i) for i in range(extent)] == expected_index_to_chunk
    assert [grid.chunk_offset(c) for c in range(len(sizes))] == expected_offsets
    assert [grid.chunk_size(c) for c in range(len(sizes))] == expected_sizes
    np.testing.assert_array_equal(
        grid.indices_to_chunks(np.arange(extent, dtype=np.intp)),
        np.asarray(expected_index_to_chunk, dtype=np.intp),
    )


def test_edge_dimension_grid_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="chunk sizes must be positive"):
        EdgeDimensionGrid((3, 0, 2))


def test_edge_dimension_grid_rejects_out_of_bounds_index() -> None:
    with pytest.raises(IndexError, match="out of bounds for an axis of extent 4"):
        EdgeDimensionGrid((3, 1)).index_to_chunk(4)


def test_edge_dimension_grid_rejects_out_of_bounds_chunk() -> None:
    with pytest.raises(IndexError, match="chunk index 2 is out of bounds"):
        EdgeDimensionGrid((3, 1)).chunk_offset(2)


@pytest.mark.parametrize(
    ("chunks", "shape", "expected"),
    [
        # Uniform chunk shape, tail clipped.
        ((3, 4), (7, 4), ((3, 3, 1), (4,))),
        # Dask-convention per-axis sizes, passed through.
        (((3, 3, 1), (2, 2)), (7, 4), ((3, 3, 1), (2, 2))),
        # A chunk longer than the axis collapses to one clipped chunk.
        ((10,), (4,), ((4,),)),
        # A zero-length axis has no chunks at all.
        ((3,), (0,), ((),)),
    ],
)
def test_dimension_grids_from_chunks(
    chunks: Any, shape: tuple[int, ...], expected: tuple[tuple[int, ...], ...]
) -> None:
    grids = dimension_grids_from_chunks(chunks, shape)
    assert tuple(grid.sizes for grid in grids) == expected


def test_dimension_grids_from_chunks_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="one entry per dimension"):
        dimension_grids_from_chunks((3,), (7, 4))


def test_dimension_grids_from_chunks_rejects_mixed_conventions() -> None:
    with pytest.raises(ValueError, match="not a mixture"):
        dimension_grids_from_chunks((3, (2, 2)), (7, 4))


def test_dimension_grids_from_chunks_rejects_wrong_total() -> None:
    with pytest.raises(ValueError, match="sum to 5, but the array extent is 7"):
        dimension_grids_from_chunks(((3, 2), (4,)), (7, 4))


def test_dimension_grids_from_chunks_rejects_non_sequence_entry() -> None:
    """A float entry is neither convention, and says so instead of raising TypeError."""
    with pytest.raises(ValueError, match=r"3\.5 at dimension 0 is neither"):
        dimension_grids_from_chunks((3.5, (4,)), (7, 4))


def test_array_map_dependent_axis_reports_no_axis() -> None:
    """A map varying over nothing answers None rather than a stale binding."""
    correlated = ArrayMap(index_array=np.array([[1], [2]], dtype=np.intp))
    assert array_map_dependent_axis(correlated) == 0
    assert array_map_dependent_axis(ArrayMap(index_array=np.array([[3]], dtype=np.intp))) is None


def test_scalar_on_a_fancy_axis_collapses_to_a_constant() -> None:
    """The degenerate-collapse rule: an all-singleton ArrayMap becomes a ConstantMap.

    Without it the map keeps an `input_dimension` naming an axis the integer
    index just removed, which after renumbering aliases a different axis.
    """
    view = IndexTransform.from_shape((7, 5)).oindex[np.array([3, 1]), slice(None)][0]
    assert view.output[0] == ConstantMap(offset=3)
    assert isinstance(view.output[1], DimensionMap)
    assert view.domain.shape == (5,)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def make_zarr_source() -> Any:
    """A zarr array holding `reference()`, whose parts are auto-discovered."""
    zarr = pytest.importorskip("zarr")
    array = zarr.create_array({}, shape=SHAPE, chunks=PART_SHAPE, dtype="int64")
    array[:] = reference()
    return array


def make_source(flavor: str) -> LazyArray:
    """Build a `LazyArray` over `reference()` with the requested partitioning."""
    data = reference()
    if flavor == "numpy-whole":
        return LazyArray(data)
    if flavor == "numpy-explicit-parts":
        return LazyArray(data).with_parts(EXPLICIT_PARTS)
    if flavor == "numpy-uniform-parts":
        return LazyArray(data).with_parts(PART_SHAPE)
    if flavor == "zarr":
        return LazyArray(make_zarr_source())
    if flavor == "zarr-misaligned":
        # Parts that deliberately straddle the zarr array's own chunks.
        return LazyArray(make_zarr_source()).with_parts((4, 3, 3))
    raise AssertionError(f"unknown source flavor {flavor!r}")


FLAVORS = [
    "numpy-whole",
    "numpy-explicit-parts",
    "numpy-uniform-parts",
    "zarr",
    "zarr-misaligned",
]


@pytest.fixture(params=FLAVORS)
def source(request: pytest.FixtureRequest) -> LazyArray:
    return make_source(request.param)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

MASK = (reference() % 11) == 0
# A mask over the two trailing axes, for the `vindex[..., mask]` idiom.
TRAILING_MASK = (reference()[0] % 3) == 0

# (id, lazy view builder, NumPy oracle). Integer scalars are deliberately absent
# from the orthogonal cases: the transform algebra keeps an int-selected axis as
# a length-1 axis under `oindex`, which `np.ix_` cannot express.
CASES: list[tuple[str, Callable[[LazyArray], LazyArray], Callable[[Any], Any]]] = [
    (
        "basic-strided-and-int-drop",
        lambda a: a.lazy[1:6:2, :, -1],
        lambda r: r[1:6:2, :, -1],
    ),
    ("basic-ellipsis", lambda a: a.lazy[..., -2], lambda r: r[..., -2]),
    ("basic-negative-scalar", lambda a: a.lazy[-3], lambda r: r[-3]),
    ("basic-empty", lambda a: a.lazy[:, 2:2, :], lambda r: r[:, 2:2, :]),
    ("basic-all-scalars", lambda a: a.lazy[-1, 0, 2], lambda r: r[-1, 0, 2]),
    (
        "oindex-unsorted-duplicates-multi-axis",
        lambda a: a.lazy.oindex[[4, 0, 0, 2], :, [3, 1]],
        lambda r: outer(r, ([4, 0, 0, 2], slice(None), [3, 1])),
    ),
    (
        "oindex-negative-and-slice",
        lambda a: a.lazy.oindex[:, [-1, 0], 1:4],
        lambda r: outer(r, (slice(None), [-1, 0], slice(1, 4))),
    ),
    (
        "oindex-boolean-axis",
        lambda a: a.lazy.oindex[np.array([True, False, True, False, False, False, True]), :, :],
        lambda r: outer(
            r,
            (np.array([True, False, True, False, False, False, True]), slice(None), slice(None)),
        ),
    ),
    (
        "vindex-coordinates",
        lambda a: a.lazy.vindex[np.array([0, 6, 3]), np.array([1, 4, 0]), np.array([2, 0, 1])],
        lambda r: r[np.array([0, 6, 3]), np.array([1, 4, 0]), np.array([2, 0, 1])],
    ),
    (
        "vindex-broadcast-pair",
        lambda a: a.lazy.vindex[np.array([[0], [6]]), np.array([1, 4]), np.array([2, 0])],
        lambda r: r[np.array([[0], [6]]), np.array([1, 4]), np.array([2, 0])],
    ),
    (
        "vindex-negative-coordinates",
        lambda a: a.lazy.vindex[np.array([-1, -7]), np.array([-2, 0]), np.array([0, -1])],
        lambda r: r[np.array([-1, -7]), np.array([-2, 0]), np.array([0, -1])],
    ),
    ("vindex-mask", lambda a: a.lazy.vindex[MASK], lambda r: r[MASK]),
    (
        "compose-basic-then-oindex",
        lambda a: a.lazy[1:6].lazy.oindex[[3, 0, 0], [4, 1], :],
        lambda r: outer(r[1:6], ([3, 0, 0], [4, 1], slice(None))),
    ),
    (
        "compose-oindex-then-basic-other-axis",
        lambda a: a.lazy.oindex[[4, 0, 2], :, :].lazy[:, 1:4, ::2],
        lambda r: outer(r, ([4, 0, 2], slice(None), slice(None)))[:, 1:4, ::2],
    ),
    (
        "compose-basic-then-basic",
        lambda a: a.lazy[2:, 1:].lazy[::2, -1],
        lambda r: r[2:, 1:][::2, -1],
    ),
    (
        "compose-basic-then-vindex",
        lambda a: a.lazy[1:6, :, 1:].lazy.vindex[
            np.array([0, 4]), np.array([2, 0]), np.array([1, 2])
        ],
        lambda r: r[1:6, :, 1:][np.array([0, 4]), np.array([2, 0]), np.array([1, 2])],
    ),
    # Scalar integers are basic indices in the positional dialect: they drop the
    # axis, in every mode, exactly as NumPy does.
    ("oindex-scalar-drops-axis", lambda a: a.lazy.oindex[0], lambda r: r[0]),
    (
        "oindex-scalar-with-arrays",
        lambda a: a.lazy.oindex[0, [1, 2], :],
        lambda r: outer(r, (0, [1, 2], slice(None))),
    ),
    (
        "oindex-scalar-middle-axis",
        lambda a: a.lazy.oindex[[3, 1], -1, :],
        lambda r: outer(r, ([3, 1], -1, slice(None))),
    ),
    ("oindex-all-scalars", lambda a: a.lazy.oindex[0, 1, 2], lambda r: r[0, 1, 2]),
    ("vindex-all-scalars", lambda a: a.lazy.vindex[0, 1, 2], lambda r: r[0, 1, 2]),
    (
        "vindex-scalar-with-arrays",
        lambda a: a.lazy.vindex[0, [1, 2], [3, 0]],
        lambda r: r[0, [1, 2], [3, 0]],
    ),
    (
        "vindex-scalar-on-middle-axis",
        lambda a: a.lazy.vindex[[1, 2], 0, [3, 0]],
        lambda r: r[[1, 2], 0, [3, 0]],
    ),
    # Scalar applied to a previously fancy-indexed axis, both orders.
    (
        "compose-oindex-then-scalar",
        lambda a: a.lazy.oindex[[3, 1], :, :].lazy[0],
        lambda r: outer(r, ([3, 1], slice(None), slice(None)))[0],
    ),
    (
        "compose-oindex-then-scalar-negative",
        lambda a: a.lazy.oindex[[3, 1, 1], :, :].lazy[-1, 2],
        lambda r: outer(r, ([3, 1, 1], slice(None), slice(None)))[-1, 2],
    ),
    (
        "compose-scalar-then-oindex",
        lambda a: a.lazy[0].lazy.oindex[[3, 1], :],
        lambda r: outer(r[0], ([3, 1], slice(None))),
    ),
    (
        "compose-vindex-then-scalar",
        lambda a: a.lazy.vindex[np.array([0, 6, 3]), np.array([1, 4, 0]), np.array([2, 0, 1])].lazy[
            1
        ],
        lambda r: r[np.array([0, 6, 3]), np.array([1, 4, 0]), np.array([2, 0, 1])][1],
    ),
    # Partial vindex whose coordinate arrays are NOT on the leading axes: NumPy
    # inserts the gathered axis where the (adjacent) advanced indices sat.
    (
        "vindex-trailing-arrays",
        lambda a: a.lazy.vindex[..., np.array([1, 4, 0]), np.array([2, 0, 1])],
        lambda r: r[..., np.array([1, 4, 0]), np.array([2, 0, 1])],
    ),
    (
        "vindex-single-trailing-array",
        lambda a: a.lazy.vindex[..., np.array([3, 0, 1])],
        lambda r: r[..., np.array([3, 0, 1])],
    ),
    (
        "vindex-trailing-mask",
        lambda a: a.lazy.vindex[..., TRAILING_MASK],
        lambda r: r[..., TRAILING_MASK],
    ),
    (
        "vindex-leading-partial",
        lambda a: a.lazy.vindex[np.array([1, 2, 2])],
        lambda r: r[np.array([1, 2, 2])],
    ),
    (
        "compose-basic-then-vindex-trailing",
        lambda a: a.lazy[2:, 1:].lazy.vindex[..., np.array([1, 3, 0])],
        lambda r: r[2:, 1:][..., np.array([1, 3, 0])],
    ),
    # A ConstantMap sitting between a slice and the coordinate arrays: NumPy
    # counts the integer as an advanced index, so the gathered axis moves to the
    # front of the chunk block even though the arrays are trailing.
    (
        "compose-vindex-trailing-then-scalar",
        lambda a: a.lazy.vindex[..., np.array([3, 0])].lazy[2],
        lambda r: r[..., np.array([3, 0])][2],
    ),
    # Two fancy axes, then a scalar on the first: the surviving ArrayMap ends up
    # behind a ConstantMap and a slice, which is where NumPy's integer-counts-as-
    # advanced rule bites.
    (
        "compose-oindex-two-axes-then-scalar",
        lambda a: a.lazy.oindex[[3, 1, 0], 3:5, [2, 0, 2]].lazy[0],
        lambda r: outer(r, ([3, 1, 0], slice(3, 5), [2, 0, 2]))[0],
    ),
    # Negative steps: the positional dialect is NumPy's, including the empty
    # cases NumPy allows where the transform algebra alone would object.
    ("reverse", lambda a: a.lazy[::-1], lambda r: r[::-1]),
    ("reverse-every-axis", lambda a: a.lazy[::-1, ::-1, ::-1], lambda r: r[::-1, ::-1, ::-1]),
    ("reverse-strided", lambda a: a.lazy[::-2], lambda r: r[::-2]),
    ("reverse-nondivisible", lambda a: a.lazy[::-3], lambda r: r[::-3]),
    ("reverse-bounded", lambda a: a.lazy[5:1:-1], lambda r: r[5:1:-1]),
    ("reverse-negative-start", lambda a: a.lazy[-1:None:-1], lambda r: r[-1:None:-1]),
    ("reverse-past-the-start", lambda a: a.lazy[:-8:-1], lambda r: r[:-8:-1]),
    ("reverse-empty", lambda a: a.lazy[2:2:-1], lambda r: r[2:2:-1]),
    # NumPy reads a reversed *positional* interval as empty; only the literal
    # layer calls it a direction error.
    ("reverse-inverted-is-empty", lambda a: a.lazy[2:5:-1], lambda r: r[2:5:-1]),
    ("reverse-with-int-drop", lambda a: a.lazy[::-2, 2, ::-1], lambda r: r[::-2, 2, ::-1]),
    ("reverse-trailing-axis", lambda a: a.lazy[..., ::-1], lambda r: r[..., ::-1]),
    (
        "compose-reverse-then-reverse",
        lambda a: a.lazy[::-1].lazy[::-1],
        lambda r: r[::-1][::-1],
    ),
    (
        "compose-strided-then-reverse",
        lambda a: a.lazy[::2].lazy[::-1],
        lambda r: r[::2][::-1],
    ),
    (
        "compose-reverse-then-strided",
        lambda a: a.lazy[::-1].lazy[::2],
        lambda r: r[::-1][::2],
    ),
    (
        "compose-reverse-then-oindex",
        lambda a: a.lazy[::-1].lazy.oindex[[3, 0, 0], :, :],
        lambda r: outer(r[::-1], ([3, 0, 0], slice(None), slice(None))),
    ),
    (
        "compose-oindex-then-reverse",
        lambda a: a.lazy.oindex[[3, 1, 2], :, :].lazy[::-1],
        lambda r: outer(r, ([3, 1, 2], slice(None), slice(None)))[::-1],
    ),
    (
        "compose-reverse-then-vindex",
        lambda a: a.lazy[::-1].lazy.vindex[..., np.array([1, 3, 0])],
        lambda r: r[::-1][..., np.array([1, 3, 0])],
    ),
    (
        "compose-vindex-trailing-then-scalar-and-slice",
        lambda a: a.lazy.vindex[..., np.array([3, 0, 1])].lazy[-1, 1:4],
        lambda r: r[..., np.array([3, 0, 1])][-1, 1:4],
    ),
]


@pytest.mark.parametrize(("build", "oracle"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_selection_matches_numpy(
    source: LazyArray,
    build: Callable[[LazyArray], LazyArray],
    oracle: Callable[[Any], Any],
) -> None:
    """Every selection resolves to what NumPy computes positionally on the same data."""
    view = build(source)
    expected = np.asarray(oracle(reference()))

    assert view.shape == expected.shape
    assert view.ndim == expected.ndim
    np.testing.assert_array_equal(np.asarray(view.result()), expected)
    # `__getitem__` is eager, and `__array__` routes through `result()`.
    np.testing.assert_array_equal(np.asarray(view), expected)


# ---------------------------------------------------------------------------
# Randomized chain sweep
# ---------------------------------------------------------------------------


def _random_basic(rng: np.random.Generator, shape: tuple[int, ...]) -> tuple[Any, ...]:
    selection: list[Any] = []
    for size in shape:
        roll = rng.random()
        if roll < 0.3:
            selection.append(int(rng.integers(-size, size)))
        elif roll < 0.55:
            start = int(rng.integers(0, size))
            stop = int(rng.integers(start, size + 1))
            selection.append(slice(start, stop, int(rng.integers(1, 4))))
        elif roll < 0.8:
            # Downward: `start >= stop` and the stop may fall off the front,
            # which is spelled `None`.
            start = int(rng.integers(0, size))
            stop_choice = int(rng.integers(-1, start + 1))
            stop = None if stop_choice < 0 else stop_choice
            selection.append(slice(start, stop, -int(rng.integers(1, 4))))
        else:
            selection.append(slice(None))
    return tuple(selection)


def _random_oindex(rng: np.random.Generator, shape: tuple[int, ...]) -> tuple[Any, ...]:
    selection: list[Any] = []
    for size in shape:
        roll = rng.random()
        if roll < 0.25:
            selection.append(int(rng.integers(-size, size)))
        elif roll < 0.65:
            count = int(rng.integers(1, 5))
            selection.append(rng.integers(-size, size, size=count).tolist())
        elif roll < 0.8:
            mask = rng.random(size) < 0.5
            mask[int(rng.integers(0, size))] = True
            selection.append(mask)
        else:
            start = int(rng.integers(0, size))
            selection.append(slice(start, size))
    return tuple(selection)


def _random_vindex(rng: np.random.Generator, shape: tuple[int, ...]) -> tuple[Any, ...]:
    ndim = len(shape)
    count = int(rng.integers(1, ndim + 1))
    trailing = bool(rng.random() < 0.5)
    axes = range(ndim - count, ndim) if trailing else range(count)
    sizes = [shape[axis] for axis in axes]

    entries: list[Any]
    if rng.random() < 0.2:
        # A single boolean mask spanning the whole covered block.
        mask = rng.random(tuple(sizes)) < 0.5
        mask.flat[int(rng.integers(0, mask.size))] = True
        entries = [mask]
    else:
        length = int(rng.integers(1, 5))
        entries = [
            int(rng.integers(-size, size))
            if rng.random() < 0.25
            else rng.integers(-size, size, size=length)
            for size in sizes
        ]
    return (Ellipsis, *entries) if trailing else tuple(entries)


def _apply_oracle(
    ref: np.ndarray[Any, Any], mode: str, selection: tuple[Any, ...]
) -> np.ndarray[Any, Any]:
    if mode == "orthogonal":
        return outer(ref, selection)
    # NumPy's own semantics *are* basic and vectorized indexing.
    return ref[selection]


def _apply_view(view: LazyArray, mode: str, selection: tuple[Any, ...]) -> LazyArray:
    if mode == "basic":
        return view.lazy[selection]
    if mode == "orthogonal":
        return view.lazy.oindex[selection]
    return view.lazy.vindex[selection]


def _random_chain(rng: np.random.Generator) -> list[tuple[str, tuple[Any, ...]]]:
    """A chain of 2-4 steps with at most one fancy step, in either order."""
    n_steps = int(rng.integers(2, 5))
    fancy_at = int(rng.integers(0, n_steps))
    fancy_mode = "orthogonal" if rng.random() < 0.5 else "vectorized"

    chain: list[tuple[str, tuple[Any, ...]]] = []
    running = reference()
    for step in range(n_steps):
        if running.ndim == 0 or running.size == 0:
            break
        mode = fancy_mode if step == fancy_at else "basic"
        if mode == "basic":
            selection = _random_basic(rng, running.shape)
        elif mode == "orthogonal":
            selection = _random_oindex(rng, running.shape)
        else:
            selection = _random_vindex(rng, running.shape)
        chain.append((mode, selection))
        running = _apply_oracle(running, mode, selection)
    return chain


@pytest.mark.parametrize("flavor", FLAVORS)
def test_random_chains_match_numpy(flavor: str) -> None:
    """A seeded sweep of composed chains, under every partitioning.

    The partitioning invariant as a property: `result()` is identical whatever
    boxes the read is broken into, including boxes deliberately misaligned with
    the source's own.
    """
    rng = np.random.default_rng(20260730)
    source = make_source(flavor)
    partitionings: list[Any] = [None, (2, 2, 2), (7, 5, 4), ((4, 3), (1, 3, 1), (3, 1))]
    # Reads against a real store cost more per chain; the NumPy flavors carry
    # the bulk of the sweep and exercise the identical code path.
    n_chains = 120 if flavor.startswith("zarr") else 400

    for _ in range(n_chains):
        chain = _random_chain(rng)
        expected = reference()
        for mode, selection in chain:
            expected = _apply_oracle(expected, mode, selection)

        view = source
        for mode, selection in chain:
            view = _apply_view(view, mode, selection)

        assert view.shape == expected.shape, f"{flavor}: {chain}"
        np.testing.assert_array_equal(
            np.asarray(view.result()), np.asarray(expected), err_msg=f"{flavor}: {chain}"
        )
        for parts in partitionings:
            np.testing.assert_array_equal(
                np.asarray(view.with_parts(parts).result()),
                np.asarray(expected),
                err_msg=f"{flavor} parts={parts}: {chain}",
            )


def test_eager_getitem_returns_data(source: LazyArray) -> None:
    """`arr[...]` reads immediately, like `numpy.ndarray.__getitem__`."""
    np.testing.assert_array_equal(np.asarray(source[1:3, ::2, -1]), reference()[1:3, ::2, -1])


# ---------------------------------------------------------------------------
# Attribute forwarding
# ---------------------------------------------------------------------------


def test_forwards_array_attributes(source: LazyArray) -> None:
    assert source.shape == SHAPE
    assert source.ndim == len(SHAPE)
    assert source.size == int(np.prod(SHAPE))
    assert source.dtype == np.dtype("int64")
    assert len(source) == SHAPE[0]
    assert "LazyArray" in repr(source)


def test_view_shape_comes_from_the_transform(source: LazyArray) -> None:
    """A view reports its own shape, not the wrapped array's."""
    view = source.lazy[1:6:2, :, -1]
    assert view.shape == (3, 5)
    assert view.ndim == 2
    assert view.size == 15
    assert view.dtype == source.dtype
    assert "view=" in repr(view)


def test_the_wrapper_has_no_chunks_vocabulary() -> None:
    """`chunks` is the *source's* word, never the wrapper's."""
    assert not hasattr(make_source("numpy-whole"), "chunks")
    assert not hasattr(make_source("zarr"), "chunks")
    with pytest.raises(TypeError):
        LazyArray(reference(), chunks=PART_SHAPE)  # type: ignore[call-arg]


def test_scalar_is_basic_even_when_a_slice_separates_it_from_the_arrays() -> None:
    """A documented, deliberate departure from one NumPy corner.

    NumPy counts an integer as an *advanced* index for its placement rule, so
    `x[0, :, i]` has shape `(len(i), x.shape[1])` — the arrays lead because the
    slice separates the integer from them. The positional dialect instead treats
    every scalar as a basic index applied first, which is the rule the rest of
    the surface follows and the only one `oindex` can express, so the same
    selection reads as `x[0][:, i]`.
    """
    data = reference()
    view = make_source("numpy-uniform-parts").lazy.vindex[0, ..., np.array([3, 0])]
    np.testing.assert_array_equal(np.asarray(view.result()), data[0][..., np.array([3, 0])])
    assert view.shape == data[0][..., np.array([3, 0])].shape
    assert view.shape != data[0, ..., np.array([3, 0])].shape


def test_zero_dimensional_result_is_an_array(source: LazyArray) -> None:
    """Both resolvers agree on the kind of a zero-rank result."""
    result = source.lazy[0, 1, 2].result()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 0
    assert result[()] == reference()[0, 1, 2]


class ForeignArray:
    """A minimal array-like whose advertised `chunks` we do not control."""

    def __init__(self, data: np.ndarray[Any, Any], chunks: Any) -> None:
        self._data = data
        self.chunks = chunks

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> Any:
        return self._data.dtype

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]


def test_malformed_discovered_parts_are_ignored() -> None:
    """A foreign object's unusable `chunks` falls back to one whole-array part.

    Discovery parses external input, so an attribute that does not describe a
    partitioning of the shape means "none I understand", not an error.
    """
    data = reference()
    for bogus in (((3, 3), (5,), (4,)), (3, 2), "nope", (3.5, 2, 2)):
        wrapped = LazyArray(ForeignArray(data, bogus))
        assert len(list(wrapped.parts())) == 1
        np.testing.assert_array_equal(np.asarray(wrapped.lazy[1:3].result()), data[1:3])


def test_discovered_parts_come_from_the_source_vocabulary() -> None:
    """`read_chunk_sizes` wins over `chunks`, and both are read as parts."""
    data = reference()
    wrapped = LazyArray(ForeignArray(data, PART_SHAPE))
    assert [part.base_coords for part in wrapped.parts()][:3] == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    assert len(list(wrapped.parts())) == 3 * 3 * 2


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

# (id, build, expected is_box, expected bounding_box). The bounds are storage
# intervals of the wrapped (7, 5, 4) array, computed by hand.
BOX_CASES: list[tuple[str, Callable[[LazyArray], LazyArray], bool, Any]] = [
    ("identity", lambda a: a, True, ((0, 7), (0, 5), (0, 4))),
    ("basic-slice", lambda a: a.lazy[1:6, :, 1:3], True, ((1, 6), (0, 5), (1, 3))),
    # Stride 2 over axis 1 touches 0, 2, 4; the hull is the closed span.
    ("strided", lambda a: a.lazy[::3, ::2, :], True, ((0, 7), (0, 5), (0, 4))),
    ("int-drop", lambda a: a.lazy[2, :, -1], True, ((2, 3), (0, 5), (3, 4))),
    ("all-scalars", lambda a: a.lazy[0, 1, 2], True, ((0, 1), (1, 2), (2, 3))),
    ("negative-and-open", lambda a: a.lazy[-2:], True, ((5, 7), (0, 5), (0, 4))),
    (
        "oindex",
        lambda a: a.lazy.oindex[[4, 0, 0], :, [3, 1]],
        False,
        ((0, 5), (0, 5), (1, 4)),
    ),
    (
        "vindex",
        lambda a: a.lazy.vindex[np.array([0, 6, 3]), np.array([1, 4, 0]), np.array([2, 0, 1])],
        False,
        ((0, 7), (0, 5), (0, 3)),
    ),
    ("mask", lambda a: a.lazy.vindex[MASK], False, ((0, 7), (0, 5), (0, 4))),
    # Composition preserves the category in both directions.
    ("box-of-box", lambda a: a.lazy[1:6].lazy[:, 1:3], True, ((1, 6), (1, 3), (0, 4))),
    (
        "box-after-fancy",
        lambda a: a.lazy.oindex[[4, 0, 2], :, :].lazy[0:2],
        False,
        ((0, 5), (0, 5), (0, 4)),
    ),
    ("empty", lambda a: a.lazy[2:2], True, None),
    (
        "empty-fancy",
        lambda a: a.lazy.oindex[np.array([], dtype=np.intp), :, :],
        False,
        None,
    ),
]


@pytest.mark.parametrize(
    ("build", "expected_is_box", "expected_bounds"),
    [case[1:] for case in BOX_CASES],
    ids=[case[0] for case in BOX_CASES],
)
def test_is_box_and_bounding_box(
    source: LazyArray,
    build: Callable[[LazyArray], LazyArray],
    expected_is_box: bool,
    expected_bounds: Any,
) -> None:
    """The box/query taxonomy, and the hull every selection has either way."""
    view = build(source)
    assert view.is_box is expected_is_box
    assert view.bounding_box() == expected_bounds


def test_a_unit_stride_box_is_dense_in_its_bounding_box() -> None:
    """Stride 1 everywhere: the hull is exactly what the view selects."""
    data = reference()
    view = make_source("numpy-uniform-parts").lazy[1:6, :, 1:3]
    assert view.is_box
    assert view.strides() == (1, 1, 1)
    bounds = view.bounding_box()
    assert bounds is not None
    np.testing.assert_array_equal(
        np.asarray(view.result()),
        data[tuple(slice(lo, hi) for lo, hi in bounds)],
    )


def test_a_strided_box_is_sparse_in_its_bounding_box() -> None:
    """Stride > 1: the hull is a superset, and `strides()` is what says by how much."""
    data = reference()
    view = make_source("numpy-uniform-parts").lazy[1:6, ::2, :]
    assert view.is_box
    assert view.strides() == (1, 2, 1)
    bounds = view.bounding_box()
    assert bounds is not None
    assert bounds == ((1, 6), (0, 5), (0, 4))

    hull = data[tuple(slice(lo, hi) for lo, hi in bounds)]
    assert hull.size == 5 * 5 * 4
    assert view.size == 5 * 3 * 4
    # A consumer that slabbed the hull and threw the rest away would over-read.
    assert hull.size > view.size
    # Applying the strides to the hull recovers the selection exactly.
    np.testing.assert_array_equal(
        np.asarray(view.result()),
        hull[tuple(slice(None, None, step) for step in view.strides() or ())],
    )


def test_strides_are_none_for_a_query() -> None:
    view = make_source("numpy-uniform-parts").lazy.oindex[[5, 1], :, :]
    assert not view.is_box
    assert view.strides() is None


def test_an_integer_indexed_dimension_has_stride_one() -> None:
    view = make_source("numpy-uniform-parts").lazy[2, ::3, :]
    assert view.strides() == (1, 3, 1)
    assert view.bounding_box() == ((2, 3), (0, 4), (0, 4))


def test_a_query_bounding_box_is_only_a_hull() -> None:
    """For a fancy selection the box is a superset, and `is_box` says so."""
    view = make_source("numpy-uniform-parts").lazy.oindex[[5, 1], :, :]
    assert not view.is_box
    assert view.bounding_box() == ((1, 6), (0, 5), (0, 4))
    # The hull spans 5 rows; the selection touches 2 of them.
    assert view.shape[0] == 2


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(
    "build",
    [
        lambda a: a,
        lambda a: a.lazy[1:6, :, 1:],
        lambda a: a.lazy.oindex[[4, 0, 0], :, [3, 1]],
        lambda a: a.lazy.vindex[..., np.array([1, 4, 0]), np.array([2, 0, 1])],
        # A reversing view drives the negative-stride branches of
        # `_intersect_dimension_map` and `iter_chunk_transforms`, which were
        # written defensively long before anything could reach them.
        lambda a: a.lazy[::-1, ::-2, :],
        lambda a: a.lazy[5:1:-1, :, ::-1],
    ],
    ids=["identity", "basic", "oindex", "vindex", "reversed", "reversed-bounded"],
)
def test_parts_tile_the_view_exactly_and_disjointly(
    flavor: str, build: Callable[[LazyArray], LazyArray]
) -> None:
    """Assembling the parts reproduces `result()`; the placements cover with no overlap."""
    view = build(make_source(flavor))
    expected = np.asarray(view.result())

    assembled = np.zeros(view.shape, dtype=view.dtype)
    hits = np.zeros(view.shape, dtype=np.int64)
    for part in view.parts():
        assembled[part.out_selection] = np.asarray(part.array.result())
        # `np.add.at` accumulates per element; `+= 1` on a fancy index would
        # count a repeated position once and hide a real overlap.
        np.add.at(hits, part.out_selection, 1)

    np.testing.assert_array_equal(assembled, expected)
    np.testing.assert_array_equal(hits, np.ones(view.shape, dtype=np.int64))


def test_parts_resolve_independently_and_concurrently() -> None:
    """Each part's `array` is a standalone `LazyArray` with no shared mutable state."""
    view = make_source("zarr").lazy[1:7, :, 1:].with_parts((2, 2, 2))
    parts = list(view.parts())
    assert len(parts) > 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda part: np.asarray(part.array.result()), parts))

    assembled = np.zeros(view.shape, dtype=view.dtype)
    for part, value in zip(parts, values, strict=True):
        assembled[part.out_selection] = value
    np.testing.assert_array_equal(assembled, np.asarray(view.result()))


def test_parts_report_completeness() -> None:
    """`is_complete` distinguishes a fully-covered box from a partial one."""
    array = make_source("numpy-uniform-parts")
    assert all(part.is_complete for part in array.parts())
    # Boxes along axis 2 are [0, 3) and [3, 4); dropping column 0 leaves the
    # first partially covered and the second whole.
    trimmed = {part.base_coords[2]: part.is_complete for part in array.lazy[:, :, 1:].parts()}
    assert trimmed == {0: False, 1: True}
    # A fancy axis is always reported incomplete.
    assert not any(part.is_complete for part in array.lazy.oindex[[4, 0, 0], :, :].parts())


def test_partition_boxes_are_global_and_tile_the_base() -> None:
    """`box` is in the wrapped array's coordinates, so parts are distinguishable.

    `array.bounding_box()` is part-local and can repeat across parts; `box` is
    what says which part of the base a `Partition` actually sits in.
    """
    view = make_source("numpy-uniform-parts").lazy[1:6, :, 1:]
    parts = {part.base_coords: part for part in view.parts()}

    # The reviewer's repro: two parts of the same view whose part-local
    # bounding boxes coincide but which cover different regions of the base.
    first, second = parts[0, 0, 0], parts[0, 1, 0]
    assert first.array.bounding_box() == second.array.bounding_box()
    assert first.box != second.box
    assert first.box == ((0, 3), (0, 2), (0, 3))
    assert second.box == ((0, 3), (2, 4), (0, 3))

    # Every box is the base partitioning's own box for those coordinates, and
    # the touched boxes tile the region the view reads without overlapping.
    grids = dimension_grids_from_chunks(PART_SHAPE, SHAPE)
    for coords, part in parts.items():
        expected = tuple(
            (grid.chunk_offset(c), grid.chunk_offset(c) + grid.chunk_size(c))
            for grid, c in zip(grids, coords, strict=True)
        )
        assert part.box == expected
    covered = np.zeros(SHAPE, dtype=np.int64)
    for part in parts.values():
        covered[tuple(slice(lo, hi) for lo, hi in part.box)] += 1
    assert covered.max() == 1
    # The view reads rows 1..5 and columns 1.., so every box it touches
    # intersects that region and no box outside it is visited.
    assert covered[1:6, :, 1:].sum() > 0
    assert covered[6:, :, :].sum() == 0


def test_partition_box_of_a_whole_array_part() -> None:
    part = next(iter(make_source("numpy-whole").parts()))
    assert part.box == tuple((0, extent) for extent in SHAPE)


def test_an_unpartitioned_wrapper_has_a_single_whole_array_part() -> None:
    view = make_source("numpy-whole")
    parts = list(view.parts())
    assert len(parts) == 1
    assert parts[0].base_coords == (0, 0, 0)
    assert parts[0].array.shape == SHAPE
    assert parts[0].is_complete
    np.testing.assert_array_equal(np.asarray(parts[0].array.result()), reference())


def test_with_parts_keeps_the_view_and_the_base() -> None:
    view = make_source("zarr").lazy[1:6, ::2]
    repartitioned = view.with_parts((2, 1, 4))
    assert repartitioned.shape == view.shape
    assert repartitioned.array is view.array
    assert repartitioned.transform == view.transform
    # A different partitioning really is a different set of boxes.
    assert [part.base_coords for part in repartitioned.parts()] != [
        part.base_coords for part in view.parts()
    ]
    np.testing.assert_array_equal(np.asarray(repartitioned.result()), np.asarray(view.result()))


def test_with_parts_none_forces_one_shot_resolution() -> None:
    view = make_source("zarr").lazy.oindex[[4, 0, 0], :, :]
    whole = view.with_parts(None)
    assert len(list(whole.parts())) == 1
    np.testing.assert_array_equal(np.asarray(whole.result()), np.asarray(view.result()))


@pytest.mark.parametrize(
    ("parts", "match"),
    [
        ((3,), "one entry per dimension"),
        ((3, (2, 2), 4), "not a mixture"),
        (((3, 3), (2, 2, 1), (3, 1)), "sum to 6, but the array extent is 7"),
        ((3, 0, 3), "chunk shape entries must be positive"),
        # A float or a None belongs to neither convention; saying "not a
        # mixture" would send the reader looking for the wrong mistake.
        ((3.5, 2, 2), r"3\.5 at dimension 0 is neither"),
        ((None, 5, 4), "None at dimension 0 is neither"),
        ((3.5, 2.5, 2.5), r"3\.5 at dimension 0, 2\.5 at dimension 1, .* are neither"),
        (((1.5, 5.5), (5,), (4,)), "per-axis chunk sizes must be integers; dimension 0"),
    ],
)
def test_with_parts_validates_strictly(parts: Any, match: str) -> None:
    """`with_parts` is our own API, so a malformed partitioning raises."""
    with pytest.raises(ValueError, match=match):
        make_source("numpy-whole").with_parts(parts)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def test_dask_token_is_deterministic_and_discriminating() -> None:
    """Same data, same view, same parts token alike; anything else differs."""
    data = reference()
    base = LazyArray(data)
    assert base.__dask_tokenize__() == LazyArray(reference()).__dask_tokenize__()

    tokens = {
        "base": base.__dask_tokenize__(),
        "view": base.lazy[1:3].__dask_tokenize__(),
        "other view": base.lazy[2:4].__dask_tokenize__(),
        "parts": base.with_parts((2, 2, 2)).__dask_tokenize__(),
        "other data": LazyArray(data + 1).__dask_tokenize__(),
    }
    assert len({repr(token) for token in tokens.values()}) == len(tokens)
    # Equivalent transforms reached different ways still token alike.
    assert base.lazy[1:5].lazy[0:2].__dask_tokenize__() == base.lazy[1:3].__dask_tokenize__()


def test_iteration_yields_eager_slices(source: LazyArray) -> None:
    rows = list(source.lazy[2:5])
    assert len(rows) == 3
    for row, expected in zip(rows, reference()[2:5], strict=True):
        np.testing.assert_array_equal(np.asarray(row), expected)


def test_iteration_over_a_zero_dimensional_view_is_rejected() -> None:
    with pytest.raises(TypeError, match="iteration over a 0-d array"):
        iter(make_source("numpy-uniform-parts").lazy[0, 0, 0])


def test_len_of_a_zero_dimensional_view_is_rejected() -> None:
    with pytest.raises(TypeError, match="len\\(\\) of unsized object"):
        len(make_source("numpy-uniform-parts").lazy[0, 0, 0])


@pytest.mark.parametrize(
    ("convert", "selection"),
    [
        (bool, (1, 1, 1)),
        (int, (1, 1, 1)),
        (float, (1, 1, 1)),
        (operator.index, (1, 1, 1)),
        (bool, (1, 1, slice(0, 1))),
    ],
    ids=["bool-0d", "int-0d", "float-0d", "index-0d", "bool-size-1"],
)
def test_scalar_conversions_match_numpy(convert: Any, selection: Any) -> None:
    """Size-1 conversions delegate to NumPy, values and all."""
    data = reference()
    view = make_source("numpy-uniform-parts").lazy[selection]
    assert convert(view) == convert(data[selection])


@pytest.mark.parametrize(
    ("convert", "selection", "error"),
    [
        (bool, (slice(0, 2), 0, 0), ValueError),
        (bool, (slice(0, 0), 0, 0), ValueError),
        (int, (slice(0, 1), 0, 0), TypeError),
        (float, (slice(0, 2), 0, 0), TypeError),
        (operator.index, (slice(0, 1), 0, 0), TypeError),
    ],
    ids=["bool-many", "bool-empty", "int-1d", "float-many", "index-1d"],
)
def test_scalar_conversions_raise_what_numpy_raises(
    convert: Any, selection: Any, error: type[Exception]
) -> None:
    data = reference()
    view = make_source("numpy-uniform-parts").lazy[selection]
    with pytest.raises(error):
        convert(view)
    with pytest.raises(error):
        convert(data[selection])


def test_pickle_round_trip() -> None:
    """A wrapper over a picklable base survives a round trip, view and parts intact."""
    view = LazyArray(reference()).with_parts((2, 2, 2)).lazy[1:6, ::2].lazy.oindex[[3, 0, 0], :, :]
    restored = pickle.loads(pickle.dumps(view))
    assert restored.shape == view.shape
    assert restored.__dask_tokenize__() == view.__dask_tokenize__()
    np.testing.assert_array_equal(np.asarray(restored.result()), np.asarray(view.result()))


# ---------------------------------------------------------------------------
# dask interop
# ---------------------------------------------------------------------------


def test_dask_from_array_roundtrip() -> None:
    """A `LazyArray` is a drop-in dask source — no translation ceremony."""
    da = pytest.importorskip("dask.array")
    source = make_source("zarr")

    lazy = da.from_array(source)
    np.testing.assert_array_equal(lazy.compute(), reference())

    # dask chooses its own blocks; the wrapper reads each of them through its
    # own parts, so the two partitionings need not agree.
    blocked = da.from_array(source, chunks=(4, 3, 3))
    assert blocked.chunks == ((4, 3), (3, 2), (3, 1))
    np.testing.assert_array_equal(blocked[2:, ::2].compute(), reference()[2:, ::2])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_boolean_scalar_is_rejected() -> None:
    with pytest.raises(IndexError, match="boolean scalars are not valid indices"):
        make_source("numpy-uniform-parts").lazy[True]


def test_mask_shape_must_match_the_view() -> None:
    array = make_source("numpy-uniform-parts")
    with pytest.raises(IndexError, match="boolean index has shape"):
        array.lazy.vindex[np.ones((2, 2, 2), dtype=bool)]


def test_scalar_index_out_of_bounds() -> None:
    with pytest.raises(IndexError, match="index 7 is out of bounds for axis 0 with size 7"):
        make_source("numpy-uniform-parts").lazy[7]


def test_scalar_index_out_of_bounds_in_a_view() -> None:
    """Bounds are the *view's*, not the wrapped array's."""
    array = make_source("numpy-uniform-parts").lazy[1:4]
    with pytest.raises(IndexError, match="index 3 is out of bounds for axis 0 with size 3"):
        array.lazy[3]


def test_index_array_out_of_bounds() -> None:
    array = make_source("numpy-uniform-parts")
    with pytest.raises(IndexError, match="index 99 is out of bounds for axis 0 with size 7"):
        array.lazy.oindex[[0, 99], :, :]


def test_too_many_indices() -> None:
    with pytest.raises(IndexError, match="too many indices"):
        make_source("numpy-uniform-parts").lazy[0, 0, 0, 0]


def test_copy_false_conversion_is_rejected() -> None:
    array = make_source("numpy-uniform-parts")
    with pytest.raises(ValueError, match="cannot be converted to a NumPy array without a copy"):
        np.array(array, copy=False)


# ---------------------------------------------------------------------------
# Negative steps
# ---------------------------------------------------------------------------


def test_reversed_box_reports_a_positive_stride(source: LazyArray) -> None:
    """A reversal is still a box; `strides()` is magnitudes, so it matches the forward twin."""
    reversed_view = source.lazy[::-2]
    forward = source.lazy[::2]
    assert reversed_view.is_box
    assert reversed_view.strides() == forward.strides() == (2, 1, 1)
    assert reversed_view.bounding_box() == ((0, 7), (0, 5), (0, 4))


def test_a_reversed_view_is_re_based_to_origin_zero() -> None:
    """The literal domain of a reversal is negative; the positional dialect hides it."""
    view = make_source("numpy-whole").lazy[::-1]
    # The algebra's own answer keeps the source frame.
    assert IndexTransform.from_shape(SHAPE)[::-1].domain.inclusive_min[0] == -6
    # The wrapper re-bases, so positions start at 0 as NumPy expects.
    assert view.transform.domain.inclusive_min == (0, 0, 0)
    assert view.shape == SHAPE
    np.testing.assert_array_equal(np.asarray(view.result()), reference()[::-1])


def test_zero_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="step cannot be zero"):
        make_source("numpy-whole").lazy[::0]


def test_reversed_positional_interval_is_empty_not_an_error() -> None:
    """NumPy's rule at the boundary; the literal layer keeps TensorStore's."""
    view = make_source("numpy-uniform-parts").lazy[2:5:-1]
    assert view.shape == (0, 5, 4)
    np.testing.assert_array_equal(np.asarray(view.result()), reference()[2:5:-1])
    # Literal coordinates, on the other hand, call it a direction error.
    with pytest.raises(IndexError, match="valid interval"):
        IndexTransform.from_shape(SHAPE)[2:5:-1]


def test_negative_step_over_a_fancy_axis_reverses_the_coordinates(source: LazyArray) -> None:
    """Reversing a gathered axis materializes, rather than attaching a stride."""
    view = source.lazy.oindex[[3, 1, 2], :, :].lazy[::-1]
    expected = outer(reference(), ([3, 1, 2], slice(None), slice(None)))[::-1]
    np.testing.assert_array_equal(np.asarray(view.result()), expected)
    m = view.transform.output[0]
    assert isinstance(m, ArrayMap)
    np.testing.assert_array_equal(m.index_array.reshape(-1), np.array([2, 1, 3]))
