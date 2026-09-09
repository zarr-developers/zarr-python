import contextlib
from typing import Any, Literal, cast

import numpy as np
import pytest

import zarr
from tests.conftest import Expect, ExpectFail
from zarr.core.chunk_grids import (
    ChunkGrid,
    ChunkLayout,
    FixedDimension,
    VaryingDimension,
    _guess_num_chunks_per_axis_shard,
    _guess_regular_chunks,
    normalize_chunks_1d,
    normalize_chunks_nd,
    resolve_outer_and_inner_chunks,
)
from zarr.errors import ZarrUserWarning


def _assert_chunks_equal(
    actual: ChunkGrid,
    expected: tuple[int | tuple[int, ...], ...],
    shape: tuple[int, ...],
) -> None:
    """Compare a normalized ChunkGrid against per-dimension expectations.

    An expected bare `int` requires a `FixedDimension` with that uniform size;
    an expected tuple requires a `VaryingDimension` with those exact edges.
    Every dimension must carry the corresponding extent from `shape`.
    """
    dims = actual.dimensions
    assert len(dims) == len(expected) == len(shape), (
        f"axis count mismatch: {len(dims)} vs {len(expected)} vs {len(shape)}"
    )
    for axis, (a, e, span) in enumerate(zip(dims, expected, shape, strict=True)):
        if isinstance(e, int):
            assert isinstance(a, FixedDimension), f"axis {axis}: expected FixedDimension, got {a!r}"
            assert a.size == e, f"axis {axis}: size {a.size} != {e}"
        else:
            assert isinstance(a, VaryingDimension), (
                f"axis {axis}: expected VaryingDimension, got {a!r}"
            )
            assert a.edges == tuple(e), f"axis {axis}: edges {a.edges} != {tuple(e)}"
        assert a.extent == span, f"axis {axis}: extent {a.extent} != {span}"


@pytest.mark.parametrize(
    "shape", [(0,), (0,) * 2, (1, 2, 0, 4, 5), (10, 0), (10,), (100,) * 3, (1000000,), (10000,) * 2]
)
@pytest.mark.parametrize("itemsize", [1, 2, 4])
def test_guess_chunks(shape: tuple[int, ...], itemsize: int) -> None:
    chunks = _guess_regular_chunks(shape, itemsize)
    chunk_size = np.prod(chunks) * itemsize
    assert isinstance(chunks, tuple)
    assert len(chunks) == len(shape)
    assert chunk_size < (64 * 1024 * 1024)
    # doesn't make any sense to allow chunks to have zero length dimension
    assert all(0 < c <= max(s, 1) for c, s in zip(chunks, shape, strict=False))


@pytest.mark.parametrize(
    ("chunks", "shape", "expected"),
    [
        # 1D cases (uniform sizes stay bare ints)
        ((10,), (100,), (10,)),
        ([10], (100,), (10,)),
        (10, (100,), (10,)),
        # 2D cases
        ((10, 10), (100, 10), (10, 10)),
        (10, (100, 10), (10, 10)),
        ((10, -1), (100, 10), (10, 10)),
        # 3D cases
        (30, (100, 20, 10), (30, 30, 30)),
        ((30, -1, -1), (100, 20, 10), (30, 20, 10)),
        ((30, 20, -1), (100, 20, 10), (30, 20, 10)),
        ((30, 20, 10), (100, 20, 10), (30, 20, 10)),
        # dask-style explicit lists always keep the per-chunk rectilinear form,
        # even when the sizes describe a regular grid (gh-4272)
        (((100, 100, 100), (50, 50)), (300, 100), ((100, 100, 100), (50, 50))),
        (((100, 100, 50),), (250,), ((100, 100, 50),)),
        (((100,),), (100,), ((100,),)),
        (((10, 20, 70), (50, 50)), (100, 100), ((10, 20, 70), (50, 50))),
        # no chunking (False means each dimension is one chunk spanning the full extent)
        (False, (100,), (100,)),
        (False, (100, 50), (100, 50)),
        # sentinel values
        (-1, (100,), (100,)),
        # False and -1 on a zero-length axis clamp to chunk size 1 (chunk sizes must be positive)
        (False, (0,), (1,)),
        (False, (0, 4), (1, 4)),
        (-1, (4, 0), (4, 1)),
        # zero-length dimensions preserve the declared chunk size
        (10, (0,), (10,)),
        ((5, 10), (0, 100), (5, 10)),
        ((5, 10), (20, 0), (5, 10)),
        # numpy integers are accepted anywhere a python int is, whether as the scalar
        # convenience form, as per-dimension entries, or as the `-1` sentinel.
        (np.int64(10), (100,), (10,)),
        ((np.int64(2), np.int64(2)), (4, 4), (2, 2)),
        ((1, 3, np.int64(16), np.int64(16)), (1, 3, 32, 32), (1, 3, 16, 16)),
        ((np.int32(30), np.int64(-1)), (100, 20), (30, 20)),
        (np.array([10, 10]), (100, 100), (10, 10)),
        # rectilinear chunks given as numpy arrays
        ((np.array([60, 40]), np.array([50, 50])), (100, 100), ((60, 40), (50, 50))),
    ],
)
def test_normalize_chunks(
    chunks: Any, shape: tuple[int, ...], expected: tuple[int | tuple[int, ...], ...]
) -> None:
    _assert_chunks_equal(normalize_chunks_nd(chunks, shape), expected, shape)


@pytest.mark.parametrize(
    ("array_shape", "chunks_input", "shard_shape", "expected_outer", "expected_inner_outer"),
    [
        # no sharding: outer = chunks, inner = None
        ((100,), (10,), None, (10,), None),
        # explicit regular shards
        ((100,), (10,), (50,), (50,), (10,)),
        # rectilinear shards keep the per-chunk form even when the sizes
        # describe a regular-with-boundary grid (gh-4272)
        ((100,), (10,), ((60, 40),), ((60, 40),), (10,)),
        ((100,), (10,), ((30, 60, 10),), ((30, 60, 10),), (10,)),
        # dict-style shards
        ((100, 100), (10, 10), {"shape": (50, 50)}, (50, 50), (10, 10)),
    ],
)
def test_resolve_outer_and_inner_chunks(
    array_shape: tuple[int, ...],
    chunks_input: tuple[int, ...],
    shard_shape: Any,
    expected_outer: tuple[int | tuple[int, ...], ...],
    expected_inner_outer: tuple[int | tuple[int, ...], ...] | None,
) -> None:
    chunks = normalize_chunks_nd(chunks_input, array_shape)
    outer_chunks, inner = resolve_outer_and_inner_chunks(
        array_shape=array_shape, chunks=chunks, shard_shape=shard_shape, item_size=1
    )
    _assert_chunks_equal(outer_chunks, expected_outer, array_shape)
    if expected_inner_outer is None:
        assert inner is None
    else:
        assert inner is not None
        _assert_chunks_equal(inner.outer_chunks, expected_inner_outer, array_shape)
        assert inner.inner is None


def test_chunk_layout_nested() -> None:
    """Test that ChunkLayout supports recursive nesting for nested sharding."""
    leaf = normalize_chunks_nd((5, 5), (100, 100))
    mid = ChunkLayout(
        outer_chunks=normalize_chunks_nd((25, 25), (100, 100)),
        inner=ChunkLayout(outer_chunks=leaf),
    )
    top = ChunkLayout(outer_chunks=normalize_chunks_nd((50, 50), (100, 100)), inner=mid)

    # Three levels: top -> mid -> leaf
    _assert_chunks_equal(top.outer_chunks, (50, 50), (100, 100))
    assert top.inner is not None
    _assert_chunks_equal(top.inner.outer_chunks, (25, 25), (100, 100))
    assert top.inner.inner is not None
    _assert_chunks_equal(top.inner.inner.outer_chunks, (5, 5), (100, 100))
    assert top.inner.inner.inner is None


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input=(0, 100),
            exception=ValueError,
            id="zero-uniform",
            msg="Chunk size must be positive",
        ),
        ExpectFail(
            input=(-2, 100),
            exception=ValueError,
            id="negative-uniform",
            msg="Chunk size must be positive",
        ),
        ExpectFail(
            input=(np.int64(0), 100),
            exception=ValueError,
            id="zero-uniform-numpy",
            msg="Chunk size must be positive",
        ),
        ExpectFail(input=([], 100), exception=ValueError, id="empty-list", msg="must not be empty"),
        # Scalars that are neither integers nor iterable name themselves in the error,
        # rather than surfacing an opaque "object is not iterable" from `list(chunks)`.
        ExpectFail(
            input=(2.5, 100),
            exception=TypeError,
            id="non-iterable-scalar",
            msg="must be an integer or an iterable of integers; got 2.5 of type float",
            escape=True,
        ),
        ExpectFail(
            input=([10, -1, 10], 100),
            exception=ValueError,
            id="negative-element",
            msg="must be positive",
        ),
        ExpectFail(
            input=([10, 0, 10], 20), exception=ValueError, id="zero-element", msg="must be positive"
        ),
        ExpectFail(
            input=([10, 20], 100), exception=ValueError, id="wrong-sum", msg="do not sum to span"
        ),
        # Nested/RLE form for a single dim is rejected with offending indices.
        ExpectFail(
            input=([[3, 3], 1], 7),
            exception=TypeError,
            id="rle-single-dim",
            msg="non-integer element(s) ([3, 3],) at indices (0,)",
            escape=True,
        ),
        # Multiple non-int elements: all offending indices reported.
        ExpectFail(
            input=([1, [2, 2], 1, [3]], 9),
            exception=TypeError,
            id="multiple-non-ints",
            msg="non-integer element(s) ([2, 2], [3]) at indices (1, 3)",
            escape=True,
        ),
        # Strings are non-integers and should be reported the same way.
        ExpectFail(
            input=([2, "3", 5], 10),
            exception=TypeError,
            id="string-element",
            msg="non-integer element(s) ('3',) at indices (1,)",
            escape=True,
        ),
    ],
    ids=lambda c: c.id,
)
def test_normalize_chunks_1d_errors(case: ExpectFail[tuple[Any, int]]) -> None:
    """Invalid 1D chunk specifications are rejected with informative error messages."""
    chunks, span = case.input
    with case.raises():
        normalize_chunks_1d(chunks, span=span)


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input=(None, (100,)),
            exception=ValueError,
            id="none",
            msg="None is not a valid chunk input",
        ),
        # `True` is rejected explicitly because bool is a subclass of int — without
        # this guard, `chunks=True` would silently produce size-1 chunks.
        ExpectFail(
            input=(True, (100,)),
            exception=ValueError,
            id="true",
            msg="True is not a valid chunk input",
        ),
        ExpectFail(input=("foo", (100,)), exception=ValueError, id="string", msg="dimensions"),
        ExpectFail(
            input=((100, 10), (100,)), exception=ValueError, id="too-many-dims", msg="dimensions"
        ),
        ExpectFail(
            input=((10,), (100, 100)), exception=ValueError, id="too-few-dims", msg="dimensions"
        ),
        # End-to-end: per-dim RLE surfaces through normalize_chunks_nd.
        ExpectFail(
            input=([[6, 4], [[3, 3], 1]], (10, 10)),
            exception=TypeError,
            id="rle-inner-dim",
            msg="non-integer element(s) ([3, 3],) at indices (0,)",
            escape=True,
        ),
    ],
    ids=lambda c: c.id,
)
def test_normalize_chunks_nd_errors(case: ExpectFail[tuple[Any, tuple[int, ...]]]) -> None:
    """Invalid N-D chunk specifications are rejected with informative error messages."""
    chunks, shape = case.input
    with case.raises():
        normalize_chunks_nd(chunks, shape)


@pytest.mark.parametrize(
    "case",
    [
        # uniform-chunks branch: O(1) size+extent record, never one entry per chunk.
        Expect(
            input=(1000, 100_000), output=FixedDimension(size=1000, extent=100_000), id="uniform"
        ),
        # uniform chunks on a span too large to expand per-chunk (creation-time
        # counterpart of the gh-4174 indexing fix).
        Expect(input=(1, 2**62), output=FixedDimension(size=1, extent=2**62), id="uniform-huge"),
        # -1 sentinel branch: one chunk covering the full span.
        Expect(input=(-1, 100), output=FixedDimension(size=100, extent=100), id="full-span"),
        # -1 on a zero-length span clamps to chunk size 1 (chunk sizes must be positive).
        Expect(input=(-1, 0), output=FixedDimension(size=1, extent=0), id="full-span-empty"),
        # zero-length span preserves the declared chunk size.
        Expect(input=(10, 0), output=FixedDimension(size=10, extent=0), id="uniform-zero-span"),
        # explicit lists always keep the per-chunk form, even when the sizes
        # describe a regular grid (gh-4272).
        Expect(
            input=([10, 10, 10], 30),
            output=VaryingDimension([10, 10, 10], extent=30),
            id="explicit-regular",
        ),
        Expect(
            input=([10, 10, 5], 25),
            output=VaryingDimension([10, 10, 5], extent=25),
            id="explicit-boundary",
        ),
        Expect(
            input=([10, 20, 70], 100),
            output=VaryingDimension([10, 20, 70], extent=100),
            id="explicit-irregular",
        ),
    ],
    ids=lambda c: c.id,
)
def test_normalize_chunks_1d(
    case: Expect[tuple[Any, int], FixedDimension | VaryingDimension],
) -> None:
    """Both output variants bind chunk sizes to the span: scalar specs become
    `FixedDimension` (O(1) regardless of chunk count), explicit per-chunk
    lists become `VaryingDimension`."""
    chunks, span = case.input
    assert normalize_chunks_1d(chunks, span) == case.output


@pytest.mark.parametrize(
    ("chunk_shape", "array_shape"),
    [((), ()), ((0,), (0,)), ((0, 0), (0, 0))],
    ids=["0d", "zero-1d", "zero-2d"],
)
def test_guess_num_chunks_per_axis_shard_degenerate(
    chunk_shape: tuple[int, ...], array_shape: tuple[int, ...]
) -> None:
    """Degenerate chunk shapes must return 1 instead of hanging the search loop.

    Regression test for https://github.com/zarr-developers/zarr-python/issues/4304.
    """
    assert (
        _guess_num_chunks_per_axis_shard(
            chunk_shape=chunk_shape,
            item_size=8,
            max_bytes=128 * 1024 * 1024,
            array_shape=array_shape,
        )
        == 1
    )


def test_create_0d_array_auto_shards_with_target_shard_size() -> None:
    """A 0-dimensional array with shards="auto" and a shard size budget must not hang.

    Regression test for https://github.com/zarr-developers/zarr-python/issues/4304.
    """
    with (
        zarr.config.set({"array.target_shard_size_bytes": 128 * 1024 * 1024}),
        pytest.warns(ZarrUserWarning, match="Automatic shard shape inference is experimental"),
    ):
        arr = zarr.create_array(store={}, shape=(), dtype="int64", shards="auto")
    assert arr.shards == ()


# -- Zero-length dimensions --
#
# One invariant: a chunk edge length is always >= 1, an extent may be 0. Every spelling
# that derives a chunk size from a span (-1, False, "auto", shards="auto") must agree on
# chunk size 1 for a zero-length axis, in both Zarr formats, with or without sharding.
# Historically each spelling clamped (or failed to clamp) on its own; see #4304, #4305,
# #4307, #4328 and, further back, #150, #241, #303, #972, #1977, #2434, #3711.

ZeroLengthChunkSpelling = Literal["minus-one", "false", "auto", "one", "ones", "rectilinear"]
ZeroLengthShards = Literal["auto", "auto-budget", "explicit"] | None

# Spellings whose chunk size is derived from the axis span rather than given explicitly.
_SPAN_DERIVED_SPELLINGS: frozenset[ZeroLengthChunkSpelling] = frozenset(
    {"minus-one", "false", "auto"}
)


def _zero_length_chunks_arg(spelling: ZeroLengthChunkSpelling, shape: tuple[int, ...]) -> Any:
    """Translate a chunk-spelling id into the `chunks=` argument for `shape`."""
    match spelling:
        case "minus-one":
            return -1
        case "false":
            return False
        case "auto":
            return "auto"
        case "one":
            return 1
        case "ones":
            return (1,) * len(shape)
        case "rectilinear":
            return [[2, 2]] * len(shape)


@pytest.mark.parametrize("spelling", ["minus-one", "false", "auto", "one", "ones", "rectilinear"])
@pytest.mark.parametrize(
    "shape",
    [(0,), (0, 4), (4, 0), (0, 0), ()],
    ids=["1d", "2d-lead", "2d-trail", "2d-both", "0d"],
)
@pytest.mark.parametrize(
    ("zarr_format", "shards"),
    [(2, None), (3, None), (3, "auto"), (3, "auto-budget"), (3, "explicit")],
    ids=["v2", "v3", "v3-auto-shards", "v3-auto-shards-budget", "v3-explicit-shards"],
)
def test_create_zero_length_array(
    spelling: ZeroLengthChunkSpelling,
    shape: tuple[int, ...],
    zarr_format: Literal[2, 3],
    shards: ZeroLengthShards,
) -> None:
    """Every chunk spelling produces a valid, usable grid on a zero-length axis.

    Span-derived spellings resolve to chunk size 1 on zero-length axes (and the full span
    elsewhere, for these small shapes); explicit spellings are stored verbatim. In every case
    the stored metadata matches `arr.chunks` / `arr.shards`, the array can grow along the
    empty axis, round-trip data, and shrink back to empty.
    """
    ndim = len(shape)
    if spelling == "rectilinear":
        if zarr_format == 2:
            pytest.skip("Zarr format 2 does not support rectilinear chunk grids")
        if shards is not None:
            pytest.skip("rectilinear chunks with sharding is not supported")
        if ndim == 0:
            pytest.skip("a 0-d array has no dimension to chunk rectilinearly")
    if shards == "explicit" and ndim == 0:
        pytest.skip("a 0-d array has no axis to shard explicitly")

    chunks = _zero_length_chunks_arg(spelling, shape)
    expected_chunks: tuple[int, ...] | None
    if spelling in _SPAN_DERIVED_SPELLINGS:
        expected_chunks = tuple(max(s, 1) for s in shape)
    elif spelling == "rectilinear":
        expected_chunks = None
    else:
        expected_chunks = (1,) * ndim

    shards_arg: Any
    expected_shards: tuple[int, ...] | None
    match shards:
        case None:
            shards_arg, expected_shards = None, None
        case "auto" | "auto-budget":
            # Axes this short never split, so the guessed shard equals the chunk.
            shards_arg, expected_shards = "auto", expected_chunks
        case "explicit":
            # A shard larger than the (zero) extent is fine: the axis has zero shards.
            shards_arg = tuple(2 if s == 0 else s for s in shape)
            expected_shards = shards_arg

    warns = (
        pytest.warns(ZarrUserWarning, match="Automatic shard shape inference is experimental")
        if shards_arg == "auto"
        else contextlib.nullcontext()
    )
    budget = 128 * 1024 * 1024 if shards == "auto-budget" else None
    # The rectilinear flag must stay set for the array's whole life, not just creation.
    with zarr.config.set(
        {"array.rectilinear_chunks": True, "array.target_shard_size_bytes": budget}
    ):
        with warns:
            arr = zarr.create_array(
                store={},
                shape=shape,
                dtype="int64",
                chunks=chunks,
                shards=shards_arg,
                zarr_format=zarr_format,
            )

        # In-memory view and stored metadata agree with the invariant.
        assert arr.shards == expected_shards
        meta = cast(dict[str, Any], arr.metadata.to_dict())
        if spelling == "rectilinear":
            grid = meta["chunk_grid"]
            assert grid["name"] == "rectilinear"
            # Stored verbatim on zero-length axes too, run-length encoded as [size, count].
            assert list(grid["configuration"]["chunk_shapes"]) == [[[2, 2]]] * ndim
            assert arr.write_chunk_sizes == tuple(() if s == 0 else (2, 2) for s in shape)
        else:
            assert arr.chunks == expected_chunks
            if zarr_format == 2:
                assert meta["chunks"] == expected_chunks
            else:
                stored = meta["chunk_grid"]["configuration"]["chunk_shape"]
                assert stored == (expected_chunks if expected_shards is None else expected_shards)
            assert all(c >= 1 for c in arr.chunks)

        # The array must remain usable.
        if ndim == 0:
            arr[...] = 7
            assert arr[...] == 7
            return
        axis = shape.index(0)
        grown = tuple(2 if i == axis else s for i, s in enumerate(shape))
        data = np.full(grown, 7, dtype="int64")
        arr.append(data, axis=axis)
        assert arr.shape == grown
        np.testing.assert_array_equal(arr[...], data)
        arr.resize(shape)
        assert arr.shape == shape
        assert np.asarray(arr[...]).shape == shape


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_create_zero_chunk_rejected(zarr_format: Literal[2, 3]) -> None:
    """An explicit chunk size of 0 is rejected up front, even for a zero-length axis."""
    with pytest.raises(ValueError, match="Chunk size must be positive or -1, got 0"):
        zarr.create_array(store={}, shape=(0,), chunks=(0,), dtype="int64", zarr_format=zarr_format)


def test_rectilinear_zero_extent_matches_resize() -> None:
    """Creating a rectilinear axis at length 0 equals resizing one down to 0.

    Both leave a `VaryingDimension` whose edges lie entirely beyond the extent, so the
    stored grids are identical and both grow into the same chunks on append.
    """
    with zarr.config.set({"array.rectilinear_chunks": True}):
        created = zarr.create_array(store={}, shape=(0,), chunks=[[2, 2]], dtype="int64")
        resized = zarr.create_array(store={}, shape=(4,), chunks=[[2, 2]], dtype="int64")
        resized.resize((0,))
        created_meta = cast(dict[str, Any], created.metadata.to_dict())
        resized_meta = cast(dict[str, Any], resized.metadata.to_dict())
        assert created_meta["chunk_grid"] == resized_meta["chunk_grid"]
        assert created_meta["shape"] == resized_meta["shape"] == (0,)

        created.append(np.arange(3, dtype="int64"))
        resized.append(np.arange(3, dtype="int64"))
        np.testing.assert_array_equal(created[...], np.arange(3))
        np.testing.assert_array_equal(resized[...], np.arange(3))
        assert created.write_chunk_sizes == resized.write_chunk_sizes == ((2, 1),)


def test_normalize_chunks_1d_zero_span_accepts_any_edges() -> None:
    """On a zero-length span the explicit edge list is stored verbatim."""
    dim = normalize_chunks_1d([3, 5], span=0)
    assert isinstance(dim, VaryingDimension)
    assert dim.edges == (3, 5)
    assert dim.extent == 0
    assert dim.nchunks == 0
    assert dim.resize(4) == VaryingDimension([3, 5], extent=4)


def test_normalize_chunks_1d_nonzero_span_still_requires_exact_sum() -> None:
    """Relaxing the sum rule for span 0 must not leak into positive spans."""
    with pytest.raises(ValueError, match="do not sum to span 1"):
        normalize_chunks_1d([3, 5], span=1)
