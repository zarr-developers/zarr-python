from typing import Any

import numpy as np
import pytest

from tests.conftest import Expect, ExpectFail
from zarr.core.chunk_grids import (
    ChunkGrid,
    ChunkLayout,
    FixedDimension,
    VaryingDimension,
    _guess_regular_chunks,
    normalize_chunks_1d,
    normalize_chunks_nd,
    resolve_outer_and_inner_chunks,
)


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
        # dask-style chunks describing a regular grid collapse to the uniform form
        (((100, 100, 100), (50, 50)), (300, 100), (100, 50)),
        (((100, 100, 50),), (250,), (100,)),
        (((100,),), (100,), (100,)),
        # genuinely irregular explicit chunks keep the per-chunk form
        (((10, 20, 70), (50, 50)), (100, 100), ((10, 20, 70), 50)),
        # no chunking (False means each dimension is one chunk spanning the full extent)
        (False, (100,), (100,)),
        (False, (100, 50), (100, 50)),
        # sentinel values
        (-1, (100,), (100,)),
        # zero-length dimensions preserve the declared chunk size
        (10, (0,), (10,)),
        ((5, 10), (0, 100), (5, 10)),
        ((5, 10), (20, 0), (5, 10)),
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
        # rectilinear shards describing a regular-with-boundary grid collapse
        ((100,), (10,), ((60, 40),), (60,), (10,)),
        # genuinely irregular rectilinear shards keep the per-chunk form
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
        ExpectFail(input=([], 100), exception=ValueError, id="empty-list", msg="must not be empty"),
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
        # zero-length span preserves the declared chunk size.
        Expect(input=(10, 0), output=FixedDimension(size=10, extent=0), id="uniform-zero-span"),
        # explicit lists that describe a regular grid collapse to the uniform form.
        Expect(
            input=([10, 10, 10], 30),
            output=FixedDimension(size=10, extent=30),
            id="explicit-regular",
        ),
        Expect(
            input=([10, 10, 5], 25),
            output=FixedDimension(size=10, extent=25),
            id="explicit-boundary",
        ),
        # genuinely irregular edges keep the explicit per-chunk form.
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
    """Both output variants bind chunk sizes to the span: uniform specs become
    `FixedDimension` (O(1) regardless of chunk count), irregular explicit
    lists become `VaryingDimension`."""
    chunks, span = case.input
    assert normalize_chunks_1d(chunks, span) == case.output
