import re
from typing import Any

import numpy as np
import pytest

from tests.test_codecs.conftest import Expect, ExpectErr
from zarr.core.chunk_grids import (
    ChunkLayout,
    _guess_regular_chunks,
    normalize_chunks_1d,
    normalize_chunks_nd,
    resolve_outer_and_inner_chunks,
)


def _assert_chunks_equal(
    actual: tuple[Any, ...],
    expected: tuple[tuple[int, ...], ...],
) -> None:
    """Compare a ChunksTuple (tuple of np.int64 arrays) against a tuple of int tuples."""
    assert len(actual) == len(expected), f"axis count mismatch: {len(actual)} vs {len(expected)}"
    for axis, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert np.array_equal(a, np.asarray(e, dtype=np.int64)), (
            f"axis {axis}: {list(a)} != {list(e)}"
        )


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
        # 1D cases
        ((10,), (100,), ((10,) * 10,)),
        ([10], (100,), ((10,) * 10,)),
        (10, (100,), ((10,) * 10,)),
        # 2D cases
        ((10, 10), (100, 10), ((10,) * 10, (10,))),
        (10, (100, 10), ((10,) * 10, (10,))),
        ((10, -1), (100, 10), ((10,) * 10, (10,))),
        # 3D cases
        (30, (100, 20, 10), ((30, 30, 30, 30), (30,), (30,))),
        ((30, -1, -1), (100, 20, 10), ((30, 30, 30, 30), (20,), (10,))),
        ((30, 20, -1), (100, 20, 10), ((30, 30, 30, 30), (20,), (10,))),
        ((30, 20, 10), (100, 20, 10), ((30, 30, 30, 30), (20,), (10,))),
        # dask-style chunks (explicit per-chunk sizes)
        (((100, 100, 100), (50, 50)), (300, 100), ((100, 100, 100), (50, 50))),
        (((100, 100, 50),), (250,), ((100, 100, 50),)),
        (((100,),), (100,), ((100,),)),
        # no chunking (False means each dimension is one chunk spanning the full extent)
        (False, (100,), ((100,),)),
        (False, (100, 50), ((100,), (50,))),
        # sentinel values
        (-1, (100,), ((100,),)),
        # zero-length dimensions preserve the declared chunk size
        (10, (0,), ((10,),)),
        ((5, 10), (0, 100), ((5,), (10,) * 10)),
        ((5, 10), (20, 0), ((5, 5, 5, 5), (10,))),
    ],
)
def test_normalize_chunks(
    chunks: Any, shape: tuple[int, ...], expected: tuple[tuple[int, ...], ...]
) -> None:
    _assert_chunks_equal(normalize_chunks_nd(chunks, shape), expected)


@pytest.mark.parametrize(
    ("array_shape", "chunks_input", "shard_shape", "expected_outer", "expected_inner_outer"),
    [
        # no sharding: outer = chunks, inner = None
        ((100,), (10,), None, ((10,) * 10,), None),
        # explicit regular shards
        ((100,), (10,), (50,), ((50, 50),), ((10,) * 10,)),
        # rectilinear shards
        ((100,), (10,), ((60, 40),), ((60, 40),), ((10,) * 10,)),
        # dict-style shards
        ((100, 100), (10, 10), {"shape": (50, 50)}, ((50, 50), (50, 50)), ((10,) * 10, (10,) * 10)),
    ],
)
def test_resolve_outer_and_inner_chunks(
    array_shape: tuple[int, ...],
    chunks_input: tuple[int, ...],
    shard_shape: Any,
    expected_outer: tuple[tuple[int, ...], ...],
    expected_inner_outer: tuple[tuple[int, ...], ...] | None,
) -> None:
    chunks = normalize_chunks_nd(chunks_input, array_shape)
    outer_chunks, inner = resolve_outer_and_inner_chunks(
        array_shape=array_shape, chunks=chunks, shard_shape=shard_shape, item_size=1
    )
    _assert_chunks_equal(outer_chunks, expected_outer)
    if expected_inner_outer is None:
        assert inner is None
    else:
        assert inner is not None
        _assert_chunks_equal(inner.outer_chunks, expected_inner_outer)
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
    _assert_chunks_equal(top.outer_chunks, ((50, 50), (50, 50)))
    assert top.inner is not None
    _assert_chunks_equal(top.inner.outer_chunks, ((25,) * 4, (25,) * 4))
    assert top.inner.inner is not None
    _assert_chunks_equal(top.inner.inner.outer_chunks, ((5,) * 20, (5,) * 20))
    assert top.inner.inner.inner is None


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(input=(0, 100), msg="Chunk size must be positive", exception_cls=ValueError),
        ExpectErr(input=(-2, 100), msg="Chunk size must be positive", exception_cls=ValueError),
        ExpectErr(input=([], 100), msg="must not be empty", exception_cls=ValueError),
        ExpectErr(input=([10, -1, 10], 100), msg="must be positive", exception_cls=ValueError),
        ExpectErr(input=([10, 0, 10], 20), msg="must be positive", exception_cls=ValueError),
        ExpectErr(input=([10, 20], 100), msg="do not sum to span", exception_cls=ValueError),
        # Nested/RLE form for a single dim is rejected with offending indices.
        ExpectErr(
            input=([[3, 3], 1], 7),
            msg="non-integer element(s) ([3, 3],) at indices (0,)",
            exception_cls=TypeError,
        ),
        # Multiple non-int elements: all offending indices reported.
        ExpectErr(
            input=([1, [2, 2], 1, [3]], 9),
            msg="non-integer element(s) ([2, 2], [3]) at indices (1, 3)",
            exception_cls=TypeError,
        ),
        # Strings are non-integers and should be reported the same way.
        ExpectErr(
            input=([2, "3", 5], 10),
            msg="non-integer element(s) ('3',) at indices (1,)",
            exception_cls=TypeError,
        ),
    ],
    ids=[
        "zero-uniform",
        "negative-uniform",
        "empty-list",
        "negative-element",
        "zero-element",
        "wrong-sum",
        "rle-single-dim",
        "multiple-non-ints",
        "string-element",
    ],
)
def test_normalize_chunks_1d_errors(case: ExpectErr[tuple[Any, int]]) -> None:
    """Invalid 1D chunk specifications are rejected with informative error messages."""
    chunks, span = case.input
    with pytest.raises(case.exception_cls, match=re.escape(case.msg)):
        normalize_chunks_1d(chunks, span=span)


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(input=(None, (100,)), msg="does not accept None", exception_cls=ValueError),
        # `True` is rejected explicitly because bool is a subclass of int — without
        # this guard, `chunks=True` would silently produce size-1 chunks.
        ExpectErr(input=(True, (100,)), msg="does not accept True", exception_cls=ValueError),
        ExpectErr(input=("foo", (100,)), msg="dimensions", exception_cls=ValueError),
        ExpectErr(input=((100, 10), (100,)), msg="dimensions", exception_cls=ValueError),
        ExpectErr(input=((10,), (100, 100)), msg="dimensions", exception_cls=ValueError),
        # End-to-end: per-dim RLE surfaces through normalize_chunks_nd.
        ExpectErr(
            input=([[6, 4], [[3, 3], 1]], (10, 10)),
            msg="non-integer element(s) ([3, 3],) at indices (0,)",
            exception_cls=TypeError,
        ),
    ],
    ids=["none", "true", "string", "too-many-dims", "too-few-dims", "rle-inner-dim"],
)
def test_normalize_chunks_nd_errors(case: ExpectErr[tuple[Any, tuple[int, ...]]]) -> None:
    """Invalid N-D chunk specifications are rejected with informative error messages."""
    chunks, shape = case.input
    with pytest.raises(case.exception_cls, match=re.escape(case.msg)):
        normalize_chunks_nd(chunks, shape)


@pytest.mark.parametrize(
    "case",
    [
        # uniform-chunks branch: one int → broadcast across span via np.full.
        Expect(input=(1000, 100_000), expected=[1000] * 100),
        # explicit-per-chunk branch.
        Expect(input=([10, 20, 30, 40], 100), expected=[10, 20, 30, 40]),
        # -1 sentinel branch: one chunk covering the full span.
        Expect(input=(-1, 100), expected=[100]),
    ],
    ids=["uniform", "explicit-list", "full-span-sentinel"],
)
def test_normalize_chunks_1d_returns_int64_array(
    case: Expect[tuple[Any, int], list[int]],
) -> None:
    """Every branch of normalize_chunks_1d must produce a 1D int64 array."""
    chunks, span = case.input
    result = normalize_chunks_1d(chunks, span)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.ndim == 1
    assert result.tolist() == case.expected
