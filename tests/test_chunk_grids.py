import re
from typing import Any

import numpy as np
import pytest

from tests.test_codecs.conftest import ExpectErr
from zarr.core.chunk_grids import (
    _guess_regular_chunks,
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
    from zarr.core.chunk_grids import ChunkLayout

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


def test_normalize_chunks_1d_errors() -> None:
    from zarr.core.chunk_grids import normalize_chunks_1d

    with pytest.raises(ValueError, match="Chunk size must be positive"):
        normalize_chunks_1d(0, 100)
    with pytest.raises(ValueError, match="Chunk size must be positive"):
        normalize_chunks_1d(-2, 100)
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_chunks_1d([], 100)
    with pytest.raises(ValueError, match="must be positive"):
        normalize_chunks_1d([10, -1, 10], 100)
    with pytest.raises(ValueError, match="do not sum to span"):
        normalize_chunks_1d([10, 20], 100)


@pytest.mark.parametrize(
    "case",
    [
        # The motivating case: nested/RLE form for a single dim.
        ExpectErr(
            input=([[3, 3], 1], 7),
            msg="non-integer element(s) ([3, 3],) at indices (0,)",
            exception_cls=TypeError,
        ),
        # Multiple non-int elements report all offending indices.
        ExpectErr(
            input=([1, [2, 2], 1, [3]], 9),
            msg="non-integer element(s) ([2, 2], [3]) at indices (1, 3)",
            exception_cls=TypeError,
        ),
        # Strings are also non-integers and should be reported the same way.
        ExpectErr(
            input=([2, "3", 5], 10),
            msg="non-integer element(s) ('3',) at indices (1,)",
            exception_cls=TypeError,
        ),
    ],
    ids=["rle-single-dim", "multiple-non-ints", "string-element"],
)
def test_normalize_chunks_1d_rejects_non_int_elements(
    case: ExpectErr[tuple[list[Any], int]],
) -> None:
    """Reject nested/RLE-style chunk specs with a precise error pointing at offending indices."""
    from zarr.core.chunk_grids import normalize_chunks_1d

    chunks, span = case.input
    with pytest.raises(case.exception_cls, match=re.escape(case.msg)):
        normalize_chunks_1d(chunks, span=span)


def test_normalize_chunks_nd_rejects_rle_inner_dim() -> None:
    """End-to-end: a per-dim RLE form like [[3, 3], 1] surfaces the precise error."""
    with pytest.raises(
        TypeError, match=re.escape("non-integer element(s) ([3, 3],) at indices (0,)")
    ):
        normalize_chunks_nd([[6, 4], [[3, 3], 1]], (10, 10))


def test_normalize_chunks_errors() -> None:
    with pytest.raises(ValueError, match="does not accept None"):
        normalize_chunks_nd(None, (100,))
    with pytest.raises(ValueError):
        normalize_chunks_nd("foo", (100,))
    with pytest.raises(ValueError, match="dimensions"):
        normalize_chunks_nd((100, 10), (100,))
    with pytest.raises(ValueError, match="dimensions"):
        normalize_chunks_nd((10,), (100, 100))


def test_normalize_chunks_1d_uniform_returns_int64_array() -> None:
    """The uniform-chunks branch must return a 1D int64 array — this is the
    representation that enables O(1) construction via np.full."""
    from zarr.core.chunk_grids import normalize_chunks_1d

    result = normalize_chunks_1d(1000, 100_000)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.ndim == 1
    assert result.shape == (100,)
    assert (result == 1000).all()


def test_normalize_chunks_1d_explicit_list_returns_int64_array() -> None:
    """The explicit-per-chunk branch must also produce an int64 array."""
    from zarr.core.chunk_grids import normalize_chunks_1d

    result = normalize_chunks_1d([10, 20, 30, 40], 100)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.tolist() == [10, 20, 30, 40]


def test_normalize_chunks_1d_full_span_returns_int64_array() -> None:
    """The -1 sentinel branch must also produce an int64 array."""
    from zarr.core.chunk_grids import normalize_chunks_1d

    result = normalize_chunks_1d(-1, 100)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.tolist() == [100]
