from typing import Any

import numpy as np
import pytest

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
    assert expected == normalize_chunks_nd(chunks, shape)


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
    assert outer_chunks == expected_outer
    if expected_inner_outer is None:
        assert inner is None
    else:
        assert inner is not None
        assert inner.outer_chunks == expected_inner_outer
        assert inner.inner is None


def test_resolved_chunking_nested() -> None:
    """Test that ResolvedChunking supports recursive nesting for nested sharding."""
    from zarr.core.chunk_grids import ResolvedChunking

    leaf = normalize_chunks_nd((5, 5), (100, 100))
    mid = ResolvedChunking(
        outer_chunks=normalize_chunks_nd((25, 25), (100, 100)),
        inner=ResolvedChunking(outer_chunks=leaf),
    )
    top = ResolvedChunking(outer_chunks=normalize_chunks_nd((50, 50), (100, 100)), inner=mid)

    # Three levels: top -> mid -> leaf
    assert top.outer_chunks == ((50, 50), (50, 50))
    assert top.inner is not None
    assert top.inner.outer_chunks == ((25,) * 4, (25,) * 4)
    assert top.inner.inner is not None
    assert top.inner.inner.outer_chunks == ((5,) * 20, (5,) * 20)
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


def test_normalize_chunks_errors() -> None:
    with pytest.raises(ValueError, match="does not accept None"):
        normalize_chunks_nd(None, (100,))
    with pytest.raises(ValueError):
        normalize_chunks_nd("foo", (100,))
    with pytest.raises(ValueError, match="dimensions"):
        normalize_chunks_nd((100, 10), (100,))
    with pytest.raises(ValueError, match="dimensions"):
        normalize_chunks_nd((10,), (100, 100))
