"""Common chunk grid tests and utilities shared across implementations."""

from typing import Any

import numpy as np
import pytest

from zarr.core.chunk_grids import _guess_chunks, _normalize_chunks


@pytest.mark.parametrize(
    "shape", [(0,), (0,) * 2, (1, 2, 0, 4, 5), (10, 0), (10,), (100,) * 3, (1000000,), (10000,) * 2]
)
@pytest.mark.parametrize("itemsize", [1, 2, 4])
def test_guess_chunks(shape: tuple[int, ...], itemsize: int) -> None:
    """Test automatic chunk size guessing."""
    chunks = _guess_chunks(shape, itemsize)
    chunk_size = np.prod(chunks) * itemsize
    assert isinstance(chunks, tuple)
    assert len(chunks) == len(shape)
    assert chunk_size < (64 * 1024 * 1024)
    # doesn't make any sense to allow chunks to have zero length dimension
    assert all(0 < c <= max(s, 1) for c, s in zip(chunks, shape, strict=False))


@pytest.mark.parametrize(
    ("chunks", "shape", "typesize", "expected"),
    [
        ((10,), (100,), 1, (10,)),
        ([10], (100,), 1, (10,)),
        (10, (100,), 1, (10,)),
        ((10, 10), (100, 10), 1, (10, 10)),
        (10, (100, 10), 1, (10, 10)),
        ((10, None), (100, 10), 1, (10, 10)),
        (30, (100, 20, 10), 1, (30, 30, 30)),
        ((30,), (100, 20, 10), 1, (30, 20, 10)),
        ((30, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, None, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, 20, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, 20, 10), (100, 20, 10), 1, (30, 20, 10)),
        # auto chunking
        (None, (100,), 1, (100,)),
        (-1, (100,), 1, (100,)),
        ((30, -1, None), (100, 20, 10), 1, (30, 20, 10)),
    ],
)
def test_normalize_chunks(
    chunks: Any, shape: tuple[int, ...], typesize: int, expected: tuple[int, ...]
) -> None:
    """Test chunk normalization with various inputs."""
    assert expected == _normalize_chunks(chunks, shape, typesize)


def test_normalize_chunks_errors() -> None:
    """Test that normalize_chunks raises appropriate errors."""
    with pytest.raises(ValueError):
        _normalize_chunks("foo", (100,), 1)
    with pytest.raises(ValueError):
        _normalize_chunks((100, 10), (100,), 1)


def test_normalize_chunks_dask_style_regular() -> None:
    """Test dask-style chunks with regular (uniform) chunks."""
    # Dask-style with uniform chunks should work without warnings
    chunks = [[10, 10, 10], [20, 20, 20, 20, 20]]
    result = _normalize_chunks(chunks, (30, 100), 1)
    assert result == (10, 20)


def test_normalize_chunks_dask_style_irregular_warning() -> None:
    """Test that irregular dask-style chunks produce a warning."""
    # Irregular chunks: different sizes in same dimension
    chunks = [[10, 10, 5], [20, 20]]  # First dim has irregular chunks

    with pytest.warns(UserWarning, match="Irregular chunks detected in dimension 0"):
        result = _normalize_chunks(chunks, (25, 40), 1)

    # Should use first chunk size from each dimension
    assert result == (10, 20)


def test_normalize_chunks_dask_style_irregular_multiple_dims() -> None:
    """Test irregular chunks in multiple dimensions."""
    # Irregular in both dimensions
    chunks = [[10, 10, 5], [20, 15, 5]]

    # Should warn about both dimensions
    with pytest.warns(UserWarning, match="Irregular chunks detected") as record:
        result = _normalize_chunks(chunks, (25, 40), 1)

    # Should have warnings for both dimensions
    assert len(record) == 2
    assert "dimension 0" in str(record[0].message)
    assert "dimension 1" in str(record[1].message)

    # Should use first chunk size from each dimension
    assert result == (10, 20)
