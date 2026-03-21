from typing import Any, cast

import numpy as np
import pytest

from zarr.core.chunk_grids import RegularChunkGrid, _guess_chunks, normalize_chunks


@pytest.mark.parametrize(
    "shape", [(0,), (0,) * 2, (1, 2, 0, 4, 5), (10, 0), (10,), (100,) * 3, (1000000,), (10000,) * 2]
)
@pytest.mark.parametrize("itemsize", [1, 2, 4])
def test_guess_chunks(shape: tuple[int, ...], itemsize: int) -> None:
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
        # dask-style chunks (uniform with optional smaller final chunk)
        (((100, 100, 100), (50, 50)), (300, 100), 1, (100, 50)),
        (((100, 100, 50),), (250,), 1, (100,)),
        (((100,),), (100,), 1, (100,)),
        # auto chunking
        (None, (100,), 1, (100,)),
        (-1, (100,), 1, (100,)),
        ((30, -1, None), (100, 20, 10), 1, (30, 20, 10)),
    ],
)
def test_normalize_chunks(
    chunks: Any, shape: tuple[int, ...], typesize: int, expected: tuple[int, ...]
) -> None:
    assert expected == normalize_chunks(chunks, shape, typesize)


def test_normalize_chunks_errors() -> None:
    with pytest.raises(ValueError):
        normalize_chunks("foo", (100,), 1)
    with pytest.raises(ValueError):
        normalize_chunks((100, 10), (100,), 1)
    # dask-style irregular chunks should raise
    with pytest.raises(ValueError, match="Irregular chunk sizes"):
        normalize_chunks(((10, 20, 30),), (60,), 1)
    with pytest.raises(ValueError, match="Irregular chunk sizes"):
        normalize_chunks(((100, 100), (10, 20)), (200, 30), 1)


def test_chunk_grid_array_shape_tracking() -> None:
    """Test that ChunkGrid can track the array shape it's associated with."""
    chunk_shape = (10, 20)
    array_shape = (100, 200)

    # Create a ChunkGrid without array_shape
    chunk_grid = RegularChunkGrid(chunk_shape=chunk_shape)
    assert chunk_grid.array_shape is None

    # Create a ChunkGrid with array_shape
    chunk_grid_with_shape = RegularChunkGrid(chunk_shape=chunk_shape, array_shape=array_shape)
    assert chunk_grid_with_shape.array_shape == array_shape


def test_chunk_grid_with_array_shape_method() -> None:
    """Test the with_array_shape method for setting array shape on existing ChunkGrid."""
    chunk_shape = (10, 20)
    array_shape = (100, 200)

    # Create initial ChunkGrid without shape
    chunk_grid = RegularChunkGrid(chunk_shape=chunk_shape)
    assert chunk_grid.array_shape is None

    # Use with_array_shape to get a new ChunkGrid with shape set
    chunk_grid_with_shape = chunk_grid.with_array_shape(array_shape)
    assert chunk_grid_with_shape.array_shape == array_shape
    assert chunk_grid_with_shape.chunk_shape == chunk_shape

    # Original should still be None
    assert chunk_grid.array_shape is None


def test_chunk_grid_array_shape_serialization() -> None:
    """Test that array_shape is not included in serialization (backward compatibility)."""
    chunk_shape = (10, 20)
    array_shape = (100, 200)

    # Create a ChunkGrid with array_shape
    chunk_grid = RegularChunkGrid(chunk_shape=chunk_shape, array_shape=array_shape)

    # Serialize to dict
    chunk_dict = chunk_grid.to_dict()

    # Verify that array_shape is not in the serialized form
    assert "array_shape" not in chunk_dict
    # Verify that chunk_shape is still there
    config = cast(dict[str, Any], chunk_dict.get("configuration"))
    assert config["chunk_shape"] == chunk_shape
