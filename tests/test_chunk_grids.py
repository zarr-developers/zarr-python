from typing import Any

import numpy as np
import pytest

from zarr.core.chunk_grids import _guess_chunks, normalize_chunks


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
