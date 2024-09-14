from collections.abc import Iterator

import numpy as np
import pytest

from zarr import Array
from zarr.store import MemoryStore, StorePath


@pytest.fixture
async def store() -> Iterator[StorePath]:
    yield StorePath(await MemoryStore.open(mode="w"))


def test_simple(store: StorePath) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "simple_v2",
        zarr_format=2,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
