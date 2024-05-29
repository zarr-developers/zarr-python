from collections.abc import Iterator

import numpy as np
import pytest

from zarr.abc.store import Store
from zarr.array import Array
from zarr.store import MemoryStore, StorePath


@pytest.fixture
def store() -> Iterator[Store]:
    yield StorePath(MemoryStore(mode="w"))


def test_simple(store: Store):
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
