from collections.abc import Iterator

import numpy as np
import pytest
from numcodecs import Delta
from numcodecs.blosc import Blosc

import zarr
from zarr import Array
from zarr.storage import MemoryStore, StorePath


@pytest.fixture
async def store() -> Iterator[StorePath]:
    return StorePath(await MemoryStore.open(mode="w"))


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


def test_implicit_fill_value(store: StorePath) -> None:
    arr = zarr.open_array(store=store, shape=(4,), fill_value=None, zarr_format=2)
    assert arr.metadata.fill_value is None
    assert arr.metadata.to_dict()["fill_value"] is None
    result = arr[:]
    expected = np.zeros(arr.shape, dtype=arr.dtype)
    np.testing.assert_array_equal(result, expected)


def test_codec_pipeline() -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2243
    store = MemoryStore(mode="w")
    array = zarr.create(
        store=store,
        shape=(1,),
        dtype="i4",
        zarr_format=2,
        filters=[Delta(dtype="i4").get_config()],
        compressor=Blosc().get_config(),
    )
    array[:] = 1
    result = array[:]
    expected = np.ones(1)
    np.testing.assert_array_equal(result, expected)
