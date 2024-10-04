import json
from collections.abc import Iterator

import numpy as np
import pytest
from numcodecs import Delta
from numcodecs.blosc import Blosc

import zarr
import zarr.core.buffer.cpu
import zarr.core.metadata
import zarr.storage
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


@pytest.mark.parametrize("dtype", ["|S", "|V"])
async def test_v2_encode_decode(dtype):
    store = zarr.storage.MemoryStore(mode="w")
    g = zarr.group(store=store, zarr_format=2)
    g.create_array(
        name="foo",
        shape=(3,),
        chunks=(3,),
        dtype=dtype,
        fill_value=b"X",
    )

    result = await store.get("foo/.zarray", zarr.core.buffer.default_buffer_prototype())
    assert result is not None

    serialized = json.loads(result.to_bytes())
    expected = {
        "chunks": [3],
        "compressor": None,
        "dtype": f"{dtype}0",
        "fill_value": "WA==",
        "filters": None,
        "order": "C",
        "shape": [3],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    assert serialized == expected

    data = zarr.open_array(store=store, path="foo")[:]
    expected = np.full((3,), b"X", dtype=dtype)
    np.testing.assert_equal(data, expected)
