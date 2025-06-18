from typing import Literal

import numpy as np
import pytest

import zarr
from zarr.abc.store import Store
from zarr.codecs import BytesCodec
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy


@pytest.mark.filterwarnings("ignore:The endianness of the requested serializer")
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("endian", ["big", "little"])
async def test_endian(store: Store, endian: Literal["big", "little"]) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "."},
        serializer=BytesCodec(endian=endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)


@pytest.mark.filterwarnings("ignore:The endianness of the requested serializer")
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype_input_endian", [">u2", "<u2"])
@pytest.mark.parametrize("dtype_store_endian", ["big", "little"])
async def test_endian_write(
    store: Store,
    dtype_input_endian: Literal[">u2", "<u2"],
    dtype_store_endian: Literal["big", "little"],
) -> None:
    data = np.arange(0, 256, dtype=dtype_input_endian).reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype="uint16",
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "."},
        serializer=BytesCodec(endian=dtype_store_endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)
