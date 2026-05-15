from typing import Literal

import numpy as np
import pytest

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import BytesCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
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


def test_bytes_codec_supports_sync() -> None:
    assert isinstance(BytesCodec(), SupportsSyncCodec)


def test_bytes_codec_sync_roundtrip() -> None:
    codec = BytesCodec()
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    nd_buf: NDBuffer = default_buffer_prototype().nd_buffer.from_numpy_array(arr)

    codec = codec.evolve_from_array_spec(spec)

    encoded = codec._encode_sync(nd_buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    np.testing.assert_array_equal(arr, decoded.as_numpy_array())


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
