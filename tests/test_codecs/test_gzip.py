import numpy as np
import pytest

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.core.array_spec import ArraySpec, ArraySpecConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=GzipCodec(),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_gzip_codec_supports_sync() -> None:
    assert isinstance(GzipCodec(), SupportsSyncCodec)


def test_gzip_codec_sync_roundtrip() -> None:
    codec = GzipCodec(level=1)
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArraySpecConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
    np.testing.assert_array_equal(arr, result)
