import numpy as np
import pytest

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(store: Store, checksum: bool) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store, path="zstd"),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=ZstdCodec(level=0, checksum=checksum),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_zstd_codec_supports_sync() -> None:
    assert isinstance(ZstdCodec(), SupportsSyncCodec)


def test_zstd_is_not_fixed_size() -> None:
    assert ZstdCodec.is_fixed_size is False


def test_zstd_codec_sync_roundtrip() -> None:
    codec = ZstdCodec(level=1)
    arr = np.arange(100, dtype="float64")
    zdtype = get_data_type_from_native_dtype(arr.dtype)
    spec = ArraySpec(
        shape=arr.shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
    np.testing.assert_array_equal(arr, result)
