import numpy as np
import pytest

import zarr
from zarr.abc.codec import SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.codecs.gzip import _gzip_streams_equal_except_mtime
from zarr.core.array_spec import ArrayConfig, ArraySpec
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
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    decoded = codec._decode_sync(encoded, spec)
    result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
    np.testing.assert_array_equal(arr, result)


def test_gzip_streams_equal_except_mtime() -> None:
    prefix = b"\x1f\x8b\x08\x00"
    mtime = b"\x01\x02\x03\x04"
    suffix = b"\x00\xff\x10\x20"

    # Identical streams are equal.
    assert _gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + suffix,
    )

    # Streams with different MTIME values are still equal.
    assert _gzip_streams_equal_except_mtime(
        prefix + b"\x01\x02\x03\x04" + suffix,
        prefix + b"\x05\x06\x07\x08" + suffix,
    )

    # Differences after MTIME are detected.
    assert not _gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + b"\x00\xff\x10\x21",
    )

    # Differences before MTIME are detected.
    assert not _gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        b"\x1f\x8b\x09\x00" + mtime + suffix,
    )

    # Different lengths are detected.
    assert not _gzip_streams_equal_except_mtime(
        prefix + mtime + suffix,
        prefix + mtime + suffix + b"\x00",
    )
