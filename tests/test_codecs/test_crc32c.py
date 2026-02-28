from __future__ import annotations

import numpy as np

from zarr.abc.codec import SupportsSyncCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import get_data_type_from_native_dtype


def test_crc32c_codec_supports_sync() -> None:
    assert isinstance(Crc32cCodec(), SupportsSyncCodec)


def test_crc32c_codec_sync_roundtrip() -> None:
    codec = Crc32cCodec()
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
