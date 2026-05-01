from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr.abc.codec import ArrayBytesCodec, Codec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.core.codec_pipeline import ChunkTransform
from zarr.core.dtype import get_data_type_from_native_dtype


class AsyncOnlyCodec(ArrayBytesCodec):
    """A codec that only supports async, for testing rejection of non-sync codecs."""

    is_fixed_size = True

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError  # pragma: no cover

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError  # pragma: no cover

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length  # pragma: no cover


def _make_array_spec(shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(dtype)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _make_nd_buffer(arr: np.ndarray[Any, np.dtype[Any]]) -> NDBuffer:
    return default_buffer_prototype().nd_buffer.from_numpy_array(arr)


@pytest.mark.parametrize(
    ("shape", "codecs"),
    [
        ((100,), (BytesCodec(),)),
        ((100,), (BytesCodec(), GzipCodec())),
        ((3, 4), (TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec())),
    ],
    ids=["bytes-only", "with-compression", "full-chain"],
)
def test_construction(shape: tuple[int, ...], codecs: tuple[Codec, ...]) -> None:
    """Construction succeeds when all codecs implement SupportsSyncCodec."""
    _ = _make_array_spec(shape, np.dtype("float64"))
    ChunkTransform(codecs=codecs)


@pytest.mark.parametrize(
    ("shape", "codecs"),
    [
        ((100,), (AsyncOnlyCodec(),)),
        ((3, 4), (TransposeCodec(order=(1, 0)), AsyncOnlyCodec())),
    ],
    ids=["async-only", "mixed-sync-and-async"],
)
def test_construction_rejects_non_sync(shape: tuple[int, ...], codecs: tuple[Codec, ...]) -> None:
    """Construction raises TypeError when any codec lacks SupportsSyncCodec."""
    _ = _make_array_spec(shape, np.dtype("float64"))
    with pytest.raises(TypeError, match="AsyncOnlyCodec"):
        ChunkTransform(codecs=codecs)


@pytest.mark.parametrize(
    ("arr", "codecs"),
    [
        (np.arange(100, dtype="float64"), (BytesCodec(),)),
        (np.arange(100, dtype="float64"), (BytesCodec(), GzipCodec(level=1))),
        (
            np.arange(12, dtype="float64").reshape(3, 4),
            (TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)),
        ),
        (np.arange(100, dtype="float64"), (BytesCodec(), Crc32cCodec())),
        (np.arange(50, dtype="int32"), (BytesCodec(), ZstdCodec(level=1))),
    ],
    ids=["bytes-only", "gzip", "transpose+zstd", "crc32c", "int32"],
)
def test_encode_decode_roundtrip(
    arr: np.ndarray[Any, np.dtype[Any]], codecs: tuple[Codec, ...]
) -> None:
    """Data survives a full encode/decode cycle."""
    spec = _make_array_spec(arr.shape, arr.dtype)
    chain = ChunkTransform(codecs=codecs)
    nd_buf = _make_nd_buffer(arr)

    encoded = chain.encode_chunk(nd_buf, spec)
    assert encoded is not None
    decoded = chain.decode_chunk(encoded, spec)
    np.testing.assert_array_equal(arr, decoded.as_numpy_array())


@pytest.mark.parametrize(
    ("shape", "codecs", "input_size", "expected_size"),
    [
        ((100,), (BytesCodec(),), 800, 800),
        ((100,), (BytesCodec(), Crc32cCodec()), 800, 804),
        ((3, 4), (TransposeCodec(order=(1, 0)), BytesCodec()), 96, 96),
    ],
    ids=["bytes-only", "crc32c", "transpose"],
)
def test_compute_encoded_size(
    shape: tuple[int, ...],
    codecs: tuple[Codec, ...],
    input_size: int,
    expected_size: int,
) -> None:
    """compute_encoded_size returns the correct byte length."""
    spec = _make_array_spec(shape, np.dtype("float64"))
    chain = ChunkTransform(codecs=codecs)
    assert chain.compute_encoded_size(input_size, spec) == expected_size


def test_encode_returns_none_propagation() -> None:
    """When an AA codec returns None, encode short-circuits and returns None."""

    class NoneReturningAACodec(TransposeCodec):
        """An ArrayArrayCodec that always returns None from encode."""

        def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer | None:
            return None

    spec = _make_array_spec((3, 4), np.dtype("float64"))
    chain = ChunkTransform(
        codecs=(NoneReturningAACodec(order=(1, 0)), BytesCodec()),
    )
    arr = np.arange(12, dtype="float64").reshape(3, 4)
    nd_buf = _make_nd_buffer(arr)
    assert chain.encode_chunk(nd_buf, spec) is None
