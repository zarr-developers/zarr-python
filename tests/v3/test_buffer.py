from __future__ import annotations

import numpy as np
import pytest

from zarr.array import AsyncArray
from zarr.buffer import ArrayLike, BufferPrototype, NDArrayLike, numpy_buffer_prototype
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.store.core import StorePath
from zarr.testing.buffer import (
    NDBufferUsingTestNDArrayLike,
    StoreExpectingTestBuffer,
    TestBuffer,
    TestNDArrayLike,
)


def test_nd_array_like(xp) -> None:
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@pytest.mark.asyncio
async def test_async_array_prototype() -> None:
    """Test the use of a custom buffer prototype"""

    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(StoreExpectingTestBuffer(mode="w")) / "test_async_array_prototype",
        shape=expect.shape,
        chunk_shape=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
    )
    expect[1:4, 3:6] = np.ones((3, 3))

    my_prototype = BufferPrototype(buffer=TestBuffer, nd_buffer=NDBufferUsingTestNDArrayLike)

    await a.setitem(
        selection=(slice(1, 4), slice(3, 6)),
        value=np.ones((3, 3)),
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), prototype=my_prototype)
    assert isinstance(got, TestNDArrayLike)
    assert np.array_equal(expect, got)


@pytest.mark.asyncio
async def test_codecs_use_of_prototype() -> None:
    expect = np.zeros((10, 10), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(StoreExpectingTestBuffer(mode="w")) / "test_codecs_use_of_prototype",
        shape=expect.shape,
        chunk_shape=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
        codecs=[
            TransposeCodec(order=(1, 0)),
            BytesCodec(),
            BloscCodec(),
            Crc32cCodec(),
            GzipCodec(),
            ZstdCodec(),
        ],
    )
    expect[:] = np.arange(100).reshape(10, 10)

    my_prototype = BufferPrototype(buffer=TestBuffer, nd_buffer=NDBufferUsingTestNDArrayLike)

    await a.setitem(
        selection=(slice(0, 10), slice(0, 10)),
        value=expect[:],
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=my_prototype)
    assert isinstance(got, TestNDArrayLike)
    assert np.array_equal(expect, got)


def test_numpy_buffer_prototype() -> None:
    buffer = numpy_buffer_prototype().buffer.create_zero_length()
    ndbuffer = numpy_buffer_prototype().nd_buffer.create(shape=(1, 2), dtype=np.dtype("int64"))
    assert isinstance(buffer.as_array_like(), np.ndarray)
    assert isinstance(ndbuffer.as_ndarray_like(), np.ndarray)
