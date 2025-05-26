from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.buffer import ArrayLike, BufferPrototype, NDArrayLike, cpu, gpu
from zarr.storage import MemoryStore, StorePath
from zarr.testing.buffer import (
    NDBufferUsingTestNDArrayLike,
    StoreExpectingTestBuffer,
    TestBuffer,
    TestNDArrayLike,
)
from zarr.testing.utils import gpu_test

if TYPE_CHECKING:
    import types

try:
    import cupy as cp
except ImportError:
    cp = None


import zarr.api.asynchronous

if TYPE_CHECKING:
    import types


def test_nd_array_like(xp: types.ModuleType) -> None:
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@pytest.mark.asyncio
async def test_async_array_prototype() -> None:
    """Test the use of a custom buffer prototype"""

    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await zarr.api.asynchronous.create_array(
        StorePath(StoreExpectingTestBuffer()) / "test_async_array_prototype",
        shape=expect.shape,
        chunks=(5, 5),
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
    # ignoring a mypy error here that TestNDArrayLike doesn't meet the NDArrayLike protocol
    # The test passes, so it clearly does.
    assert isinstance(got, TestNDArrayLike)
    assert np.array_equal(expect, got)  # type: ignore[unreachable]


@gpu_test
@pytest.mark.asyncio
async def test_async_array_gpu_prototype() -> None:
    """Test the use of the GPU buffer prototype"""

    expect = cp.zeros((9, 9), dtype="uint16", order="F")
    a = await zarr.api.asynchronous.create_array(
        StorePath(MemoryStore()) / "test_async_array_gpu_prototype",
        shape=expect.shape,
        chunks=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
    )
    expect[1:4, 3:6] = cp.ones((3, 3))

    await a.setitem(
        selection=(slice(1, 4), slice(3, 6)),
        value=cp.ones((3, 3)),
        prototype=gpu.buffer_prototype,
    )
    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), prototype=gpu.buffer_prototype)
    assert isinstance(got, cp.ndarray)
    assert cp.array_equal(expect, got)


@pytest.mark.asyncio
async def test_codecs_use_of_prototype() -> None:
    expect = np.zeros((10, 10), dtype="uint16", order="F")
    a = await zarr.api.asynchronous.create_array(
        StorePath(StoreExpectingTestBuffer()) / "test_codecs_use_of_prototype",
        shape=expect.shape,
        chunks=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
        compressors=[BloscCodec(), Crc32cCodec(), GzipCodec(), ZstdCodec()],
        filters=[TransposeCodec(order=(1, 0))],
    )
    expect[:] = np.arange(100).reshape(10, 10)

    my_prototype = BufferPrototype(buffer=TestBuffer, nd_buffer=NDBufferUsingTestNDArrayLike)

    await a.setitem(
        selection=(slice(0, 10), slice(0, 10)),
        value=expect[:],
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=my_prototype)
    # ignoring a mypy error here that TestNDArrayLike doesn't meet the NDArrayLike protocol
    # The test passes, so it clearly does.
    assert isinstance(got, TestNDArrayLike)
    assert np.array_equal(expect, got)  # type: ignore[unreachable]


@gpu_test
@pytest.mark.asyncio
async def test_codecs_use_of_gpu_prototype() -> None:
    expect = cp.zeros((10, 10), dtype="uint16", order="F")
    a = await zarr.api.asynchronous.create_array(
        StorePath(MemoryStore()) / "test_codecs_use_of_gpu_prototype",
        shape=expect.shape,
        chunks=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
        compressors=[BloscCodec(), Crc32cCodec(), GzipCodec(), ZstdCodec()],
        filters=[TransposeCodec(order=(1, 0))],
    )
    expect[:] = cp.arange(100).reshape(10, 10)

    await a.setitem(
        selection=(slice(0, 10), slice(0, 10)),
        value=expect[:],
        prototype=gpu.buffer_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=gpu.buffer_prototype)
    assert isinstance(got, cp.ndarray)
    assert cp.array_equal(expect, got)


@gpu_test
@pytest.mark.asyncio
async def test_sharding_use_of_gpu_prototype() -> None:
    with zarr.config.enable_gpu():
        expect = cp.zeros((10, 10), dtype="uint16", order="F")

        a = await zarr.api.asynchronous.create_array(
            StorePath(MemoryStore()) / "test_codecs_use_of_gpu_prototype",
            shape=expect.shape,
            chunks=(5, 5),
            shards=(10, 10),
            dtype=expect.dtype,
            fill_value=0,
        )
        expect[:] = cp.arange(100).reshape(10, 10)

        await a.setitem(
            selection=(slice(0, 10), slice(0, 10)),
            value=expect[:],
            prototype=gpu.buffer_prototype,
        )
        got = await a.getitem(
            selection=(slice(0, 10), slice(0, 10)), prototype=gpu.buffer_prototype
        )
        assert isinstance(got, cp.ndarray)
        assert cp.array_equal(expect, got)


def test_numpy_buffer_prototype() -> None:
    buffer = cpu.buffer_prototype.buffer.create_zero_length()
    ndbuffer = cpu.buffer_prototype.nd_buffer.create(shape=(1, 2), dtype=np.dtype("int64"))
    assert isinstance(buffer.as_array_like(), np.ndarray)
    assert isinstance(ndbuffer.as_ndarray_like(), np.ndarray)
    with pytest.raises(ValueError, match="Buffer does not contain a single scalar value"):
        ndbuffer.as_scalar()


@gpu_test
def test_gpu_buffer_prototype() -> None:
    buffer = gpu.buffer_prototype.buffer.create_zero_length()
    ndbuffer = gpu.buffer_prototype.nd_buffer.create(shape=(1, 2), dtype=cp.dtype("int64"))
    assert isinstance(buffer.as_array_like(), cp.ndarray)
    assert isinstance(ndbuffer.as_ndarray_like(), cp.ndarray)
    with pytest.raises(ValueError, match="Buffer does not contain a single scalar value"):
        ndbuffer.as_scalar()


# TODO: the same test for other buffer classes
def test_cpu_buffer_as_scalar() -> None:
    buf = cpu.buffer_prototype.nd_buffer.create(shape=(), dtype="int64")
    assert buf.as_scalar() == buf.as_ndarray_like()[()]  # type: ignore[index]
