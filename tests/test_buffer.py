from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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
from zarr.core.buffer.core import ScalarWrapper

if TYPE_CHECKING:
    import types


def test_nd_array_like(xp: types.ModuleType) -> None:
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@pytest.mark.parametrize("value", [1, 1.4, "a", b"a", np.array(1), False, True])
def test_scalar_wrapper(value: Any) -> None:
    x = ScalarWrapper(value)
    assert x == value
    assert value == x
    assert x == x[()]
    assert x.view(str) == x
    assert x.copy() == x
    assert x.transpose() == x
    assert x.ravel() == x
    assert x.all() == bool(value)
    if isinstance(value, (int, float)):
        assert -x == -value
        assert abs(x) == abs(value)
        assert int(x) == int(value)
        assert float(x) == float(value)
        assert complex(x) == complex(value)
        assert x + 1 == value + 1
        assert x - 1 == value - 1
        assert x * 2 == value * 2
        assert x / 2 == value / 2
        assert x // 2 == value // 2
        assert x % 2 == value % 2
        assert x**2 == value**2
        assert x == value
        assert x != value + 1
        assert bool(x) == bool(value)
        assert hash(x) == hash(value)
        assert str(x) == str(value)
        assert format(x, "") == format(value, "")
        x.fill(2)
        x[()] += 1
        assert x == 3
    elif isinstance(value, str):
        assert str(x) == value
        with pytest.raises(TypeError, match=re.escape("bad operand type for abs(): 'str'")):
            abs(x)

    with pytest.raises(ValueError, match="Cannot reshape scalar to non-scalar shape."):
        x.reshape((1, 2))
    with pytest.raises(IndexError, match="Invalid index for scalar."):
        x[10] = value
    with pytest.raises(IndexError, match="Invalid index for scalar."):
        x[10]
    with pytest.raises(TypeError, match=re.escape("len() of unsized object.")):
        len(x)


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
    assert isinstance(got, TestNDArrayLike)  # type: ignore[unreachable]
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
    assert isinstance(got, TestNDArrayLike)  # type: ignore[unreachable]
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


def test_numpy_buffer_prototype() -> None:
    buffer = cpu.buffer_prototype.buffer.create_zero_length()
    ndbuffer = cpu.buffer_prototype.nd_buffer.create(shape=(1, 2), dtype=np.dtype("int64"))
    assert isinstance(buffer.as_array_like(), np.ndarray)
    assert isinstance(ndbuffer.as_ndarray_like(), np.ndarray)
    with pytest.raises(ValueError, match="Buffer does not contain a single scalar value"):
        ndbuffer.as_scalar()
