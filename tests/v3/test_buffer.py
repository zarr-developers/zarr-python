from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from zarr.array import AsyncArray
from zarr.buffer import (
    ArrayLike,
    Buffer,
    BufferPrototype,
    NDArrayLike,
    cpu,
    gpu,
)
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.store.core import StorePath
from zarr.store.memory import MemoryStore

if TYPE_CHECKING:
    from typing_extensions import Self

try:
    import cupy as cp
except ImportError:
    cp = None


class MyNDArrayLike(np.ndarray):
    """An example of a ndarray-like class"""


class MyBuffer(cpu.Buffer):
    """Example of a custom Buffer that handles ArrayLike"""


class MyNDBuffer(cpu.NDBuffer):
    """Example of a custom NDBuffer that handles MyNDArrayLike"""

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
    ) -> Self:
        """Overwrite `NDBuffer.create` to create an MyNDArrayLike instance"""
        ret = cls(MyNDArrayLike(shape=shape, dtype=dtype, order=order))
        if fill_value is not None:
            ret.fill(fill_value)
        return ret


class MyStore(MemoryStore):
    """Example of a custom Store that expect MyBuffer for all its non-metadata

    We assume that keys containing "json" is metadata
    """

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        if "json" not in key:
            assert isinstance(value, MyBuffer)
        await super().set(key, value, byte_range)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int, int | None] | None = None,
    ) -> Buffer | None:
        if "json" not in key:
            assert prototype.buffer is MyBuffer
        ret = await super().get(key=key, prototype=prototype, byte_range=byte_range)
        if ret is not None:
            assert isinstance(ret, prototype.buffer)
        return ret


def test_nd_array_like(xp):
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@pytest.mark.asyncio
async def test_async_array_prototype():
    """Test the use of a custom buffer prototype"""

    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MyStore(mode="w")) / "test_async_array_prototype",
        shape=expect.shape,
        chunk_shape=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
    )
    expect[1:4, 3:6] = np.ones((3, 3))

    my_prototype = BufferPrototype(buffer=MyBuffer, nd_buffer=MyNDBuffer)

    await a.setitem(
        selection=(slice(1, 4), slice(3, 6)),
        value=np.ones((3, 3)),
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), prototype=my_prototype)
    assert isinstance(got, MyNDArrayLike)
    assert np.array_equal(expect, got)


@pytest.mark.skipif(cp is None, reason="requires cupy")
@pytest.mark.asyncio
async def test_async_array_gpu_prototype():
    """Test the use of the GPU buffer prototype"""

    expect = cp.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MemoryStore(mode="w")) / "test_async_array_gpu_prototype",
        shape=expect.shape,
        chunk_shape=(5, 5),
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
async def test_codecs_use_of_prototype():
    expect = np.zeros((10, 10), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MyStore(mode="w")) / "test_codecs_use_of_prototype",
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

    my_prototype = BufferPrototype(buffer=MyBuffer, nd_buffer=MyNDBuffer)

    await a.setitem(
        selection=(slice(0, 10), slice(0, 10)),
        value=expect[:],
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=my_prototype)
    assert isinstance(got, MyNDArrayLike)
    assert np.array_equal(expect, got)


@pytest.mark.skipif(cp is None, reason="requires cupy")
@pytest.mark.asyncio
async def test_codecs_use_of_gpu_prototype():
    expect = cp.zeros((10, 10), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MemoryStore(mode="w")) / "test_codecs_use_of_gpu_prototype",
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
    expect[:] = cp.arange(100).reshape(10, 10)

    await a.setitem(
        selection=(slice(0, 10), slice(0, 10)),
        value=expect[:],
        prototype=gpu.buffer_prototype,
    )
    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=gpu.buffer_prototype)
    assert isinstance(got, cp.ndarray)
    assert cp.array_equal(expect, got)
