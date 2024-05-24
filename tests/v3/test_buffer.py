from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from zarr.array import AsyncArray
from zarr.buffer import ArrayLike, Buffer, NDArrayLike, NDBuffer, Prototype
from zarr.store.core import StorePath
from zarr.store.memory import MemoryStore

if TYPE_CHECKING:
    from typing_extensions import Self


class MyNDArrayLike(np.ndarray):
    """An example of a ndarray-like class"""


class MyBuffer(Buffer):
    """Example of a custom Buffer that handles ArrayLike"""


class MyNDBuffer(NDBuffer):
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

    def as_buffer(self) -> Buffer:
        return MyBuffer.from_array_like(self.as_ndarray_like().ravel().view(dtype="b"))


class MyStore(MemoryStore):
    """Example of a custom Store that expect MyBuffer for all its non-metadata"""

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # Check that non-metadata is using MyBuffer
        if "json" not in key:
            assert isinstance(value, MyBuffer)
        await super().set(key, value, byte_range)


def test_nd_array_like(xp):
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@pytest.mark.asyncio
async def test_async_array_prototype():
    """Test the use of a custom buffer prototype"""

    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        StorePath(MyStore()) / "test_async_array_prototype",
        shape=expect.shape,
        chunk_shape=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
    )
    expect[1:4, 3:6] = np.ones((3, 3))

    my_prototype = Prototype(buffer=MyBuffer, nd_buffer=MyNDBuffer)

    await a.setitem(
        selection=(slice(1, 4), slice(3, 6)),
        value=np.ones((3, 3)),
        prototype=my_prototype,
    )
    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), prototype=my_prototype)
    assert isinstance(got, MyNDArrayLike)
    assert np.array_equal(expect, got)
