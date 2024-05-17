from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from zarr.array import AsyncArray
from zarr.buffer import NDBuffer
from zarr.store.core import StorePath
from zarr.store.memory import MemoryStore

if TYPE_CHECKING:
    from typing_extensions import Self


class MyNDArrayLike(np.ndarray):
    """An example of a ndarray-like class"""

    pass


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


@pytest.mark.asyncio
async def test_async_array_factory():
    store = StorePath(MemoryStore())
    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        store / "test_async_array",
        shape=expect.shape,
        chunk_shape=(5, 5),
        dtype=expect.dtype,
        fill_value=0,
    )
    expect[1:4, 3:6] = np.ones((3, 3))

    await a.setitem(
        selection=(slice(1, 4), slice(3, 6)),
        value=np.ones((3, 3)),
        factory=MyNDBuffer.from_ndarray_like,
    )
    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), factory=MyNDBuffer.create)
    assert isinstance(got, MyNDArrayLike)
    assert np.array_equal(expect, got)
