from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from zarr.array import AsyncArray
from zarr.buffer import ArrayLike, NDArrayLike, NDBuffer
from zarr.testing.utils import IS_WASM

if TYPE_CHECKING:
    from typing_extensions import Self


# Helper function to skip async tests on WASM platforms
def asyncio_tests_wrapper(func):
    return func if IS_WASM else pytest.mark.asyncio(func)


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


def test_nd_array_like(xp):
    ary = xp.arange(10)
    assert isinstance(ary, ArrayLike)
    assert isinstance(ary, NDArrayLike)


@asyncio_tests_wrapper
async def test_async_array_factory(store_path):
    expect = np.zeros((9, 9), dtype="uint16", order="F")
    a = await AsyncArray.create(
        store_path,
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
