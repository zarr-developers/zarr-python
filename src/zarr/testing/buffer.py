# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from zarr.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.store import MemoryStore

if TYPE_CHECKING:
    from typing_extensions import Self


class TestNDArrayLike(np.ndarray):
    """An example of a ndarray-like class"""


class TestBuffer(Buffer):
    """Example of a custom Buffer that handles ArrayLike"""


class NDBufferUsingTestNDArrayLike(NDBuffer):
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
        """Overwrite `NDBuffer.create` to create an TestNDArrayLike instance"""
        ret = cls(TestNDArrayLike(shape=shape, dtype=dtype, order=order))
        if fill_value is not None:
            ret.fill(fill_value)
        return ret


class StoreExpectingTestBuffer(MemoryStore):
    """Example of a custom Store that expect MyBuffer for all its non-metadata

    We assume that keys containing "json" is metadata
    """

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        if "json" not in key:
            assert isinstance(value, TestBuffer)
        await super().set(key, value, byte_range)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int, int | None] | None = None,
    ) -> Buffer | None:
        if "json" not in key:
            assert prototype.buffer is TestBuffer
        ret = await super().get(key=key, prototype=prototype, byte_range=byte_range)
        if ret is not None:
            assert isinstance(ret, prototype.buffer)
        return ret
