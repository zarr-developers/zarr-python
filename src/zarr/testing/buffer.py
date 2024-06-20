from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from zarr.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.store import MemoryStore

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
        return await super().get(key, byte_range)
