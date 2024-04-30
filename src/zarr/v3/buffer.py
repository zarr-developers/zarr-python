from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
import numpy as np


if TYPE_CHECKING:
    from typing_extensions import Self


class NDBuffer:
    # TODO: replace np.ndarray with this n-dimensional buffer
    pass


class Buffer(NDBuffer):
    """Contiguous memory block

    We use `Buffer` throughout Zarr to represent a contiguous block of memory.
    For now, we only support host memory but the plan is to support other types
    of memory such as CUDA device memory.
    """

    def __init__(self, data: bytearray):
        assert isinstance(data, bytearray)
        self._data = data

    def as_bytearray(self) -> bytearray:
        return self._data

    def as_numpy_array(self, dtype: np.DTypeLike) -> np.ndarray:
        return np.frombuffer(self._data, dtype=dtype)

    def __getitem__(self, key) -> Self:
        return self.__class__(self.as_bytearray().__getitem__(key))

    def __setitem__(self, key, value) -> None:
        self.as_bytearray().__setitem__(key, value)

    def __len__(self) -> int:
        return len(self.as_bytearray())

    def __add__(self, other: Buffer) -> Self:
        return self.__class__(self.as_bytearray() + other.as_bytearray())


def as_buffer(data: Any) -> Buffer:
    if isinstance(data, Buffer):
        return data
    if isinstance(data, bytearray):
        return Buffer(data)
    if isinstance(data, bytes):
        return Buffer(bytearray(data))
    if hasattr(data, "to_bytes"):
        return as_buffer(data.to_bytes())
    return Buffer(bytearray(np.asarray(data)))


def as_bytes_wrapper(func, buf: Buffer) -> Buffer:
    return as_buffer(func(buf.as_bytearray()))


def return_as_bytes_wrapper(func, *arg, **kwargs) -> Buffer:
    return as_buffer(func(*arg, **kwargs))


def as_bytearray(data: Optional[Buffer]):
    if data is None:
        return data
    return data.as_bytearray()
