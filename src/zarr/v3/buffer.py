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

    def __init__(self, data: memoryview):
        assert isinstance(data, memoryview)
        assert data.ndim == 1
        assert data.contiguous
        assert data.itemsize == 1
        self._data = data

    def memoryview(self) -> memoryview:
        return self._data

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def as_numpy_array(self, dtype: np.DTypeLike) -> np.ndarray:
        return np.frombuffer(self.memoryview(), dtype=dtype)

    def __getitem__(self, key) -> Self:
        return self.__class__(self.memoryview().__getitem__(key))

    def __setitem__(self, key, value) -> None:
        self.memoryview().__setitem__(key, value)

    def __len__(self) -> int:
        return len(self.memoryview())

    def __add__(self, other: Buffer) -> Self:
        return self.__class__(memoryview(self.to_bytes() + other.to_bytes()))


def as_buffer(data: Any) -> Buffer:
    if isinstance(data, Buffer):
        return data
    if isinstance(data, bytearray | bytes):
        return Buffer(memoryview(data))
    if hasattr(data, "to_bytes"):
        return as_buffer(memoryview(data.to_bytes()))
    return Buffer(memoryview(np.asanyarray(data).reshape(-1).view(dtype="int8")))


def as_bytes_wrapper(func, buf: Buffer) -> Buffer:
    return as_buffer(func(buf.to_bytes()))


def return_as_bytes_wrapper(func, *arg, **kwargs) -> Buffer:
    return as_buffer(func(*arg, **kwargs))


def as_bytearray(data: Optional[Buffer]) -> Optional[bytes]:
    if data is None:
        return data
    return data.to_bytes()
