from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Tuple
import numpy as np


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.codecs.bytes import Endian


class NDBuffer:
    def __init__(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        assert array.dtype != object
        self._data = array

    @classmethod
    def create_empty(
        cls, *, shape: Iterable[int], dtype: np.DTypeLike, order: Literal["C", "F"] = "C"
    ) -> Self:
        return cls(np.empty(shape=shape, dtype=dtype, order=order))

    @classmethod
    def create_zeros(
        cls, *, shape: Iterable[int], dtype: np.DTypeLike, order: Literal["C", "F"] = "C"
    ) -> Self:
        return cls(np.zeros(shape=shape, dtype=dtype, order=order))

    def as_numpy_array(self, dtype: Optional[np.DTypeLike] = None) -> np.ndarray:
        if dtype is None:
            return self._data
        return self._data.astype(dtype=dtype, copy=False)

    @property
    def dtype(self) -> np.DTypeLike:
        return self.as_numpy_array().dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.as_numpy_array().shape

    @property
    def byteorder(self) -> Endian:
        from zarr.codecs.bytes import Endian

        if self.dtype.byteorder == "<":
            return Endian.little
        elif self.dtype.byteorder == ">":
            return Endian.big
        else:
            return Endian(sys.byteorder)

    def reshape(self, newshape: Iterable[int]) -> Self:
        return self.__class__(self.as_numpy_array().reshape(newshape))

    def astype(self, dtype: np.DTypeLike, order: Literal["K", "A", "C", "F"] = "K") -> Self:
        return self.__class__(self.as_numpy_array().astype(dtype=dtype, order=order))

    def __getitem__(self, key: Any) -> Self:
        # print("__getitem__: \n", np.asanyarray(self.as_numpy_array().__getitem__(key)))
        return self.__class__(np.asanyarray(self.as_numpy_array().__getitem__(key)))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value.as_numpy_array()
        self.as_numpy_array().__setitem__(key, value)

    def __len__(self) -> int:
        return self.as_numpy_array().__len__()

    def fill(self, value: Any) -> None:
        self.as_numpy_array().fill(value)

    def copy(self) -> Self:
        return self.__class__(self.as_numpy_array().copy())

    def transpose(self, *axes: np.SupportsIndex) -> Self:
        return self.__class__(self.as_numpy_array().transpose(*axes))


class Buffer(NDBuffer):
    """Contiguous memory block

    We use `Buffer` throughout Zarr to represent a contiguous block of memory.
    For now, we only support host memory but the plan is to support other types
    of memory such as CUDA device memory.
    """

    @classmethod
    def create_empty(
        cls, *, shape: Iterable[int], dtype: np.DTypeLike = "b", order: Literal["C", "F"] = "C"
    ) -> Self:
        return cls(np.empty(shape=shape, dtype=dtype, order=order))

    def memoryview(self) -> memoryview:
        return memoryview(self._data.reshape(-1).view(dtype="b"))

    def as_numpy_array(self, dtype: Optional[np.DTypeLike] = "b") -> np.ndarray:
        return self._data.reshape(-1).view(dtype=dtype)

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(self.as_numpy_array().__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        self.as_numpy_array().__setitem__(key, value)

    def __len__(self) -> int:
        return self._data.nbytes

    def __add__(self, other: Buffer) -> Self:
        return self.__class__(np.frombuffer(self.to_bytes() + other.to_bytes(), dtype="b"))


def as_nd_buffer(data: Any) -> NDBuffer:
    if isinstance(data, NDBuffer):
        return data
    return NDBuffer(np.asanyarray(data))


def as_ndarray(data: Optional[NDBuffer]) -> Optional[np.ndarray]:
    if data is None:
        return data
    return data.as_numpy_array()


def as_buffer(data: Any) -> Buffer:
    if isinstance(data, Buffer):
        return data
    if isinstance(data, NDBuffer):
        return Buffer(data.as_numpy_array())
    return Buffer(np.asanyarray(data))


def as_bytes_wrapper(func: Callable[[bytes], bytes], buf: Buffer) -> Buffer:
    return as_buffer(func(buf.to_bytes()))


def return_as_bytes_wrapper(func: Callable[[Any], Any], *arg: Any, **kwargs: Any) -> Buffer:
    return as_buffer(func(*arg, **kwargs))


def as_bytearray(data: Optional[Buffer]) -> Optional[bytes]:
    if data is None:
        return data
    return data.to_bytes()
