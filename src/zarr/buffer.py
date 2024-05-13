from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Tuple
import numpy as np

from zarr.common import BytesLike


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.codecs.bytes import Endian


class Buffer:
    """A flat contiguous version of `NDBuffer` with an item size of 1"""

    def __init__(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.itemsize == 1
        assert array.dtype == np.dtype("b")
        self._data = array

    @classmethod
    def create_empty(cls, *, nbytes: int) -> Self:
        return cls(np.empty(shape=(nbytes,), dtype="b"))

    @classmethod
    def from_numpy_array(cls, array: np.ArrayLike) -> Self:
        return cls(np.asanyarray(array).reshape(-1).view(dtype="b"))

    @classmethod
    def from_bytes(cls, data: BytesLike) -> Self:
        return cls.from_numpy_array(np.frombuffer(data, dtype="b"))

    def as_nd_buffer(self, *, dtype: np.DTypeLike) -> NDBuffer:
        return NDBuffer(self._data.view(dtype=dtype))

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def memoryview(self) -> memoryview:
        return memoryview(self._data)

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(self._data.__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data.__setitem__(key, value)

    def __len__(self) -> int:
        return self._data.nbytes

    def __add__(self, other: Buffer) -> Self:
        return self.__class__(np.frombuffer(self.to_bytes() + other.to_bytes(), dtype="b"))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (bytes, bytearray)):
            return self.to_bytes() == other
        raise ValueError(
            f"equal operator not supported between {self.__class__} and {other.__class__}"
        )


class NDBuffer:
    """A n-dimensional memory block

    We use `NDBuffer` throughout Zarr to represent a block of memory.
    For now, we only support host memory but the plan is to support other types
    of memory such as CUDA device memory.
    """

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

    @classmethod
    def from_numpy_array(cls, array: np.ArrayLike) -> Self:
        return cls(np.asanyarray(array))

    def as_buffer(self) -> Buffer:
        return Buffer(self._data.reshape(-1).view(dtype="b"))

    def as_numpy_array(self) -> np.ndarray:
        """Return the buffer as a NumPy array.

        Warning
        -------
        Might have to copy data, only use this method for small buffers such as metadata

        Return
        ------
            NumPy array of this buffer (might be a data copy)
        """
        return self._data

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

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
        return self.__class__(self._data.reshape(newshape))

    def astype(self, dtype: np.DTypeLike, order: Literal["K", "A", "C", "F"] = "K") -> Self:
        return self.__class__(self._data.astype(dtype=dtype, order=order))

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(np.asanyarray(self._data.__getitem__(key)))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value._data
        self._data.__setitem__(key, value)

    def __len__(self) -> int:
        return self._data.__len__()

    def all_equal(self, other: Any) -> bool:
        return bool((self._data == other).all())

    def fill(self, value: Any) -> None:
        self._data.fill(value)

    def copy(self) -> Self:
        return self.__class__(self._data.copy())

    def transpose(self, *axes: np.SupportsIndex) -> Self:
        return self.__class__(self._data.transpose(*axes))


def as_bytes_wrapper(func: Callable[[bytes], bytes], buf: Buffer) -> Buffer:
    return Buffer.from_bytes(func(buf.to_bytes()))
