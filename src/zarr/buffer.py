from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
)

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.codecs.bytes import Endian
    from zarr.common import BytesLike

# TODO: create a protocol for the attributes we need, for now we just aliasing numpy
NDArrayLike: TypeAlias = np.ndarray


def check_item_key_is_1d_contiguous(key: Any) -> None:
    if not isinstance(key, slice):
        raise TypeError(
            f"Item key has incorrect type (expected slice, got {key.__class__.__name__})"
        )
    if not (key.step is None or key.step == 1):
        raise ValueError("slice must be contiguous")


class Factory:
    class Create(Protocol):
        def __call__(
            self,
            *,
            shape: Iterable[int],
            dtype: np.DTypeLike,
            order: Literal["C", "F"],
            fill_value: Optional[Any],
        ) -> NDBuffer:
            """Factory function to create a new NDBuffer (or subclass)

            Callables implementing the `Factor.Create` protocol must create a new
            instance of NDBuffer (or subclass) given the following parameters.

            Parameters
            ----------
            shape
                The shape of the new buffer
            dtype
                The datatype of each element in the new buffer
            order
                Whether to store multi-dimensional data in row-major (C-style) or
                column-major (Fortran-style) order in memory.
            fill_value
                If not None, fill the new buffer with a scalar value.

            Return
            ------
                A new NDBuffer or subclass instance
            """

    class NDArrayLike(Protocol):
        def __call__(self, ndarray_like: NDArrayLike) -> NDBuffer:
            """Factory function to coerce an array into a NDBuffer (or subclass)

            Callables implementing the `Factor.NDArrayLike` protocol must return
            an instance of NDBuffer (or subclass) given an ndarray-like object.

            Parameters
            ----------
            ndarray_like
                ndarray-like object

            Return
            ------
                A NDBuffer or subclass instance that represents `ndarray_like`
            """


class Buffer:
    """A flat contiguous memory block

    We use `Buffer` throughout Zarr to represent a contiguous block of memory.
    For now, we only support host memory but the plan is to support other types
    of memory such as CUDA device memory.
    """

    def __init__(self, array: NDArrayLike):
        assert array.ndim == 1
        assert array.itemsize == 1
        assert array.dtype == np.dtype("b")
        self._data = array

    @classmethod
    def create_zero_length(cls) -> Self:
        return cls(np.array([], dtype="b"))

    @classmethod
    def from_numpy_array(cls, array_like: np.ArrayLike) -> Self:
        return cls(np.asarray(array_like).reshape(-1).view(dtype="b"))

    @classmethod
    def from_bytes(cls, data: BytesLike) -> Self:
        return cls.from_numpy_array(np.frombuffer(data, dtype="b"))

    def as_nd_buffer(self, *, dtype: np.DTypeLike) -> NDBuffer:
        return NDBuffer(self._data.view(dtype=dtype))

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def memoryview(self) -> memoryview:
        return memoryview(self._data)

    def __getitem__(self, key: slice) -> Self:
        check_item_key_is_1d_contiguous(key)
        return self.__class__(self._data.__getitem__(key))

    def __setitem__(self, key: slice, value: Any) -> None:
        check_item_key_is_1d_contiguous(key)
        self._data.__setitem__(key, value)

    def __len__(self) -> int:
        return self._data.size

    def __add__(self, other: Buffer) -> Self:
        assert other._data.dtype == np.dtype("b")
        return self.__class__(np.concatenate((self._data, other._data)))

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

    def __init__(self, array: NDArrayLike):
        assert array.ndim > 0
        assert array.dtype != object
        self._data = array

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: np.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Optional[Any] = None,
    ) -> Self:
        ret = cls(np.empty(shape=shape, dtype=dtype, order=order))
        if fill_value is not None:
            ret.fill(fill_value)
        return ret

    @classmethod
    def from_numpy_array(cls, array_like: np.ArrayLike) -> Self:
        return cls(np.asanyarray(array_like))

    @classmethod
    def from_ndarray_like(cls, ndarray_like: NDArrayLike) -> Self:
        return cls(ndarray_like)

    def as_ndarray_like(self) -> NDArrayLike:
        """Return the underlying array instance representing the memory of this buffer

        This will never copy data.

        Return
        ------
            The underlying array such as a NumPy or CuPy array.
        """
        return self._data

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
