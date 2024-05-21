from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np

from zarr.common import ChunkCoords

if TYPE_CHECKING:
    from typing_extensions import Self

    from zarr.codecs.bytes import Endian
    from zarr.common import BytesLike

# TODO: create a protocol for the attributes we need, for now we alias Numpy's ndarray
#       both for the array-like and ndarray-like
ArrayLike: TypeAlias = np.ndarray


@runtime_checkable
class NDArrayLike(Protocol):
    """Protocol for the nd-array-like type that underlie NDBuffer"""

    dtype: np.DTypeLike
    shape: ChunkCoords
    ndim: int
    size: int

    @runtime_checkable
    class flags(Protocol):
        contiguous: bool

    def __len__(self) -> int: ...

    def __getitem__(self, key: slice) -> Self: ...

    def __setitem__(self, key: slice, value: Any) -> None: ...

    def reshape(self, shape: ChunkCoords, order: Literal["A", "C", "F"] = ...) -> Self: ...

    def view(self, dtype: np.DTypeLike) -> Self: ...

    def astype(self, dtype: np.DTypeLike, order: Literal["K", "A", "C", "F"] = ...) -> Self: ...

    def fill(self, value: Any) -> None: ...

    def copy(self) -> Self: ...

    def transpose(self, *axes: np.SupportsIndex) -> Self: ...

    def ravel(self, order: Literal["K", "A", "C", "F"]) -> Self: ...


def check_item_key_is_1d_contiguous(key: Any) -> None:
    """Raises error if `key` isn't a 1d contiguous slice"""
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
            fill_value: Any | None,
        ) -> NDBuffer:
            """Factory function to create a new NDBuffer (or subclass)

            Callables implementing the `Factory.Create` protocol must create a new
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

            Callables implementing the `Factory.NDArrayLike` protocol must return
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

    We use Buffer throughout Zarr to represent a contiguous block of memory.

    A Buffer is backed by a underlying array-like instance that represents
    the memory. The memory type is unspecified; can be regular host memory,
    CUDA device memory, or something else. The only requirement is that the
    array-like instance can be copied/converted to a regular Numpy array
    (host memory).

    Note
    ----
    This buffer is untyped, so all indexing and sizes are in bytes.

    Parameters
    ----------
    array_like
        array-like object that must be 1-dim, contiguous, and byte dtype.
    """

    def __init__(self, array_like: ArrayLike):
        if array_like.ndim != 1:
            raise ValueError("array_like: only 1-dim allowed")
        if array_like.dtype != np.dtype("b"):
            raise ValueError("array_like: only byte dtype allowed")
        self._data = array_like

    @classmethod
    def create_zero_length(cls) -> Self:
        """Create an empty buffer with length zero

        Return
        ------
            New empty 0-length buffer
        """
        return cls(np.array([], dtype="b"))

    @classmethod
    def from_array_like(cls, array_like: NDArrayLike) -> Self:
        """Create a new buffer of a array-like object

        Parameters
        ----------
        array_like
            array-like object that must be 1-dim, contiguous, and byte dtype.

        Return
        ------
            New buffer representing `array_like`
        """
        return cls(array_like)

    @classmethod
    def from_bytes(cls, bytes_like: BytesLike) -> Self:
        """Create a new buffer of a bytes-like object (host memory)

        Parameters
        ----------
        bytes_like
           bytes-like object

        Return
        ------
            New buffer representing `bytes_like`
        """
        return cls.from_array_like(np.frombuffer(bytes_like, dtype="b"))

    def as_array_like(self) -> NDArrayLike:
        """Return the underlying array (host or device memory) of this buffer

        This will never copy data.

        Return
        ------
            The underlying 1d array such as a NumPy or CuPy array.
        """
        return self._data

    def as_nd_buffer(self, *, dtype: np.DTypeLike) -> NDBuffer:
        """Create a new NDBuffer from this one.

        This will never copy data.

        Parameters
        ----------
        dtype
           The datatype of the returned buffer (reinterpretation of the bytes)

        Return
        ------
            New NDbuffer representing `self.as_array_like()`
        """
        return NDBuffer.from_ndarray_like(self._data.view(dtype=dtype))

    def as_numpy_array(self) -> np.ndarray:
        """Return the buffer as a NumPy array (host memory).

        Warning
        -------
        Might have to copy data, consider using `.as_array_like()` instead.

        Return
        ------
            NumPy array of this buffer (might be a data copy)
        """
        return np.asanyarray(self._data)

    def to_bytes(self) -> bytes:
        """Return the buffer as `bytes` (host memory).

        Warning
        -------
        Will always copy data, only use this method for small buffers such as metadata
        buffers. If possible, use `.as_numpy_array()` or `.as_array_like()` instead.

        Return
        ------
            `bytes` of this buffer (data copy)
        """
        return bytes(self.as_numpy_array())

    def __getitem__(self, key: slice) -> Self:
        check_item_key_is_1d_contiguous(key)
        return self.__class__(self._data.__getitem__(key))

    def __setitem__(self, key: slice, value: Any) -> None:
        check_item_key_is_1d_contiguous(key)
        self._data.__setitem__(key, value)

    def __len__(self) -> int:
        return self._data.size

    def __add__(self, other: Buffer) -> Self:
        """Concatenate two buffers"""

        other_array = other.as_array_like()
        assert other_array.dtype == np.dtype("b")
        return self.__class__(np.concatenate((self._data, other_array)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bytes | bytearray):
            # Many of the tests compares `Buffer` with `bytes` so we
            # convert the bytes to a Buffer and try again
            return self == self.from_bytes(other)
        if isinstance(other, Buffer):
            return (self._data == other.as_array_like()).all()
        raise ValueError(
            f"equal operator not supported between {self.__class__} and {other.__class__}"
        )


class NDBuffer:
    """A n-dimensional memory block

    We use NDBuffer throughout Zarr to represent a n-dimensional memory block.

    A NDBuffer is backed by a underlying ndarray-like instance that represents
    the memory. The memory type is unspecified; can be regular host memory,
    CUDA device memory, or something else. The only requirement is that the
    ndarray-like instance can be copied/converted to a regular Numpy array
    (host memory).

    Note
    ----
    The two buffer classes Buffer and NDBuffer are very similar. In fact, Buffer
    is a special case of NDBuffer where dim=1, stride=1, and dtype="b". However,
    in order to use Python's type system to differentiate between the contiguous
    Buffer and the n-dim (non-contiguous) NDBuffer, we keep the definition of the
    two classes separate.

    Parameters
    ----------
    ndarray_like
        ndarray-like object that is convertible to a regular Numpy array.
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
        fill_value: Any | None = None,
    ) -> Self:
        """Create a new buffer and its underlying ndarray-like object

        Parameters
        ----------
        shape
            The shape of the buffer and its underlying ndarray-like object
        dtype
            The datatype of the buffer and its underlying ndarray-like object
        order
            Whether to store multi-dimensional data in row-major (C-style) or
            column-major (Fortran-style) order in memory.
        fill_value
            If not None, fill the new buffer with a scalar value.

        Return
        ------
            New buffer representing a new ndarray_like object

        Developer Notes
        ---------------
        A subclass can overwrite this method to create a ndarray-like object
        other then the default Numpy array.
        """
        ret = cls(np.empty(shape=shape, dtype=dtype, order=order))
        if fill_value is not None:
            ret.fill(fill_value)
        return ret

    @classmethod
    def from_ndarray_like(cls, ndarray_like: NDArrayLike) -> Self:
        """Create a new buffer of a ndarray-like object

        Parameters
        ----------
        ndarray_like
            ndarray-like object

        Return
        ------
            New buffer representing `ndarray_like`
        """
        return cls(ndarray_like)

    @classmethod
    def from_numpy_array(cls, array_like: np.ArrayLike) -> Self:
        """Create a new buffer of Numpy array-like object

        Parameters
        ----------
        array_like
            Object that can be coerced into a Numpy array

        Return
        ------
            New buffer representing `array_like`
        """
        return cls.from_ndarray_like(np.asanyarray(array_like))

    def as_ndarray_like(self) -> NDArrayLike:
        """Return the underlying array (host or device memory) of this buffer

        This will never copy data.

        Return
        ------
            The underlying array such as a NumPy or CuPy array.
        """
        return self._data

    def as_buffer(self) -> Buffer:
        """Create a new Buffer from this one.

        Warning
        -------
        Copies data if the buffer is non-contiguous.

        Return
        ------
            The new buffer (might be data copy)
        """
        data = self._data
        if not self._data.flags.contiguous:
            data = np.ascontiguousarray(self._data)
        return Buffer(data.reshape((-1,)).view(dtype="b"))  # Flatten the array without copy

    def as_numpy_array(self) -> np.ndarray:
        """Return the buffer as a NumPy array (host memory).

        Warning
        -------
        Might have to copy data, consider using `.as_ndarray_like()` instead.

        Return
        ------
            NumPy array of this buffer (might be a data copy)
        """
        return np.asanyarray(self._data)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
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

    def reshape(self, newshape: ChunkCoords) -> Self:
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


def as_numpy_array_wrapper(func: Callable[[np.ndarray], bytes], buf: Buffer) -> Buffer:
    """Converts the input of `func` to a numpy array and the output back to `Buffer`.

    This function is useful when calling a `func` that only support host memory such
    as `GZip.decode` and `Blosc.decode`. In this case, use this wrapper to convert
    the input `buf` to a Numpy array and convert the result back into a `Buffer`.

    Parameters
    ----------
    func
        The callable that will be called with the converted `buf` as input.
        `func` must return bytes, which will be converted into a `Buffer`
        before returned.
    buf
        The buffer that will be converted to a Numpy array before given as
        input to `func`.

    Return
    ------
        The result of `func` converted to a `Buffer`
    """
    return Buffer.from_bytes(func(buf.as_numpy_array()))
