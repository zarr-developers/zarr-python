from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    SupportsIndex,
    cast,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from zarr.registry import (
    get_buffer_class,
    get_ndbuffer_class,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self

    from zarr.codecs.bytes import Endian
    from zarr.core.common import BytesLike, ChunkCoords

# Everything here is imported into ``zarr.core.buffer`` namespace.
__all__: list[str] = []


@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for the array-like type that underlie Buffer"""

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def size(self) -> int: ...

    def __getitem__(self, key: slice) -> Self: ...

    def __setitem__(self, key: slice, value: Any) -> None: ...


@runtime_checkable
class NDArrayLike(Protocol):
    """Protocol for the nd-array-like type that underlie NDBuffer"""

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def shape(self) -> ChunkCoords: ...

    def __len__(self) -> int: ...

    def __getitem__(self, key: slice) -> Self: ...

    def __setitem__(self, key: slice, value: Any) -> None: ...

    def __array__(self) -> npt.NDArray[Any]: ...

    def reshape(
        self, shape: ChunkCoords | Literal[-1], *, order: Literal["A", "C", "F"] = ...
    ) -> Self: ...

    def view(self, dtype: npt.DTypeLike) -> Self: ...

    def astype(self, dtype: npt.DTypeLike, order: Literal["K", "A", "C", "F"] = ...) -> Self: ...

    def fill(self, value: Any) -> None: ...

    def copy(self) -> Self: ...

    def transpose(self, axes: SupportsIndex | Sequence[SupportsIndex] | None) -> Self: ...

    def ravel(self, order: Literal["K", "A", "C", "F"] = ...) -> Self: ...

    def all(self) -> bool: ...

    def __eq__(self, other: object) -> Self:  # type: ignore[explicit-override, override]
        """Element-wise equal

        Notes
        -----
        Type checkers such as mypy complains because the return type isn't a bool like
        its supertype "object", which violates the Liskov substitution principle.
        This is true, but since NumPy's ndarray is defined as an element-wise equal,
        our hands are tied.
        """


def check_item_key_is_1d_contiguous(key: Any) -> None:
    """Raises error if `key` isn't a 1d contiguous slice"""
    if not isinstance(key, slice):
        raise TypeError(
            f"Item key has incorrect type (expected slice, got {key.__class__.__name__})"
        )
    if not (key.step is None or key.step == 1):
        raise ValueError("slice must be contiguous")


class Buffer(ABC):
    """A flat contiguous memory block

    We use Buffer throughout Zarr to represent a contiguous block of memory.

    A Buffer is backed by a underlying array-like instance that represents
    the memory. The memory type is unspecified; can be regular host memory,
    CUDA device memory, or something else. The only requirement is that the
    array-like instance can be copied/converted to a regular Numpy array
    (host memory).

    Notes
    -----
    This buffer is untyped, so all indexing and sizes are in bytes.

    Parameters
    ----------
    array_like
        array-like object that must be 1-dim, contiguous, and byte dtype.
    """

    def __init__(self, array_like: ArrayLike) -> None:
        if array_like.ndim != 1:
            raise ValueError("array_like: only 1-dim allowed")
        if array_like.dtype != np.dtype("b"):
            raise ValueError("array_like: only byte dtype allowed")
        self._data = array_like

    @classmethod
    @abstractmethod
    def create_zero_length(cls) -> Self:
        """Create an empty buffer with length zero

        Returns
        -------
            New empty 0-length buffer
        """
        if cls is Buffer:
            raise NotImplementedError("Cannot call abstract method on the abstract class 'Buffer'")
        return cls(
            cast(ArrayLike, None)
        )  # This line will never be reached, but it satisfies the type checker

    @classmethod
    def from_array_like(cls, array_like: ArrayLike) -> Self:
        """Create a new buffer of an array-like object

        Parameters
        ----------
        array_like
            array-like object that must be 1-dim, contiguous, and byte dtype.

        Returns
        -------
            New buffer representing `array_like`
        """
        return cls(array_like)

    @classmethod
    @abstractmethod
    def from_buffer(cls, buffer: Buffer) -> Self:
        """Create a new buffer of an existing Buffer

        This is useful if you want to ensure that an existing buffer is
        of the correct subclass of Buffer. E.g., MemoryStore uses this
        to return a buffer instance of the subclass specified by its
        BufferPrototype argument.

        Typically, this only copies data if the data has to be moved between
        memory types, such as from host to device memory.

        Parameters
        ----------
        buffer
            buffer object.

        Returns
        -------
            A new buffer representing the content of the input buffer

        Notes
        -----
        Subclasses of `Buffer` must override this method to implement
        more optimal conversions that avoid copies where possible
        """
        if cls is Buffer:
            raise NotImplementedError("Cannot call abstract method on the abstract class 'Buffer'")
        return cls(
            cast(ArrayLike, None)
        )  # This line will never be reached, but it satisfies the type checker

    @classmethod
    @abstractmethod
    def from_bytes(cls, bytes_like: BytesLike) -> Self:
        """Create a new buffer of a bytes-like object (host memory)

        Parameters
        ----------
        bytes_like
           bytes-like object

        Returns
        -------
            New buffer representing `bytes_like`
        """
        if cls is Buffer:
            raise NotImplementedError("Cannot call abstract method on the abstract class 'Buffer'")
        return cls(
            cast(ArrayLike, None)
        )  # This line will never be reached, but it satisfies the type checker

    def as_array_like(self) -> ArrayLike:
        """Returns the underlying array (host or device memory) of this buffer

        This will never copy data.

        Returns
        -------
            The underlying 1d array such as a NumPy or CuPy array.
        """
        return self._data

    @abstractmethod
    def as_numpy_array(self) -> npt.NDArray[Any]:
        """Returns the buffer as a NumPy array (host memory).

        Notes
        -----
        Might have to copy data, consider using `.as_array_like()` instead.

        Returns
        -------
            NumPy array of this buffer (might be a data copy)
        """
        ...

    def to_bytes(self) -> bytes:
        """Returns the buffer as `bytes` (host memory).

        Warnings
        --------
        Will always copy data, only use this method for small buffers such as metadata
        buffers. If possible, use `.as_numpy_array()` or `.as_array_like()` instead.

        Returns
        -------
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

    @abstractmethod
    def __add__(self, other: Buffer) -> Self:
        """Concatenate two buffers"""
        ...

    def __eq__(self, other: object) -> bool:
        # Another Buffer class can override this to choose a more efficient path
        return isinstance(other, Buffer) and np.array_equal(
            self.as_numpy_array(), other.as_numpy_array()
        )


class NDBuffer:
    """An n-dimensional memory block

    We use NDBuffer throughout Zarr to represent a n-dimensional memory block.

    A NDBuffer is backed by a underlying ndarray-like instance that represents
    the memory. The memory type is unspecified; can be regular host memory,
    CUDA device memory, or something else. The only requirement is that the
    ndarray-like instance can be copied/converted to a regular Numpy array
    (host memory).

    Notes
    -----
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

    def __init__(self, array: NDArrayLike) -> None:
        self._data = array

    @classmethod
    @abstractmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
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

        Returns
        -------
            New buffer representing a new ndarray_like object

        Notes
        -----
        A subclass can overwrite this method to create a ndarray-like object
        other then the default Numpy array.
        """
        if cls is NDBuffer:
            raise NotImplementedError(
                "Cannot call abstract method on the abstract class 'NDBuffer'"
            )
        return cls(
            cast(NDArrayLike, None)
        )  # This line will never be reached, but it satisfies the type checker

    @classmethod
    def from_ndarray_like(cls, ndarray_like: NDArrayLike) -> Self:
        """Create a new buffer of a ndarray-like object

        Parameters
        ----------
        ndarray_like
            ndarray-like object

        Returns
        -------
            New buffer representing `ndarray_like`
        """
        return cls(ndarray_like)

    @classmethod
    @abstractmethod
    def from_numpy_array(cls, array_like: npt.ArrayLike) -> Self:
        """Create a new buffer of Numpy array-like object

        Parameters
        ----------
        array_like
            Object that can be coerced into a Numpy array

        Returns
        -------
            New buffer representing `array_like`
        """
        if cls is NDBuffer:
            raise NotImplementedError(
                "Cannot call abstract method on the abstract class 'NDBuffer'"
            )
        return cls(
            cast(NDArrayLike, None)
        )  # This line will never be reached, but it satisfies the type checker

    def as_ndarray_like(self) -> NDArrayLike:
        """Returns the underlying array (host or device memory) of this buffer

        This will never copy data.

        Returns
        -------
            The underlying array such as a NumPy or CuPy array.
        """
        return self._data

    @abstractmethod
    def as_numpy_array(self) -> npt.NDArray[Any]:
        """Returns the buffer as a NumPy array (host memory).

        Warnings
        --------
        Might have to copy data, consider using `.as_ndarray_like()` instead.

        Returns
        -------
            NumPy array of this buffer (might be a data copy)
        """
        ...

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

    def reshape(self, newshape: ChunkCoords | Literal[-1]) -> Self:
        return self.__class__(self._data.reshape(newshape))

    def squeeze(self, axis: tuple[int, ...]) -> Self:
        newshape = tuple(a for i, a in enumerate(self.shape) if i not in axis)
        return self.__class__(self._data.reshape(newshape))

    def astype(self, dtype: npt.DTypeLike, order: Literal["K", "A", "C", "F"] = "K") -> Self:
        return self.__class__(self._data.astype(dtype=dtype, order=order))

    @abstractmethod
    def __getitem__(self, key: Any) -> Self: ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None: ...

    def __len__(self) -> int:
        return self._data.__len__()

    def __repr__(self) -> str:
        return f"<NDBuffer shape={self.shape} dtype={self.dtype} {self._data!r}>"

    def all_equal(self, other: Any, equal_nan: bool = True) -> bool:
        """Compare to `other` using np.array_equal."""
        if other is None:
            # Handle None fill_value for Zarr V2
            return False
        # use array_equal to obtain equal_nan=True functionality
        # Since fill-value is a scalar, isn't there a faster path than allocating a new array for fill value
        # every single time we have to write data?
        _data, other = np.broadcast_arrays(self._data, other)
        return np.array_equal(
            self._data, other, equal_nan=equal_nan if self._data.dtype.kind not in "USTO" else False
        )

    def fill(self, value: Any) -> None:
        self._data.fill(value)

    def copy(self) -> Self:
        return self.__class__(self._data.copy())

    def transpose(self, axes: SupportsIndex | Sequence[SupportsIndex] | None) -> Self:
        return self.__class__(self._data.transpose(axes))


class BufferPrototype(NamedTuple):
    """Prototype of the Buffer and NDBuffer class

    The protocol must be pickable.

    Attributes
    ----------
    buffer
        The Buffer class to use when Zarr needs to create new Buffer.
    nd_buffer
        The NDBuffer class to use when Zarr needs to create new NDBuffer.
    """

    buffer: type[Buffer]
    nd_buffer: type[NDBuffer]


# The default buffer prototype used throughout the Zarr codebase.
def default_buffer_prototype() -> BufferPrototype:
    return BufferPrototype(buffer=get_buffer_class(), nd_buffer=get_ndbuffer_class())
