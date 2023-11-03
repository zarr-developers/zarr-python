from __future__ import annotations
from abc import abstractproperty, abstractmethod, ABC
from typing import Tuple, Any, Dict

import numpy as np

from zarr.v3.abc.store import ReadStore, WriteStore
from zarr.v3.common import Selection


class BaseArray(ABC):
    @abstractproperty
    def store_path(self) -> str:  # TODO: rename to `path`?
        """Path to this array in the underlying store."""
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        """Data type of the array elements.

        Returns
        -------
        dtype
            array data type
        """
        ...

    @abstractproperty
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Returns
        -------
        int
            number of array dimensions (axes)
        """
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions.

        Returns
        -------
        tuple of int
            array dimensions
        """
        ...

    @abstractproperty
    def size(self) -> int:
        """Number of elements in the array.

        Returns
        -------
        int
            number of elements in an array.
        """

    @abstractproperty
    def attrs(self) -> Dict[str, Any]:
        """Array attributes.

        Returns
        -------
        dict
            user defined attributes
        """
        ...

    @abstractproperty
    def info(self) -> Any:
        """Report some diagnostic information about the array.

        Returns
        -------
        out
        """
        ...


class AsynchronousArray(BaseArray):
    """This class can be implemented as a v2 or v3 array"""

    @classmethod
    @abstractmethod
    async def from_json(cls, zarr_json: Any, store: ReadStore) -> AsynchronousArray:
        ...

    @classmethod
    @abstractmethod
    async def open(cls, store: ReadStore) -> AsynchronousArray:
        ...

    @classmethod
    @abstractmethod
    async def create(cls, store: WriteStore, *, shape, **kwargs) -> AsynchronousArray:
        ...

    @abstractmethod
    async def getitem(self, selection: Selection):
        ...

    @abstractmethod
    async def setitem(self, selection: Selection, value: np.ndarray) -> None:
        ...


class SynchronousArray(BaseArray):
    """
    This class can be implemented as a v2 or v3 array
    """

    @classmethod
    @abstractmethod
    def from_json(cls, zarr_json: Any, store: ReadStore) -> SynchronousArray:
        ...

    @classmethod
    @abstractmethod
    def open(cls, store: ReadStore) -> SynchronousArray:
        ...

    @classmethod
    @abstractmethod
    def create(cls, store: WriteStore, *, shape, **kwargs) -> SynchronousArray:
        ...

    @abstractmethod
    def __getitem__(self, selection: Selection):  # TODO: type as np.ndarray | scalar
        ...

    @abstractmethod
    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        ...

    # some day ;)
    # @property
    # def __array_api_version__(self) -> str:
    #     return "2022.12"
