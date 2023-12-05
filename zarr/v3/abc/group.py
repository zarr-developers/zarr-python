from __future__ import annotations

from abc import abstractproperty, abstractmethod, ABC
from collections.abc import MutableMapping
from typing import Dict, Any, AsyncIterator, Union, Iterator

from zarr.v3.abc.array import AsyncArray, SyncArray


class BaseGroup(ABC):
    @abstractproperty
    def attrs(self) -> Dict[str, Any]:
        """User-defined attributes."""
        ...

    @abstractproperty
    def info(self) -> Any:  # TODO: type this later
        """Return diagnostic information about the group."""
        ...


class AsyncGroup(BaseGroup):
    @abstractmethod
    async def nchildren(self) -> int:
        ...

    @abstractmethod
    async def children(self) -> AsyncIterator:
        ...

    @abstractmethod
    async def contains(self, child: str) -> bool:
        """check if child exists"""
        ...

    @abstractmethod
    async def getitem(self, child: str) -> Union[AsyncArray, "AsyncGroup"]:
        """get child"""
        ...

    @abstractmethod
    async def group_keys(self) -> AsyncIterator[str]:
        """iterate over child group keys"""
        ...

    @abstractmethod
    async def groups(self) -> AsyncIterator["AsyncGroup"]:
        """iterate over child groups"""
        ...

    @abstractmethod
    async def array_keys(self) -> AsyncIterator[str]:
        """iterate over child array keys"""
        ...

    @abstractmethod
    async def arrays(self) -> AsyncIterator[AsyncArray]:
        """iterate over child arrays"""
        ...

    @abstractmethod
    async def tree(self, expand=False, level=None) -> Any:  # TODO: type this later
        ...

    @abstractmethod
    async def create_group(self, name: str, **kwargs) -> "AsyncGroup":
        ...

    @abstractmethod
    async def create_array(self, name: str, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def empty(self, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def zeros(self, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def ones(self, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def full(self, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def empty_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def zeros_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def ones_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def full_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        ...

    @abstractmethod
    async def move(self, source: str, dest: str) -> None:
        ...

    # TODO / maybes:
    # store_path (rename to path?)
    # visit
    # visitkeys
    # visitvalues


class SyncGroup(BaseGroup, MutableMapping):
    @abstractproperty
    def nchildren(self) -> int:
        ...

    @abstractproperty
    def children(self) -> Iterator:
        ...

    @abstractmethod
    def __contains__(self, child: str) -> bool:
        """check if child exists"""
        ...

    @abstractmethod
    def __getitem__(self, child: str) -> Union[SyncArray, "SyncGroup"]:
        """get child"""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: Union[SyncArray, "SyncGroup"]) -> None:
        """get child"""
        ...

    @abstractmethod
    def group_keys(self) -> AsyncIterator[str]:
        """iterate over child group keys"""
        ...

    @abstractmethod
    def groups(self) -> AsyncIterator["SyncGroup"]:
        """iterate over child groups"""
        ...

    @abstractmethod
    def array_keys(self) -> AsyncIterator[str]:
        """iterate over child array keys"""
        ...

    @abstractmethod
    def arrays(self) -> AsyncIterator[SyncArray]:
        """iterate over child arrays"""
        ...

    @abstractmethod
    def tree(self) -> Any:
        ...

    @abstractmethod
    def create_group(self, name: str, **kwargs) -> "SyncGroup":
        ...

    @abstractmethod
    def create_array(self, name: str, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def empty(self, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def zeros(self, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def ones(self, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def full(self, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def empty_like(self, prototype: SyncArray, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def zeros_like(self, prototype: SyncArray, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def ones_like(self, prototype: SyncArray, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def full_like(self, prototype: SyncArray, **kwargs) -> SyncArray:
        ...

    @abstractmethod
    def move(self, source: str, dest: str) -> None:
        ...
