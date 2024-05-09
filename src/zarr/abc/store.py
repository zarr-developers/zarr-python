from abc import abstractmethod, ABC

from collections.abc import AsyncGenerator
from typing import List, Protocol, Tuple, Optional, runtime_checkable

from zarr.common import BytesLike


class Store(ABC):
    @abstractmethod
    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[bytes]:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        byte_range : tuple[int, Optional[int]], optional

        Returns
        -------
        bytes
        """
        ...

    @abstractmethod
    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[Optional[bytes]]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        key_ranges : list[tuple[str, tuple[int, int]]]
            Ordered set of key, range pairs, a key may occur multiple times with different ranges

        Returns
        -------
        list[bytes]
            list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        ...

    @property
    @abstractmethod
    def supports_writes(self) -> bool:
        """Does the store support writes?"""
        ...

    @abstractmethod
    async def set(self, key: str, value: BytesLike) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : BytesLike
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from the store

        Parameters
        ----------
        key : str
        """
        ...

    @property
    @abstractmethod
    def supports_partial_writes(self) -> bool:
        """Does the store support partial writes?"""
        ...

    @abstractmethod
    async def set_partial_values(self, key_start_values: list[tuple[str, int, BytesLike]]) -> None:
        """Store values at a given key, starting at byte range_start.

        Parameters
        ----------
        key_start_values : list[tuple[str, int, BytesLike]]
            set of key, range_start, values triples, a key may occur multiple times with different
            range_starts, range_starts (considering the length of the respective values) must not
            specify overlapping ranges for the same key
        """
        ...

    @property
    @abstractmethod
    def supports_listing(self) -> bool:
        """Does the store support listing?"""
        ...

    @abstractmethod
    def list(self) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncGenerator[str, None]
        """
        ...

    @abstractmethod
    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store with a given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncGenerator[str, None]
        """
        ...

    @abstractmethod
    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncGenerator[str, None]
        """
        ...


@runtime_checkable
class ByteGetter(Protocol):
    async def get(
        self, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(
        self, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]: ...

    async def set(self, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None) -> None: ...

    async def delete(self) -> None: ...


async def set_or_delete(byte_setter: ByteSetter, value: BytesLike | None) -> None:
    if value is None:
        await byte_setter.delete()
    else:
        await byte_setter.set(value)
