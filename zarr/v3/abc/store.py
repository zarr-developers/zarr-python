from abc import abstractmethod, ABC

from typing import List, Tuple


class Store(ABC):
    pass


class ReadStore(Store):
    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str

        Returns
        -------
        bytes
        """
        ...

    @abstractmethod
    async def get_partial_values(self, key_ranges: List[Tuple[str, int]]) -> bytes:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        key_ranges : list[tuple[str, int]]
            Ordered set of key, range pairs, a key may occur multiple times with different ranges

        Returns
        -------
        list[bytes]
            list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        ...


class WriteStore(ReadStore):
    @abstractmethod
    async def set(self, key: str, value: bytes) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : bytes
        """
        ...

    @abstractmethod
    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        """Store values at a given key, starting at byte range_start.

        Parameters
        ----------
        key_start_values : list[tuple[str, int, bytes]]
            set of key, range_start, values triples, a key may occur multiple times with different
            range_starts, range_starts (considering the length of the respective values) must not
            specify overlapping ranges for the same key
        """
        ...


class ListMixin:
    @abstractmethod
    async def list(self) -> List[str]:
        """Retrieve all keys in the store.

        Returns
        -------
        list[str]
        """
        ...

    @abstractmethod
    async def list_prefix(self, prefix: str) -> List[str]:
        """Retrieve all keys in the store.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        list[str]
        """
        ...

    @abstractmethod
    async def list_dir(self, prefix: str) -> List[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        list[str]
        """
        ...


class ReadListStore(ReadStore, ListMixin):
    pass


class WriteListStore(WriteStore, ListMixin):
    pass
