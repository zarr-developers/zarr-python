from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, NamedTuple, Protocol, runtime_checkable

from typing_extensions import Self

from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import AccessModeLiteral, BytesLike

__all__ = ["Store", "AccessMode", "ByteGetter", "ByteSetter", "set_or_delete"]


class AccessMode(NamedTuple):
    str: AccessModeLiteral
    readonly: bool
    overwrite: bool
    create: bool
    update: bool

    @classmethod
    def from_literal(cls, mode: AccessModeLiteral) -> Self:
        if mode in ("r", "r+", "a", "w", "w-"):
            return cls(
                str=mode,
                readonly=mode == "r",
                overwrite=mode == "w",
                create=mode in ("a", "w", "w-"),
                update=mode in ("r+", "a"),
            )
        raise ValueError("mode must be one of 'r', 'r+', 'w', 'w-', 'a'")


class Store(ABC):
    _mode: AccessMode
    _is_open: bool

    def __init__(self, mode: AccessModeLiteral = "r", *args: Any, **kwargs: Any):
        self._is_open = False
        self._mode = AccessMode.from_literal(mode)

    @classmethod
    async def open(cls, *args: Any, **kwargs: Any) -> Self:
        store = cls(*args, **kwargs)
        await store._open()
        return store

    def __enter__(self) -> Self:
        """Enter a context manager that will close the store upon exiting."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the store."""
        self.close()

    async def _open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")
        if not await self.empty():
            if self.mode.update or self.mode.readonly:
                pass
            elif self.mode.overwrite:
                await self.clear()
            else:
                raise FileExistsError("Store already exists")
        self._is_open = True

    async def _ensure_open(self) -> None:
        if not self._is_open:
            await self._open()

    @abstractmethod
    async def empty(self) -> bool: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @property
    def mode(self) -> AccessMode:
        """Access mode of the store."""
        return self._mode

    def _check_writable(self) -> None:
        if self.mode.readonly:
            raise ValueError("store mode does not support writing")

    @abstractmethod
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        byte_range : tuple[int, Optional[int]], optional

        Returns
        -------
        Buffer
        """
        ...

    @abstractmethod
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: list[tuple[str, tuple[int | None, int | None]]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        key_ranges : list[tuple[str, tuple[int, int]]]
            Ordered set of key, range pairs, a key may occur multiple times with different ranges

        Returns
        -------
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
    async def set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        ...

    @property
    @abstractmethod
    def supports_deletes(self) -> bool:
        """Does the store support deletes?"""
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

    def close(self) -> None:
        """Close the store."""
        self._is_open = False


@runtime_checkable
class ByteGetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: tuple[int, int | None] | None = None
    ) -> Buffer | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: tuple[int, int | None] | None = None
    ) -> Buffer | None: ...

    async def set(self, value: Buffer, byte_range: tuple[int, int] | None = None) -> None: ...

    async def delete(self) -> None: ...


async def set_or_delete(byte_setter: ByteSetter, value: Buffer | None) -> None:
    if value is None:
        await byte_setter.delete()
    else:
        await byte_setter.set(value)
