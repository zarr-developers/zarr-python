from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, MutableMapping, Optional, Tuple, Union, List

from zarr.v3.common import BytesLike, concurrent_map
from zarr.v3.abc.store import Store, ReadStore, WriteStore

if TYPE_CHECKING:
    from upath import UPath


def _dereference_path(root: str, path: str) -> str:
    assert isinstance(root, str)
    assert isinstance(path, str)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root != "" else path
    path = path.rstrip("/")
    return path


class StorePath:
    store: Store
    path: str

    def __init__(self, store: Store, path: Optional[str] = None):
        self.store = store
        self.path = path or ""

    @classmethod
    def from_path(cls, pth: Path) -> StorePath:
        return cls(Store.from_path(pth))

    async def get(
        self, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        return await self.store.get(self.path, byte_range)

    async def set(
        self, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        await self.store.set(self.path, value, byte_range)

    async def delete(self) -> None:
        await self.store.delete(self.path)

    async def exists(self) -> bool:
        return await self.store.exists(self.path)

    def __truediv__(self, other: str) -> StorePath:
        return self.__class__(self.store, _dereference_path(self.path, other))

    def __str__(self) -> str:
        return _dereference_path(str(self.store), self.path)

    def __repr__(self) -> str:
        return f"StorePath({self.store.__class__.__name__}, {repr(str(self))})"

    def __eq__(self, other: Any) -> bool:
        try:
            if self.store == other.store and self.path == other.path:
                return True
        except Exception:
            pass
        return False

class BaseStore(Store):
    supports_partial_writes = False

    # Does this really need to be on the Store? Could just
    # be a convenience function
    @classmethod
    def from_path(cls, pth: Path) -> Store:
        try:
            from upath import UPath
            from upath.implementations.local import PosixUPath, WindowsUPath

            if isinstance(pth, UPath) and not isinstance(pth, (PosixUPath, WindowsUPath)):
                storage_options = pth._kwargs.copy()
                storage_options.pop("_url", None)
                return RemoteStore(str(pth), **storage_options)
        except ImportError:
            pass

        return LocalStore(pth)

    # async def set(
    #     self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    # ) -> None:
    #     raise NotImplementedError

    # async def delete(self, key: str) -> None:
    #     raise NotImplementedError

    # async def exists(self, key: str) -> bool:
    #     raise NotImplementedError

    # def __truediv__(self, other: str) -> StorePath:
    #     return StorePath(self, other)

class BaseReadStore(ReadStore)

    async def get_partial_values(self, key_ranges: List[Tuple[str, Tuple[int, int]]]) -> List[bytes]:
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
        # fallback for stores that don't support partial reads
        async def _get_then_slice(key: str, key_range: tuple[int, int]) -> bytes:
            value = await self.get(key)
            return value[key_range[0]:key_range[1]]

        return await concurrent_map(
            key_ranges,
            _get_then_slice,
            limit=None # TODO: wire this to config
        )

    # async def multi_get(
    #     self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    # ) -> List[Optional[BytesLike]]:
    #     return await asyncio.gather(*[self.get(key, byte_range) for key, byte_range in keys])


class BaseWriteStore(WriteStore):







StoreLike = Union[BaseStore, StorePath, Path, str]


def make_store_path(store_like: StoreLike) -> StorePath:
    if isinstance(store_like, StorePath):
        return store_like
    elif isinstance(store_like, BaseStore):
        return StorePath(store_like)
    elif isinstance(store_like, Path):
        return StorePath(Store.from_path(store_like))
    elif isinstance(store_like, str):
        try:
            from upath import UPath

            return StorePath(Store.from_path(UPath(store_like)))
        except ImportError:
            return StorePath(LocalStore(Path(store_like)))
    raise TypeError
