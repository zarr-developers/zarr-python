from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

from object_store import ObjectStore as _ObjectStore
from object_store import Path as ObjectPath

from zarr.v3.abc.store import Store


class ObjectStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    store: _ObjectStore

    def init(self, store: _ObjectStore):
        self.store = store

    def __str__(self) -> str:
        return f"object://{self.store}"

    def __repr__(self) -> str:
        return f"ObjectStore({repr(str(self))})"

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[bytes]:
        if byte_range is None:
            return await self.store.get_async(ObjectPath(key))

        start, end = byte_range
        if end is None:
            # Have to wrap a separate object-store function to support this
            raise NotImplementedError

        return await self.store.get_range_async(ObjectPath(key), start, end - start)

    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[bytes]:
        # TODO: use rust-based concurrency inside object-store
        futs = [self.get(key, byte_range=byte_range) for (key, byte_range) in key_ranges]

        # Seems like a weird type match where `get()` returns `Optional[bytes]` but
        # `get_partial_values` is non-optional?
        return await asyncio.gather(*futs)  # type: ignore

    async def exists(self, key: str) -> bool:
        try:
            _ = await self.store.head_async(ObjectPath(key))
            return True
        except FileNotFoundError:
            return False

    async def set(self, key: str, value: bytes) -> None:
        await self.store.put_async(ObjectPath(key), value)

    async def delete(self, key: str) -> None:
        await self.store.delete_async(ObjectPath(key))

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def list(self) -> List[str]:
        objects = await self.store.list_async(None)
        return [str(obj.location) for obj in objects]

    async def list_prefix(self, prefix: str) -> List[str]:
        objects = await self.store.list_async(ObjectPath(prefix))
        return [str(obj.location) for obj in objects]

    async def list_dir(self, prefix: str) -> List[str]:
        list_result = await self.store.list_with_delimiter_async(ObjectPath(prefix))
        common_prefixes = set(list_result.common_prefixes)
        return [str(obj.location) for obj in list_result.objects if obj not in common_prefixes]
