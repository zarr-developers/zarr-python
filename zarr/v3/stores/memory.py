from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

from zarr.v3.common import BytesLike
from zarr.v3.abc.store import WriteListStore
from zarr.v3.stores.core import BaseStore


# TODO: this store could easily be extended to wrap any MutuableMapping store from v2
# When that is done, the `MemoryStore` will just be a store that wraps a dict.
class MemoryStore(WriteListStore, BaseStore):
    supports_partial_writes = True
    store_dict: MutableMapping[str, bytes]

    def __init__(self, store_dict: Optional[MutableMapping[str, bytes]] = None):
        self.store_dict = store_dict or {}

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        try:
            value = self.store_dict[key]
            if byte_range is not None:
                value = value[byte_range[0] : byte_range[1]]
            return value
        except KeyError:
            return None

    async def get_partial_values(self, key_ranges: List[Tuple[str, int]]) -> bytes:
        raise NotImplementedError

    async def set(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)

        if byte_range is not None:
            buf = bytearray(self.store_dict[key])
            buf[byte_range[0] : byte_range[1]] = value
            self.store_dict[key] = buf
        else:
            self.store_dict[key] = value

    async def delete(self, key: str) -> None:
        try:
            del self.store_dict[key]
        except KeyError:
            pass

    async def exists(self, key: str) -> bool:
        return key in self.store_dict

    def __str__(self) -> str:
        return f"memory://{id(self.store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore({repr(str(self))})"
