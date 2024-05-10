from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import List, MutableMapping, Optional, Tuple

from zarr.abc.store import Store
from zarr.common import BytesLike, concurrent_map


# TODO: this store could easily be extended to wrap any MutuableMapping store from v2
# When that is done, the `MemoryStore` will just be a store that wraps a dict.
class MemoryStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    _store_dict: MutableMapping[str, bytes]

    def __init__(self, store_dict: Optional[MutableMapping[str, bytes]] = None):
        self._store_dict = store_dict or {}

    def __str__(self) -> str:
        return f"memory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore({repr(str(self))})"

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        try:
            value = self._store_dict[key]
            if byte_range is not None:
                value = value[byte_range[0] : byte_range[1]]
            return value
        except KeyError:
            return None

    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[Optional[BytesLike]]:
        vals = await concurrent_map(key_ranges, self.get, limit=None)
        return vals

    async def exists(self, key: str) -> bool:
        return key in self._store_dict

    async def set(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected BytesLike. Got {type(value)}.")

        if byte_range is not None:
            buf = bytearray(self._store_dict[key])
            buf[byte_range[0] : byte_range[1]] = value
            self._store_dict[key] = buf
        else:
            self._store_dict[key] = value

    async def delete(self, key: str) -> None:
        try:
            del self._store_dict[key]
        except KeyError:
            pass  # Q(JH): why not raise?

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        if prefix.endswith("/"):
            prefix = prefix[:-1]

        if prefix == "":
            for key in self._store_dict:
                yield key.split("/", maxsplit=1)[0]
        else:
            for key in self._store_dict:
                if key.startswith(prefix + "/") and key != prefix:
                    yield key.strip(prefix + "/").split("/")[0]
