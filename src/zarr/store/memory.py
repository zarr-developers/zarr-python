from __future__ import annotations

from collections.abc import AsyncGenerator, MutableMapping
from typing import TYPE_CHECKING

from zarr.abc.store import Store
from zarr.core.buffer import Buffer
from zarr.core.common import concurrent_map
from zarr.store._utils import _normalize_interval_index

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import AccessModeLiteral


# TODO: this store could easily be extended to wrap any MutableMapping store from v2
# When that is done, the `MemoryStore` will just be a store that wraps a dict.
class MemoryStore(Store):
    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    _store_dict: MutableMapping[str, Buffer]

    def __init__(
        self,
        store_dict: MutableMapping[str, Buffer] | None = None,
        *,
        mode: AccessModeLiteral = "r",
    ):
        super().__init__(mode=mode)
        self._store_dict = store_dict or {}

    async def empty(self) -> bool:
        return not self._store_dict

    async def clear(self) -> None:
        self._store_dict.clear()

    def __str__(self) -> str:
        return f"memory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore({str(self)!r})"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        if not self._is_open:
            await self._open()
        assert isinstance(key, str)
        try:
            value = self._store_dict[key]
            start, length = _normalize_interval_index(value, byte_range)
            return prototype.buffer.from_buffer(value[start : start + length])
        except KeyError:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: list[tuple[str, tuple[int | None, int | None]]],
    ) -> list[Buffer | None]:
        # All the key-ranges arguments goes with the same prototype
        async def _get(key: str, byte_range: tuple[int, int | None]) -> Buffer | None:
            return await self.get(key, prototype=prototype, byte_range=byte_range)

        vals = await concurrent_map(key_ranges, _get, limit=None)
        return vals

    async def exists(self, key: str) -> bool:
        return key in self._store_dict

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        if not self._is_open:
            await self._open()
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(f"Expected Buffer. Got {type(value)}.")

        if byte_range is not None:
            buf = self._store_dict[key]
            buf[byte_range[0] : byte_range[1]] = value
            self._store_dict[key] = buf
        else:
            self._store_dict[key] = value

    async def delete(self, key: str) -> None:
        self._check_writable()
        try:
            del self._store_dict[key]
        except KeyError:
            pass  # Q(JH): why not raise?

    async def set_partial_values(self, key_start_values: list[tuple[str, int, bytes]]) -> None:
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
            keys_unique = set(k.split("/")[0] for k in self._store_dict.keys())
            for key in keys_unique:
                yield key
        else:
            for key in self._store_dict:
                if key.startswith(prefix + "/") and key != prefix:
                    yield key.removeprefix(prefix + "/").split("/")[0]
