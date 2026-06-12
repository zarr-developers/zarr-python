from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import _collect_aiterator, sync
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.abc.store import Store

# Alias to avoid shadowing the `list` builtin with the `StoreShim.list` method
# in mypy's class-scope name resolution.
_list = builtins.list


class StoreShim:
    """
    Synchronous adapter over an async `Store`, called from Rust worker threads.

    Each method blocks the calling thread by submitting a coroutine to the zarr
    event-loop thread (`zarr.core.sync`). Methods must never be called from the
    zarr event-loop thread itself; the Rust bindings only call them from
    `asyncio.to_thread` worker threads.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._prototype = default_buffer_prototype()

    def get(self, key: str) -> bytes | None:
        buf = sync(self._store.get(key, prototype=self._prototype))
        return None if buf is None else buf.to_bytes()

    def get_range(self, key: str, offset: int, length: int | None) -> bytes | None:
        byte_range = (
            RangeByteRequest(offset, offset + length)
            if length is not None
            else OffsetByteRequest(offset)
        )
        buf = sync(self._store.get(key, prototype=self._prototype, byte_range=byte_range))
        return None if buf is None else buf.to_bytes()

    def get_suffix(self, key: str, suffix: int) -> bytes | None:
        buf = sync(
            self._store.get(key, prototype=self._prototype, byte_range=SuffixByteRequest(suffix))
        )
        return None if buf is None else buf.to_bytes()

    def set(self, key: str, value: bytes) -> None:
        sync(self._store.set(key, self._prototype.buffer.from_bytes(value)))

    def delete(self, key: str) -> None:
        sync(self._store.delete(key))

    def delete_prefix(self, prefix: str) -> None:
        sync(self._store.delete_dir(prefix.rstrip("/")))

    def getsize(self, key: str) -> int | None:
        try:
            return sync(self._store.getsize(key))
        except FileNotFoundError:
            return None

    def getsize_prefix(self, prefix: str) -> int:
        return sync(self._store.getsize_prefix(prefix.rstrip("/")))

    def list(self) -> _list[str]:
        return sorted(sync(_collect_aiterator(self._store.list())))

    def list_prefix(self, prefix: str) -> _list[str]:
        return sorted(sync(_collect_aiterator(self._store.list_prefix(prefix))))

    def list_dir(self, prefix: str) -> tuple[_list[str], _list[str]]:
        """Return `(keys, prefixes)` directly under `prefix`, as zarrs expects:
        full keys, and child prefixes ending in `/`."""
        stripped = prefix.rstrip("/")
        children = sorted(sync(_collect_aiterator(self._store.list_dir(stripped))))
        keys: _list[str] = []
        prefixes: _list[str] = []
        for child in children:
            full = f"{stripped}/{child}" if stripped else child
            if sync(self._store.exists(full)):
                keys.append(full)
            else:
                prefixes.append(full + "/")
        return keys, prefixes


def resolve_store(store: Store) -> StoreShim | dict[str, str]:
    """
    Convert a zarr `Store` into the representation `_zarrs_bindings` expects:
    a config dict for stores with a native Rust implementation, otherwise a
    `StoreShim` that Rust calls back into.
    """
    if isinstance(store, LocalStore) and not store.read_only:
        return {"filesystem": str(store.root)}
    return StoreShim(store)
