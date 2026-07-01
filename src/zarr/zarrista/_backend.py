from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import zarrista

from zarr.crud import CrudBackend, ReferenceBackend
from zarr.crud._common import parse_array_metadata
from zarr.storage import LocalStore, ObjectStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


class UnsupportedStoreError(TypeError):
    """Raised when the zarrista backend is handed a store it cannot ingest.

    Unlike the in-tree zarrs bindings, zarrista has no generic Python-store
    callback bridge: it can only operate on its own native store types. This
    backend maps a `LocalStore` to a zarrista `FilesystemStore` (sync path) and
    an obstore-backed `ObjectStore` to zarrista's async API, and rejects
    everything else. Use the ``"reference"`` or ``"zarrs"`` backend for stores
    zarrista cannot ingest.
    """


def _node_path(path: str) -> str:
    """Convert a zarr path (`""`, `"foo/bar"`) to a zarrista node path
    (`"/"`, `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


def _resolve_store(store: Store) -> zarrista.FilesystemStore | Any:
    """Map a zarr `Store` to the zarrista store that backs it, or raise.

    zarrista exposes Rust-native store types rather than a callback into an
    arbitrary zarr `Store`, so only stores with a direct zarrista equivalent are
    supported: a `LocalStore` maps to a zarrista `FilesystemStore` (sync API),
    and an obstore-backed `ObjectStore` unwraps to its inner obstore store
    (async API). Any other store raises `UnsupportedStoreError`.
    """
    if isinstance(store, LocalStore):
        return zarrista.FilesystemStore(str(store.root))
    if isinstance(store, ObjectStore):
        # obstore is importable whenever an ObjectStore instance exists.
        import obstore.store

        if isinstance(store.store, obstore.store.MemoryStore):
            # zarrista rejects memory-backed obstore stores at open time
            # ("expected an async compatible storage object"); fail at the gate
            # instead so the rejection is consistent across all methods.
            raise UnsupportedStoreError(
                "the zarrista backend cannot ingest an ObjectStore backed by an "
                "obstore MemoryStore. Use the 'reference' or 'zarrs' backend for "
                "in-memory stores."
            )
        return store.store
    raise UnsupportedStoreError(
        f"the zarrista backend cannot ingest a {type(store).__name__}; it supports "
        f"LocalStore (mapped to a zarrista FilesystemStore) and obstore-backed "
        f"ObjectStore (mapped to zarrista's async API). Use the 'reference' or "
        f"'zarrs' backend for other stores."
    )


def _check_writable(store: Store) -> None:
    """Raise like `Store._check_writable`: zarrista writes bypass the zarr-level
    `read_only` flag, so enforce it here for parity with the native path."""
    if store.read_only:
        raise ValueError("store was opened in read-only mode and does not support writing")


async def _open_async(target: Any, path: str) -> zarrista.AsyncArray:
    """Open a zarrista `AsyncArray` on an obstore store, translating zarrista's
    store-rejection `TypeError` into `UnsupportedStoreError`."""
    try:
        return await zarrista.AsyncArray.open_async(target, _node_path(path))
    except TypeError as err:
        raise UnsupportedStoreError(
            f"zarrista rejected the obstore store {type(target).__name__}: {err}"
        ) from err


def _native_dtype(metadata: Mapping[str, JSON]) -> np.dtype[Any]:
    """Numpy dtype in native byte order, matching the facade's reassembly."""
    return parse_array_metadata(metadata).dtype.to_native_dtype().newbyteorder("=")


def _to_bytes(decoded: Any, np_dtype: np.dtype[Any]) -> bytes:
    """Reinterpret a zarrista decoded array as C-contiguous native-dtype bytes."""
    arr = np.asarray(decoded.to_numpy(), dtype=np_dtype)
    return np.ascontiguousarray(arr).tobytes()


class ZarristaBackend(CrudBackend):
    """CRUD backend backed by the Rust `zarrs` crate via the `zarrista` package.

    zarrista accelerates the chunk-level I/O paths (`read_chunk`, `read_subset`,
    `write_chunk`, `delete_chunk`). Node metadata documents are written, read,
    listed and deleted with zarr-python's own machinery (delegated to the
    `ReferenceBackend`); zarrista has no "write this exact metadata document"
    primitive for arrays, and these operations are not performance-critical.

    All methods first resolve the store with `_resolve_store`, so the backend
    consistently rejects stores it cannot ingest (raising
    `UnsupportedStoreError`) rather than half-working on them. Two store
    families are ingestable: `LocalStore` (zarrista's sync API on a worker
    thread) and obstore-backed `ObjectStore` (zarrista's async API, natively
    async — no thread offload).

    Limitations:

    - Memory-backed obstore stores are rejected (zarrista cannot ingest them);
      icechunk `Session` support is future work.
    - Zarr V2 is supported only to the extent zarrs supports it (the
      V3-compatible subset); v2 group metadata is written by the reference
      backend here, sidestepping the zarrs v2-group `.zattrs` divergence.
    """

    def __init__(self) -> None:
        # Metadata-document CRUD is store-neutral and identical to the reference
        # backend; reuse it rather than reimplementing it.
        self._metadata = ReferenceBackend()

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        _resolve_store(store)
        await self._metadata.create_array(store, path, metadata, overwrite=overwrite)

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        _resolve_store(store)
        await self._metadata.create_group(store, path, metadata, overwrite=overwrite)

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]:
        _resolve_store(store)
        return await self._metadata.read_metadata(store, path)

    async def delete_node(self, store: Store, path: str) -> None:
        _resolve_store(store)
        await self._metadata.delete_node(store, path)

    async def list_children(self, store: Store, path: str) -> list[tuple[str, dict[str, JSON]]]:
        _resolve_store(store)
        return await self._metadata.list_children(store, path)

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes:
        target = _resolve_store(store)
        np_dtype = _native_dtype(metadata)
        if isinstance(target, zarrista.FilesystemStore):

            def _read() -> bytes:
                array = zarrista.Array.open(target, _node_path(path))
                return _to_bytes(array.retrieve_chunk(list(coords)), np_dtype)

            return await asyncio.to_thread(_read)
        array = await _open_async(target, path)
        return _to_bytes(await array.retrieve_chunk(list(coords)), np_dtype)

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        target = _resolve_store(store)
        np_dtype = _native_dtype(metadata)
        selection = tuple(slice(s, s + length) for s, length in zip(start, shape, strict=True))
        if isinstance(target, zarrista.FilesystemStore):

            def _read() -> bytes:
                array = zarrista.Array.open(target, _node_path(path))
                return _to_bytes(array.retrieve_array_subset(selection), np_dtype)

            return await asyncio.to_thread(_read)
        array = await _open_async(target, path)
        return _to_bytes(await array.retrieve_array_subset(selection), np_dtype)

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        target = _resolve_store(store)
        _check_writable(store)
        if isinstance(target, zarrista.FilesystemStore):

            def _write() -> None:
                array = zarrista.Array.open(target, _node_path(path))
                array.store_chunk(list(coords), zarrista.ArrayBytes(data))

            await asyncio.to_thread(_write)
            return
        array = await _open_async(target, path)
        await array.store_chunk(list(coords), zarrista.ArrayBytes(data))

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        target = _resolve_store(store)
        _check_writable(store)
        if isinstance(target, zarrista.FilesystemStore):

            def _erase() -> None:
                array = zarrista.Array.open(target, _node_path(path))
                array.erase_chunk(list(coords))

            await asyncio.to_thread(_erase)
            return
        array = await _open_async(target, path)
        await array.erase_chunk(list(coords))
