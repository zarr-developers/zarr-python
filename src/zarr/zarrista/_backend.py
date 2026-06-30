from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import zarrista

from zarr.crud import ReferenceBackend
from zarr.crud._common import parse_array_metadata
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


class UnsupportedStoreError(TypeError):
    """Raised when the zarrista backend is handed a store it cannot ingest.

    Unlike the in-tree zarrs bindings, zarrista has no generic Python-store
    callback bridge: it can only operate on its own native store types. This
    backend currently maps a `LocalStore` to a zarrista `FilesystemStore` and
    rejects everything else. Use the ``"reference"`` or ``"zarrs"`` backend for
    stores zarrista cannot ingest.
    """


def _node_path(path: str) -> str:
    """Convert a zarr path (`""`, `"foo/bar"`) to a zarrista node path
    (`"/"`, `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


def _resolve_store(store: Store) -> zarrista.FilesystemStore:
    """Map a zarr `Store` to the zarrista store that backs it, or raise.

    zarrista exposes Rust-native store types rather than a callback into an
    arbitrary zarr `Store`, so only stores with a direct zarrista equivalent are
    supported. A `LocalStore` maps to a zarrista `FilesystemStore`; any other
    store raises `UnsupportedStoreError`.
    """
    if isinstance(store, LocalStore):
        return zarrista.FilesystemStore(str(store.root))
    raise UnsupportedStoreError(
        f"the zarrista backend cannot ingest a {type(store).__name__}; it supports "
        f"only LocalStore (mapped to a zarrista FilesystemStore). Use the "
        f"'reference' or 'zarrs' backend for other stores."
    )


def _native_dtype(metadata: Mapping[str, JSON]) -> np.dtype[Any]:
    """Numpy dtype in native byte order, matching the facade's reassembly."""
    return parse_array_metadata(metadata).dtype.to_native_dtype().newbyteorder("=")


def _to_bytes(decoded: Any, np_dtype: np.dtype[Any]) -> bytes:
    """Reinterpret a zarrista decoded array as C-contiguous native-dtype bytes."""
    arr = np.asarray(decoded.to_numpy(), dtype=np_dtype)
    return np.ascontiguousarray(arr).tobytes()


class ZarristaBackend:
    """CRUD backend backed by the Rust `zarrs` crate via the `zarrista` package.

    zarrista accelerates the chunk-level I/O paths (`read_chunk`, `read_subset`,
    `write_chunk`, `delete_chunk`). Node metadata documents are written, read,
    listed and deleted with zarr-python's own machinery (delegated to the
    `ReferenceBackend`); zarrista has no "write this exact metadata document"
    primitive for arrays, and these operations are not performance-critical.

    All methods first resolve the store with `_resolve_store`, so the backend
    consistently rejects stores it cannot ingest (raising
    `UnsupportedStoreError`) rather than half-working on them.

    Limitations:

    - Only `LocalStore` is ingestable today (maps to a zarrista
      `FilesystemStore`). obstore- and icechunk-backed stores are future work.
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
        zstore = _resolve_store(store)
        np_dtype = _native_dtype(metadata)

        def _read() -> bytes:
            array = zarrista.Array.open(zstore, _node_path(path))
            return _to_bytes(array.retrieve_chunk(list(coords)), np_dtype)

        return await asyncio.to_thread(_read)

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        zstore = _resolve_store(store)
        np_dtype = _native_dtype(metadata)
        selection = tuple(slice(s, s + length) for s, length in zip(start, shape, strict=True))

        def _read() -> bytes:
            array = zarrista.Array.open(zstore, _node_path(path))
            return _to_bytes(array.retrieve_array_subset(selection), np_dtype)

        return await asyncio.to_thread(_read)

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        zstore = _resolve_store(store)

        def _write() -> None:
            array = zarrista.Array.open(zstore, _node_path(path))
            array.store_chunk(list(coords), zarrista.ArrayBytes(data))

        await asyncio.to_thread(_write)

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        zstore = _resolve_store(store)

        def _erase() -> None:
            array = zarrista.Array.open(zstore, _node_path(path))
            array.erase_chunk(list(coords))

        await asyncio.to_thread(_erase)
