"""Translate zarr-python stores into stores Zarrista can consume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.abc.store import Store

_SYNC_SUPPORTED = "LocalStore"
_ASYNC_SUPPORTED = "zarr.storage.ObjectStore (obstore-backed) or an icechunk store"


def translate_store_sync(store: Store) -> Any:
    """zarr store -> zarrista sync store (`zarrista.store.FilesystemStore`)."""
    import zarrista

    if isinstance(store, LocalStore):
        return zarrista.store.FilesystemStore(store.root)
    raise UnsupportedEngineError(
        f"the zarrista sync engine cannot serve a {type(store).__name__}; "
        f"supported: {_SYNC_SUPPORTED}. Note: zarr's MemoryStore lives in the "
        "Python process and cannot be shared with the Rust extension."
    )


def translate_store_async(store: Store) -> Any:
    """zarr store -> zarrista async store (obstore `ObjectStore` or icechunk `Session`)."""
    from zarr.storage import ObjectStore

    if isinstance(store, ObjectStore):
        return store.store  # the underlying obstore instance
    try:
        from icechunk import IcechunkStore

        if isinstance(store, IcechunkStore):
            return store.session
    except ImportError:
        pass
    raise UnsupportedEngineError(
        f"the zarrista async engine cannot serve a {type(store).__name__}; "
        f"supported: {_ASYNC_SUPPORTED}."
    )
