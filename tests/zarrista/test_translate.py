from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("zarrista")

import zarrista

from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore, MemoryStore
from zarr.zarrista._translate import translate_store_async, translate_store_sync

if TYPE_CHECKING:
    from pathlib import Path


def test_local_store_translates_to_filesystem_store(tmp_path: Path) -> None:
    zs = translate_store_sync(LocalStore(tmp_path))
    assert isinstance(zs, zarrista.store.FilesystemStore)


def test_memory_store_rejected_sync(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedEngineError, match="MemoryStore"):
        translate_store_sync(MemoryStore())


def test_local_store_rejected_async(tmp_path: Path) -> None:
    # async side wants obstore/icechunk; LocalStore is sync-only in v1
    with pytest.raises(UnsupportedEngineError):
        translate_store_async(LocalStore(tmp_path))


def test_object_store_translates_to_obstore(tmp_path: Path) -> None:
    obstore = pytest.importorskip("obstore")
    from zarr.storage import ObjectStore

    inner = obstore.store.LocalStore(prefix=str(tmp_path))
    zstore = ObjectStore(inner)
    assert translate_store_async(zstore) is inner
