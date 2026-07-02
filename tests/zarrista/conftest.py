from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from zarr.abc.store import Store


@pytest.fixture
async def local_store(tmp_path: Path) -> AsyncIterator[Store]:
    """A LocalStore — the only store the zarrista backend can ingest."""
    s = await LocalStore.open(root=tmp_path / "store")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
async def memory_store() -> AsyncIterator[Store]:
    """A MemoryStore — a store the zarrista backend cannot ingest."""
    s = await MemoryStore.open()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
async def object_store(tmp_path: Path) -> AsyncIterator[Store]:
    """A zarr ObjectStore wrapping an obstore LocalStore — ingested via
    zarrista's async API."""
    obstore_store = pytest.importorskip("obstore.store", reason="obstore is not installed")

    from zarr.storage import ObjectStore

    root = tmp_path / "obj"
    root.mkdir()
    s: Store = await ObjectStore.open(obstore_store.LocalStore(str(root)))
    try:
        yield s
    finally:
        s.close()


def array_metadata(**kwargs: Any) -> dict[str, Any]:
    """An array metadata document built via zarr-python itself."""
    params: dict[str, Any] = {
        "shape": (8, 8),
        "chunks": (4, 4),
        "dtype": "uint16",
        "zarr_format": 3,
    } | kwargs
    arr = zarr.create_array(store=MemoryStore(), **params)
    doc = dict(arr.metadata.to_dict())
    if params["zarr_format"] == 2:
        doc.pop("attributes", None)
    return doc


def ramp(shape: tuple[int, ...] = (8, 8), dtype: str = "uint16") -> np.ndarray[Any, np.dtype[Any]]:
    """A deterministic ramp array for round-trip checks."""
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
