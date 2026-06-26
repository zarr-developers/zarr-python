from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from zarr.abc.store import Store


@pytest.fixture(params=["memory", "local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    """A writable store: MemoryStore exercises the generic Python-callback bridge,
    LocalStore exercises the native zarrs filesystem store."""
    s: Store
    if request.param == "memory":
        s = await MemoryStore.open()
    else:
        s = await LocalStore.open(root=tmp_path / "store")
    try:
        yield s
    finally:
        s.close()


def array_metadata(**kwargs: Any) -> dict[str, Any]:
    """Build an array metadata document using zarr-python itself, so the
    documents fed to zarrs always match what zarr-python would write."""
    params: dict[str, Any] = {
        "shape": (8, 8),
        "chunks": (4, 4),
        "dtype": "uint16",
        "zarr_format": 3,
    } | kwargs
    arr = zarr.create_array(store=MemoryStore(), **params)
    doc = dict(arr.metadata.to_dict())
    if params["zarr_format"] == 2:
        # v2 attributes live in .zattrs, not in the .zarray document
        doc.pop("attributes", None)
    return doc
