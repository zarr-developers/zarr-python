from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip(
    "_zarrs_bindings", reason="zarrs-bindings is not installed", exc_type=ImportError
)

from zarr.storage import LocalStore, MemoryStore
from zarr.zarrs._bridge import StoreShim, resolve_store

if TYPE_CHECKING:
    from pathlib import Path


def test_shim_get_set_delete() -> None:
    shim = StoreShim(MemoryStore())
    assert shim.get("a/b") is None
    shim.set("a/b", b"xyz")
    assert shim.get("a/b") == b"xyz"
    assert shim.get_range("a/b", 1, 1) == b"y"
    assert shim.get_range("a/b", 1, None) == b"yz"
    assert shim.get_suffix("a/b", 2) == b"yz"
    assert shim.getsize("a/b") == 3
    assert shim.getsize("missing") is None
    assert shim.get_range("missing", 0, 1) is None
    assert shim.get_suffix("missing", 1) is None
    shim.delete("a/b")
    assert shim.get("a/b") is None


def test_shim_listing() -> None:
    shim = StoreShim(MemoryStore())
    shim.set("zarr.json", b"{}")
    shim.set("a/zarr.json", b"{}")
    shim.set("a/c/0/0", b"\x00")
    assert shim.list() == ["a/c/0/0", "a/zarr.json", "zarr.json"]
    assert shim.list_prefix("a/") == ["a/c/0/0", "a/zarr.json"]
    assert shim.list_dir("a/") == (["a/zarr.json"], ["a/c/"])
    assert shim.list_dir("") == (["zarr.json"], ["a/"])
    assert shim.getsize_prefix("a/") == 3
    shim.delete_prefix("a/")
    assert shim.list() == ["zarr.json"]


def test_resolve_store(tmp_path: Path) -> None:
    local = LocalStore(tmp_path)
    assert resolve_store(local) == {"filesystem": str(tmp_path)}
    # read-only LocalStore must go through the shim so writes are rejected in Python
    assert isinstance(resolve_store(LocalStore(tmp_path, read_only=True)), StoreShim)
    assert isinstance(resolve_store(MemoryStore()), StoreShim)
