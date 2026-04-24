# tests/test_store/test_protocols.py
"""Runtime and static conformance tests for zarr.storage._protocols.SupportsGetRanges."""

from __future__ import annotations

import pytest

from zarr.storage._protocols import SupportsGetRanges


def test_fsspec_store_satisfies_supports_get_ranges() -> None:
    pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem

    from zarr.storage import FsspecStore
    from zarr.storage._fsspec import _make_async

    fs = MemoryFileSystem()
    fs.store.clear()
    store = FsspecStore(fs=_make_async(fs), path="/x")
    assert isinstance(store, SupportsGetRanges)


def test_memory_store_does_not_satisfy_supports_get_ranges() -> None:
    """Sanity check: stores that don't implement get_ranges shouldn't satisfy the protocol."""
    from zarr.storage import MemoryStore

    store = MemoryStore()
    assert not isinstance(store, SupportsGetRanges)


def test_type_assignment_at_module_level() -> None:
    """Smoke-test the module-level `_: type[SupportsGetRanges] = FsspecStore`.

    If this runs without error the module imported cleanly; the static check is in mypy.
    """
    pytest.importorskip("fsspec")
    from zarr.storage import _fsspec  # noqa: F401
