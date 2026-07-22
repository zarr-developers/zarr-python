from __future__ import annotations

import gc
from typing import Any

import pytest

import zarr
from zarr.core.engine import (
    DefaultArrayEngine,
    DefaultAsyncArrayEngine,
    resolve_async_engine,
    resolve_sync_engine,
)
from zarr.core.engine._resolve import _hierarchy_cache
from zarr.storage import MemoryStore


def _array() -> zarr.Array[Any]:
    return zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")


def test_resolution_combinations() -> None:
    z = _array()
    store = z.store
    path = z.path
    meta = z.async_array.metadata

    # None and "default" produce default engines
    for spec in (None, "default"):
        assert isinstance(
            resolve_async_engine(spec, store=store, path=path, metadata=meta),
            DefaultAsyncArrayEngine,
        )
        assert isinstance(
            resolve_sync_engine(spec, store=store, path=path, metadata=meta),
            DefaultArrayEngine,
        )

    # instances pass through untouched
    inst = resolve_sync_engine(None, store=store, path=path, metadata=meta)
    assert resolve_sync_engine(inst, store=store, path=path, metadata=meta) is inst

    # engines minted from the same store share a hierarchy engine
    e1 = resolve_async_engine(None, store=store, path=path, metadata=meta)
    e2 = resolve_async_engine(None, store=store, path="other", metadata=meta)
    assert e1.store_path.store is e2.store_path.store  # type: ignore[attr-defined]


def test_hierarchy_cache_evicts_when_store_and_engines_are_unreferenced() -> None:
    # A dedicated store (not shared with other tests) so the cache starts clean
    # for this key. Measure the baseline *before* creating the array: creating it
    # resolves the array's own engine, which already populates the (default,
    # async, id(store)) cache entry that `e1`/`e2` then reuse.
    store = MemoryStore()
    # Collect any hierarchies left unreferenced by earlier tests first, so the
    # baseline reflects only entries kept alive by still-referenced arrays.
    gc.collect()
    before = len(_hierarchy_cache)
    z = zarr.create_array(store, shape=(4,), chunks=(2,), dtype="int8")
    path = z.path
    meta = z.async_array.metadata

    e1 = resolve_async_engine(None, store=store, path=path, metadata=meta)
    e2 = resolve_async_engine(None, store=store, path="other", metadata=meta)
    assert len(_hierarchy_cache) == before + 1

    # engines minted while the store is in concurrent use share one hierarchy
    assert (
        e1._resolve_hierarchy_keepalive  # type: ignore[attr-defined]
        is e2._resolve_hierarchy_keepalive  # type: ignore[attr-defined]
    )

    del z, e1, e2, store
    gc.collect()
    assert len(_hierarchy_cache) == before


def test_unknown_name_raises() -> None:
    z = _array()
    with pytest.raises(ValueError, match="unknown engine"):
        resolve_async_engine(
            "bogus",  # type: ignore[arg-type]
            store=z.store,
            path=z.path,
            metadata=z.async_array.metadata,
        )


def test_zarrista_missing_raises_import_error() -> None:
    try:
        import zarrista  # noqa: F401

        pytest.skip("zarrista installed; missing-module error not testable")
    except ImportError:
        pass
    z = _array()
    with pytest.raises(ImportError, match="zarrista"):
        resolve_async_engine(
            "zarrista", store=z.store, path=z.path, metadata=z.async_array.metadata
        )
