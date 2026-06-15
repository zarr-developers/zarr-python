from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip(
    "_zarrs_bindings", reason="zarrs-bindings is not installed", exc_type=ImportError
)

import _zarrs_bindings as zb

import zarr
from zarr.storage import LocalStore, MemoryStore
from zarr.zarrs import decode_chunk, encode_chunk

if TYPE_CHECKING:
    from pathlib import Path


def _meta(store: Any, name: str = "a") -> dict[str, Any]:
    arr = zarr.create_array(store=store, name=name, shape=(8, 8), chunks=(4, 4), dtype="uint16")
    arr[:, :] = np.arange(64, dtype="uint16").reshape(8, 8)
    return dict(arr.metadata.to_dict())


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    zb.clear_array_cache()


async def test_localstore_populates_cache(tmp_path: Path) -> None:
    store = await LocalStore.open(root=tmp_path / "s")
    meta = _meta(store)
    assert zb.array_cache_len() == 0
    await decode_chunk(meta, store, "a", (0, 0))
    assert zb.array_cache_len() == 1
    # second op on the SAME array reuses the entry, does not grow the cache
    await decode_chunk(meta, store, "a", (1, 1))
    assert zb.array_cache_len() == 1


async def test_memorystore_is_not_cached() -> None:
    store = MemoryStore()
    meta = _meta(store)
    await decode_chunk(meta, store, "a", (0, 0))
    assert zb.array_cache_len() == 0


async def test_distinct_metadata_distinct_entries(tmp_path: Path) -> None:
    store = await LocalStore.open(root=tmp_path / "s")
    meta_a = _meta(store, "a")
    meta_b = _meta(store, "b")
    await decode_chunk(meta_a, store, "a", (0, 0))
    await decode_chunk(meta_b, store, "b", (0, 0))
    assert zb.array_cache_len() == 2


async def test_cache_keyed_on_root_not_just_metadata(tmp_path: Path) -> None:
    # two stores at different roots, identical metadata + path, different data.
    # A correct cache (keyed on root) must return each store's own data.
    s1 = await LocalStore.open(root=tmp_path / "s1")
    s2 = await LocalStore.open(root=tmp_path / "s2")
    a1 = zarr.create_array(store=s1, name="a", shape=(4, 4), chunks=(4, 4), dtype="uint16")
    a1[:, :] = 1
    a2 = zarr.create_array(store=s2, name="a", shape=(4, 4), chunks=(4, 4), dtype="uint16")
    a2[:, :] = 2
    meta = dict(a1.metadata.to_dict())  # identical metadata document
    out1 = await decode_chunk(meta, s1, "a", (0, 0))
    out2 = await decode_chunk(meta, s2, "a", (0, 0))
    np.testing.assert_array_equal(out1, np.full((4, 4), 1, dtype="uint16"))
    np.testing.assert_array_equal(out2, np.full((4, 4), 2, dtype="uint16"))
    assert zb.array_cache_len() == 2


async def test_cache_reflects_writes_through_store(tmp_path: Path) -> None:
    # after the Array is cached, a write via the cached Array must be visible to
    # a subsequent read (proves the cache does not stale-cache chunk data)
    store = await LocalStore.open(root=tmp_path / "s")
    meta = _meta(store)
    await decode_chunk(meta, store, "a", (0, 0))  # caches the Array
    new = np.full((4, 4), 99, dtype="uint16")
    await encode_chunk(meta, store, "a", (0, 0), new)  # write via (cached) Array
    out = await decode_chunk(meta, store, "a", (0, 0))
    np.testing.assert_array_equal(out, new)
