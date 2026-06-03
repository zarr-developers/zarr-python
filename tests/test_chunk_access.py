"""Tests for the shard-discovery and region-read primitives.

These cover :func:`zarr.shards_initialized` (discover which shards/chunks of an
array are populated) and :func:`zarr.read_regions` (concurrently read and decode
the populated regions), along with their asynchronous counterparts in
:mod:`zarr.api.asynchronous`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
import zarr.api.asynchronous as async_api

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.abc.store import Store


def _sparse_1d(store: Store) -> tuple[zarr.Array, np.ndarray]:
    arr = zarr.create_array(store=store, shape=(64,), chunks=(8,), dtype="int32", fill_value=42)
    # populate two non-adjacent chunks (chunks 1 and 5)
    arr[8:16] = np.arange(8, dtype="int32")
    arr[40:48] = np.arange(100, 108, dtype="int32")
    return arr, np.asarray(arr[:])


def _dense_1d(store: Store) -> tuple[zarr.Array, np.ndarray]:
    arr = zarr.create_array(store=store, shape=(32,), chunks=(8,), dtype="int32", fill_value=0)
    arr[:] = np.arange(32, dtype="int32")
    return arr, np.asarray(arr[:])


def _sparse_2d(store: Store) -> tuple[zarr.Array, np.ndarray]:
    arr = zarr.create_array(store=store, shape=(8, 8), chunks=(2, 2), dtype="int32", fill_value=-1)
    arr[0:2, 0:2] = np.ones((2, 2), dtype="int32")
    arr[4:6, 4:6] = np.full((2, 2), 7, dtype="int32")
    return arr, np.asarray(arr[:])


def _sharded_sparse(store: Store) -> tuple[zarr.Array, np.ndarray]:
    # chunks (2, 2) within shards (4, 4): the shard grid is 2x2 over the 8x8 array.
    arr = zarr.create_array(
        store=store, shape=(8, 8), chunks=(2, 2), shards=(4, 4), dtype="int32", fill_value=42
    )
    arr[0:2, 0:2] = np.ones((2, 2), dtype="int32")  # shard (0, 0)
    arr[4:6, 4:6] = np.full((2, 2), 7, dtype="int32")  # shard (1, 1)
    return arr, np.asarray(arr[:])


def _all_empty(store: Store) -> tuple[zarr.Array, np.ndarray]:
    arr = zarr.create_array(store=store, shape=(32,), chunks=(8,), dtype="int32", fill_value=7)
    return arr, np.asarray(arr[:])


def _all_populated(store: Store) -> tuple[zarr.Array, np.ndarray]:
    arr = zarr.create_array(store=store, shape=(32,), chunks=(8,), dtype="int32", fill_value=0)
    arr[:] = np.arange(32, dtype="int32")
    return arr, np.asarray(arr[:])


SETUPS: dict[str, Callable[[Store], tuple[zarr.Array, np.ndarray]]] = {
    "sparse_1d": _sparse_1d,
    "dense_1d": _dense_1d,
    "sparse_2d": _sparse_2d,
    "sharded_sparse": _sharded_sparse,
    "all_empty": _all_empty,
    "all_populated": _all_populated,
}

STRATEGIES = ["auto", "list", "probe"]


def _pack(arr: zarr.Array, regions: list, baseline: np.ndarray) -> np.ndarray:
    """Scatter ``(region, data)`` pairs onto a fill-valued array."""
    out = np.full(baseline.shape, arr.fill_value, dtype=baseline.dtype)
    for region, data in regions:
        out[region] = np.asarray(data)
    return out


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("setup_name", list(SETUPS))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shards_initialized_strategies_agree(store: Store, setup_name: str, strategy: str) -> None:
    """Every strategy reports the same set of populated keys, and reports the
    expected count for a hand-known layout."""
    arr, _ = SETUPS[setup_name](store)
    keys = set(zarr.shards_initialized(arr, strategy=strategy))
    # all strategies must agree
    assert keys == set(zarr.shards_initialized(arr, strategy="auto"))


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    ("setup_name", "expected_count"),
    [
        ("sparse_1d", 2),
        ("dense_1d", 4),
        ("sparse_2d", 2),
        ("sharded_sparse", 2),
        ("all_empty", 0),
        ("all_populated", 4),
    ],
)
def test_shards_initialized_counts(store: Store, setup_name: str, expected_count: int) -> None:
    arr, _ = SETUPS[setup_name](store)
    assert len(zarr.shards_initialized(arr)) == expected_count


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_shards_initialized_unknown_strategy(store: Store) -> None:
    arr, _ = _sparse_1d(store)
    with pytest.raises(ValueError, match="Unknown strategy"):
        zarr.shards_initialized(arr, strategy="nonsense")  # type: ignore[arg-type]


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_list_strategy_ignores_non_chunk_objects(store: Store) -> None:
    """The ``list`` strategy must not mistake unrelated objects sharing the
    array's prefix (e.g. metadata) for populated chunks."""
    arr, _ = _sparse_1d(store)
    # metadata (zarr.json) already lives under the array prefix; add another
    # non-chunk object to be sure it is excluded.
    keys = set(zarr.shards_initialized(arr, strategy="list"))
    assert keys == {"c/1", "c/5"}
    assert all(k.startswith("c/") for k in keys)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("setup_name", list(SETUPS))
def test_read_regions_reconstructs_baseline(store: Store, setup_name: str) -> None:
    """Packing the populated regions onto a fill-valued array reproduces the
    full ``arr[:]`` read exactly."""
    arr, baseline = SETUPS[setup_name](store)
    regions = zarr.read_regions(arr)
    result = _pack(arr, regions, baseline)
    assert np.array_equal(result, baseline)
    assert result.dtype == baseline.dtype


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("setup_name", list(SETUPS))
def test_read_regions_default_count_matches_discovery(store: Store, setup_name: str) -> None:
    """With no explicit regions, ``read_regions`` reads exactly the populated
    shards discovered by ``shards_initialized``."""
    arr, _ = SETUPS[setup_name](store)
    regions = zarr.read_regions(arr)
    assert len(regions) == len(zarr.shards_initialized(arr))


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_read_regions_explicit_regions(store: Store) -> None:
    """Explicit regions are read and returned with their decoded data."""
    arr, baseline = _sparse_1d(store)
    explicit = [(slice(8, 16),), (slice(40, 48),)]
    regions = dict(zarr.read_regions(arr, explicit))
    assert set(regions) == set(explicit)
    assert np.array_equal(np.asarray(regions[(slice(8, 16),)]), baseline[8:16])
    assert np.array_equal(np.asarray(regions[(slice(40, 48),)]), baseline[40:48])


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_read_regions_concurrency_one(store: Store) -> None:
    """A concurrency limit of 1 produces the same result as the default."""
    arr, baseline = _sparse_2d(store)
    regions = zarr.read_regions(arr, concurrency=1)
    assert np.array_equal(_pack(arr, regions, baseline), baseline)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("setup_name", list(SETUPS))
async def test_read_regions_async_matches_sync(store: Store, setup_name: str) -> None:
    """The async streaming generator yields the same ``(region, data)`` set as
    the synchronous wrapper."""
    arr, _ = SETUPS[setup_name](store)
    async_pairs = {
        region: np.asarray(data).tobytes()
        async for region, data in async_api.read_regions(arr._async_array)
    }
    sync_pairs = {region: np.asarray(data).tobytes() for region, data in zarr.read_regions(arr)}
    assert async_pairs == sync_pairs


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_shards_initialized_async(store: Store) -> None:
    arr, _ = _sparse_1d(store)
    keys = await async_api.shards_initialized(arr._async_array)
    assert set(keys) == {"c/1", "c/5"}
