from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.abc.engine import AsyncArrayEngine, Region
from zarr.core.buffer import default_buffer_prototype
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.core.sync import sync
from zarr.errors import ChunkNotFoundError
from zarr.storage import MemoryStore


def _make_array() -> zarr.Array[Any]:
    z = zarr.create_array(MemoryStore(), shape=(10, 9), chunks=(3, 4), dtype="int32", fill_value=0)
    z[:, :] = np.arange(90, dtype="int32").reshape(10, 9)
    return z


def test_default_async_engine_read_write_roundtrip() -> None:
    z = _make_array()
    eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    assert isinstance(eng, AsyncArrayEngine)
    proto = default_buffer_prototype()
    region = Region(start=(2, 1), end_exclusive=(7, 5))

    out = sync(eng.read_selection(region, prototype=proto))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(z[2:7, 1:5]))

    new: np.ndarray[Any, Any] = np.full((5, 4), -1, dtype="int32")
    value = proto.nd_buffer.from_ndarray_like(new)
    sync(eng.write_selection(region, value, prototype=proto))
    np.testing.assert_array_equal(np.asarray(z[2:7, 1:5]), new)


def test_default_sync_engine_matches_async() -> None:
    z = _make_array()
    async_eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    eng = DefaultArrayEngine(async_eng)
    proto = default_buffer_prototype()
    region = Region(start=(0, 0), end_exclusive=(10, 9))
    np.testing.assert_array_equal(
        np.asarray(eng.read_selection(region, prototype=proto)), np.asarray(z[:, :])
    )


def test_with_metadata_rebinds() -> None:
    z = _make_array()
    eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    new_meta = z.async_array.metadata
    assert eng.with_metadata(new_meta) is not eng


def test_read_missing_chunks_false_raises() -> None:
    z = zarr.create_array(
        MemoryStore(),
        shape=(6,),
        chunks=(2,),
        dtype="int16",
        config={"read_missing_chunks": False},
    )
    z[0:2] = np.arange(2, dtype="int16")  # chunks 1 and 2 never written
    eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    with pytest.raises(ChunkNotFoundError):
        sync(
            eng.read_selection(
                Region(start=(0,), end_exclusive=(6,)), prototype=default_buffer_prototype()
            )
        )
