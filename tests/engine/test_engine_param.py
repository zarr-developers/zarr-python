"""`engine=` threading through `zarr.create_array` / `zarr.open_array` and their
async counterparts (Task 7 of the array-engine-protocol plan)."""

from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.core.array import AsyncArray
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.storage import MemoryStore


def test_engine_param_combinations() -> None:
    store = MemoryStore()
    z = zarr.create_array(store, name="a", shape=(4,), chunks=(2,), dtype="int8", engine="default")
    z[:] = np.arange(4, dtype="int8")
    assert isinstance(z.engine, DefaultArrayEngine)

    z2 = zarr.open_array(store, path="a", engine="default")
    np.testing.assert_array_equal(np.asarray(z2[:]), np.arange(4, dtype="int8"))
    assert isinstance(z2.async_array.engine, DefaultAsyncArrayEngine)

    # user-provided sync instance
    inst = z2.engine
    z3 = zarr.open_array(store, path="a", engine=inst)
    assert z3.engine is inst
    # the wrapped AsyncArray keeps its own (default) engine -- a sync instance
    # must never reach AsyncArray.
    assert isinstance(z3.async_array.engine, DefaultAsyncArrayEngine)


def test_engine_param_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        zarr.create_array(
            MemoryStore(),
            shape=(2,),
            chunks=(2,),
            dtype="int8",
            engine="nope",  # type: ignore[arg-type]
        )


def test_async_array_rejects_sync_engine_instance() -> None:
    """A sync `ArrayEngine` instance passed to `AsyncArray` -- any construction
    path, not just the public API -- must raise `TypeError` naming both
    protocols."""
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    sync_engine = z.engine
    aa = z.async_array
    with pytest.raises(TypeError, match="ArrayEngine"):
        AsyncArray(
            metadata=aa.metadata,
            store_path=aa.store_path,
            engine=sync_engine,  # type: ignore[call-overload]
        )


def test_sync_api_rejects_async_engine_instance() -> None:
    """An `AsyncArrayEngine` instance passed to the sync `create_array` entry
    point must raise `TypeError` immediately (not lazily on first data
    access)."""
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    async_engine = z.async_array.engine
    with pytest.raises(TypeError, match="ArrayEngine"):
        zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8", engine=async_engine)


def test_engine_missing_read_selection_raises_type_error() -> None:
    """An object implementing neither engine protocol (no `read_selection`)
    must raise `TypeError` naming the protocols, not some other error."""
    with pytest.raises(TypeError, match="read_selection"):
        zarr.create_array(
            MemoryStore(),
            shape=(4,),
            chunks=(2,),
            dtype="int8",
            engine=object(),  # type: ignore[arg-type]
        )
