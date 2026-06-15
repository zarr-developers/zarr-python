from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.crud import NodeExistsError, get_backend
from zarr.errors import NodeNotFoundError
from zarr.storage import MemoryStore


def _array_meta() -> dict[str, Any]:
    arr = zarr.create_array(store=MemoryStore(), shape=(8, 8), chunks=(4, 4), dtype="uint16")
    return dict(arr.metadata.to_dict())


async def test_reference_round_trip_chunk() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    meta = _array_meta()
    await be.create_array(store, "a", meta, overwrite=False)
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await be.write_chunk(store, "a", meta, (0, 1), value.tobytes())
    raw = await be.read_chunk(store, "a", meta, (0, 1))
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="uint16").reshape(4, 4), value)


async def test_reference_read_subset_spans_chunks() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    arr = zarr.create_array(store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    data = np.arange(64, dtype="uint16").reshape(8, 8)
    arr[:, :] = data
    meta = dict(arr.metadata.to_dict())
    raw = await be.read_subset(store, "a", meta, (2, 1), (5, 4))
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="uint16").reshape(5, 4), data[2:7, 1:5])


async def test_reference_create_exists_raises() -> None:
    be = get_backend("reference")
    store = MemoryStore()
    meta = _array_meta()
    await be.create_array(store, "a", meta, overwrite=False)
    with pytest.raises(NodeExistsError):
        await be.create_array(store, "a", meta, overwrite=False)


async def test_reference_read_metadata_missing_raises() -> None:
    be = get_backend("reference")
    with pytest.raises(NodeNotFoundError):
        await be.read_metadata(MemoryStore(), "nope")
