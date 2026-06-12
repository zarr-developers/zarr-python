from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from tests.zarrs.conftest import array_metadata
from zarr.core.buffer.core import default_buffer_prototype
from zarr.errors import NodeNotFoundError
from zarr.zarrs import (
    NodeExistsError,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    read_metadata,
)

if TYPE_CHECKING:
    from zarr.abc.store import Store

GROUP_META: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {"answer": 42},
}


async def test_create_new_group(store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo")
    group = zarr.open_group(store=store, path="foo", mode="r")
    assert dict(group.attrs) == {"answer": 42}


async def test_create_new_group_at_root(store: Store) -> None:
    await create_new_group(GROUP_META, store, "")
    group = zarr.open_group(store=store, mode="r")
    assert dict(group.attrs) == {"answer": 42}


async def test_create_new_group_existing_node(store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo")
    with pytest.raises(NodeExistsError):
        await create_new_group(GROUP_META, store, "foo")


async def test_create_overwrite_group(store: Store) -> None:
    # an array and its chunks previously occupied the path; overwrite removes both
    arr = zarr.create_array(store=store, name="foo", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    assert await store.exists("foo/c/0")
    await create_overwrite_group(GROUP_META, store, "foo")
    group = zarr.open_group(store=store, path="foo", mode="r")
    assert dict(group.attrs) == {"answer": 42}
    assert not await store.exists("foo/c/0")
    assert await store.get("foo/zarr.json", prototype=default_buffer_prototype()) is not None


async def test_create_new_array(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    arr = zarr.open_array(store=store, path="arr", mode="r")
    assert arr.shape == (8, 8)
    assert arr.chunks == (4, 4)
    assert arr.dtype == np.dtype("uint16")


async def test_create_new_array_existing_node(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    with pytest.raises(NodeExistsError):
        await create_new_array(array_metadata(), store, "arr")


async def test_create_overwrite_array(store: Store) -> None:
    zarr.create_group(store=store, path="arr")
    await create_overwrite_array(array_metadata(), store, "arr")
    arr = zarr.open_array(store=store, path="arr", mode="r")
    assert arr.shape == (8, 8)


async def test_read_metadata_matches_stored_document(store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr")
    observed = await read_metadata(store, "arr")
    raw = await store.get("arr/zarr.json", prototype=default_buffer_prototype())
    assert raw is not None
    assert observed == json.loads(raw.to_bytes())


async def test_read_metadata_zarr_python_group(store: Store) -> None:
    zarr.create_group(store=store, path="g", attributes={"a": 1})
    observed = await read_metadata(store, "g")
    assert observed["node_type"] == "group"
    assert observed["attributes"] == {"a": 1}


async def test_read_metadata_missing(store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await read_metadata(store, "nope")
