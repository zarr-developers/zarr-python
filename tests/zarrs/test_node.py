from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import zarr
from zarr.core.buffer.core import default_buffer_prototype
from zarr.zarrs import NodeExistsError, create_new_group, create_overwrite_group

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
