from __future__ import annotations
from typing import TYPE_CHECKING

from zarr.buffer import as_buffer
from zarr.sync import sync

if TYPE_CHECKING:
    from zarr.store import MemoryStore, LocalStore
import pytest
import numpy as np

from zarr.group import AsyncGroup, Group, GroupMetadata
from zarr.store import LocalStore, StorePath
from zarr.config import RuntimeConfiguration


# todo: put RemoteStore in here
@pytest.mark.parametrize("store_type", ("local_store", "memory_store"))
def test_group_members(store_type, request):
    """
    Test that `Group.members` returns correct values, i.e. the arrays and groups
    (explicit and implicit) contained in that group.
    """

    store: LocalStore | MemoryStore = request.getfixturevalue(store_type)
    path = "group"
    agroup = AsyncGroup(
        metadata=GroupMetadata(),
        store_path=StorePath(store=store, path=path),
    )
    group = Group(agroup)
    members_expected = {}

    members_expected["subgroup"] = group.create_group("subgroup")
    # make a sub-sub-subgroup, to ensure that the children calculation doesn't go
    # too deep in the hierarchy
    _ = members_expected["subgroup"].create_group("subsubgroup")

    members_expected["subarray"] = group.create_array(
        "subarray", shape=(100,), dtype="uint8", chunk_shape=(10,), exists_ok=True
    )

    # add an extra object to the domain of the group.
    # the list of children should ignore this object.
    sync(store.set(f"{path}/extra_object-1", as_buffer(b"000000")))
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates a directory with a random key in it
    # this should not show up as a member
    sync(store.set(f"{path}/extra_directory/extra_object-2", b"000000"))
    members_observed = group.members
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)


@pytest.mark.parametrize("store_type", (("local_store",)))
def test_group(store_type, request) -> None:
    store = request.getfixturevalue(store_type)
    store_path = StorePath(store)
    agroup = AsyncGroup(
        metadata=GroupMetadata(),
        store_path=store_path,
        runtime_configuration=RuntimeConfiguration(),
    )
    group = Group(agroup)
    assert agroup.metadata is group.metadata

    # create two groups
    foo = group.create_group("foo")
    bar = foo.create_group("bar", attributes={"baz": "qux"})

    # create an array from the "bar" group
    data = np.arange(0, 4 * 4, dtype="uint16").reshape((4, 4))
    arr = bar.create_array(
        "baz", shape=data.shape, dtype=data.dtype, chunk_shape=(2, 2), exists_ok=True
    )
    arr[:] = data

    # check the array
    assert arr == bar["baz"]
    assert arr.shape == data.shape
    assert arr.dtype == data.dtype

    # TODO: update this once the array api settles down
    # assert arr.chunk_shape == (2, 2)

    bar2 = foo["bar"]
    assert dict(bar2.attrs) == {"baz": "qux"}

    # update a group's attributes
    bar2.attrs.update({"name": "bar"})
    # bar.attrs was modified in-place
    assert dict(bar2.attrs) == {"baz": "qux", "name": "bar"}

    # and the attrs were modified in the store
    bar3 = foo["bar"]
    assert dict(bar3.attrs) == {"baz": "qux", "name": "bar"}


def test_group_sync_constructor(store_path) -> None:
    group = Group.create(
        store=store_path,
        attributes={"title": "test 123"},
        runtime_configuration=RuntimeConfiguration(),
    )

    assert group._async_group.metadata.attributes["title"] == "test 123"
