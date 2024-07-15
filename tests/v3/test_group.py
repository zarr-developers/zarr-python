from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.array import AsyncArray
from zarr.buffer import default_buffer_prototype
from zarr.store.core import make_store_path
from zarr.sync import sync

if TYPE_CHECKING:
    from zarr.common import ZarrFormat
    from zarr.store import LocalStore, MemoryStore

import numpy as np
import pytest

from zarr.group import AsyncGroup, Group, GroupMetadata
from zarr.store import StorePath


# todo: put RemoteStore in here
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_group_children(store: MemoryStore | LocalStore) -> None:
    """
    Test that `Group.members` returns correct values, i.e. the arrays and groups
    (explicit and implicit) contained in that group.
    """

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
    sync(store.set(f"{path}/extra_object-1", default_buffer_prototype.buffer.from_bytes(b"000000")))
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates a directory with a random key in it
    # this should not show up as a member
    sync(
        store.set(
            f"{path}/extra_directory/extra_object-2",
            default_buffer_prototype.buffer.from_bytes(b"000000"),
        )
    )
    members_observed = group.members
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)


@pytest.mark.parametrize("store", (("local", "memory")), indirect=["store"])
def test_group(store: MemoryStore | LocalStore) -> None:
    store_path = StorePath(store)
    agroup = AsyncGroup(metadata=GroupMetadata(), store_path=store_path)
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
    assert arr.chunks == (2, 2)

    bar2 = foo["bar"]
    assert dict(bar2.attrs) == {"baz": "qux"}

    # update a group's attributes
    bar2.attrs.update({"name": "bar"})
    # bar.attrs was modified in-place
    assert dict(bar2.attrs) == {"baz": "qux", "name": "bar"}

    # and the attrs were modified in the store
    bar3 = foo["bar"]
    assert dict(bar3.attrs) == {"baz": "qux", "name": "bar"}


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("exists_ok", (True, False))
def test_group_create(store: MemoryStore | LocalStore, exists_ok: bool) -> None:
    """
    Test that `Group.create` works as expected.
    """
    attributes = {"foo": 100}
    group = Group.create(store, attributes=attributes, exists_ok=exists_ok)

    assert group.attrs == attributes

    if not exists_ok:
        with pytest.raises(AssertionError):
            group = Group.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
            )


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("exists_ok", (True, False))
async def test_asyncgroup_create(
    store: MemoryStore | LocalStore,
    exists_ok: bool,
    zarr_format: ZarrFormat,
) -> None:
    """
    Test that `AsyncGroup.create` works as expected.
    """
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(
        store,
        attributes=attributes,
        exists_ok=exists_ok,
        zarr_format=zarr_format,
    )

    assert agroup.metadata == GroupMetadata(zarr_format=zarr_format, attributes=attributes)
    assert agroup.store_path == make_store_path(store)

    if not exists_ok:
        with pytest.raises(AssertionError):
            agroup = await AsyncGroup.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
            )


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_attrs(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(store, zarr_format=zarr_format, attributes=attributes)

    assert agroup.attrs == agroup.metadata.attributes == attributes


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_info(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    agroup = await AsyncGroup.create(  # noqa
        store,
        zarr_format=zarr_format,
    )
    pytest.xfail("Info is not implemented for metadata yet")
    # assert agroup.info == agroup.metadata.info


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_open(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
) -> None:
    """
    Create an `AsyncGroup`, then ensure that we can open it using `AsyncGroup.open`
    """
    attributes = {"foo": 100}
    group_w = await AsyncGroup.create(
        store=store,
        attributes=attributes,
        exists_ok=False,
        zarr_format=zarr_format,
    )

    group_r = await AsyncGroup.open(store=store, zarr_format=zarr_format)

    assert group_w.attrs == group_w.attrs == attributes
    assert group_w == group_r


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_open_wrong_format(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
) -> None:
    _ = await AsyncGroup.create(store=store, exists_ok=False, zarr_format=zarr_format)

    # try opening with the wrong zarr format
    if zarr_format == 3:
        zarr_format_wrong = 2
    elif zarr_format == 2:
        zarr_format_wrong = 3
    else:
        raise AssertionError

    with pytest.raises(FileNotFoundError):
        await AsyncGroup.open(store=store, zarr_format=zarr_format_wrong)


# todo: replace the dict[str, Any] type with something a bit more specific
# should this be async?
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize(
    "data",
    (
        {"zarr_format": 3, "node_type": "group", "attributes": {"foo": 100}},
        {"zarr_format": 2, "attributes": {"foo": 100}},
    ),
)
def test_asyncgroup_from_dict(store: MemoryStore | LocalStore, data: dict[str, Any]) -> None:
    """
    Test that we can create an AsyncGroup from a dict
    """
    path = "test"
    store_path = StorePath(store=store, path=path)
    group = AsyncGroup.from_dict(store_path, data=data)

    assert group.metadata.zarr_format == data["zarr_format"]
    assert group.metadata.attributes == data["attributes"]


# todo: replace this with a declarative API where we model a full hierarchy


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_getitem(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    """
    Create an `AsyncGroup`, then create members of that group, and ensure that we can access those
    members via the `AsyncGroup.getitem` method.
    """
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    sub_array_path = "sub_array"
    sub_array = await agroup.create_array(
        path=sub_array_path, shape=(10,), dtype="uint8", chunk_shape=(2,)
    )
    assert await agroup.getitem(sub_array_path) == sub_array

    sub_group_path = "sub_group"
    sub_group = await agroup.create_group(sub_group_path, attributes={"foo": 100})
    assert await agroup.getitem(sub_group_path) == sub_group

    # check that asking for a nonexistent key raises KeyError
    with pytest.raises(KeyError):
        await agroup.getitem("foo")


# todo: replace this with a declarative API where we model a full hierarchy


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_delitem(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    sub_array_path = "sub_array"
    _ = await agroup.create_array(
        path=sub_array_path, shape=(10,), dtype="uint8", chunk_shape=(2,), attributes={"foo": 100}
    )
    await agroup.delitem(sub_array_path)

    #  todo: clean up the code duplication here
    if zarr_format == 2:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zarray")
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zattrs")
    elif zarr_format == 3:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + "zarr.json")
    else:
        raise AssertionError

    sub_group_path = "sub_group"
    _ = await agroup.create_group(sub_group_path, attributes={"foo": 100})
    await agroup.delitem(sub_group_path)
    if zarr_format == 2:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zgroup")
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zattrs")
    elif zarr_format == 3:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + "zarr.json")
    else:
        raise AssertionError


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_create_group(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
) -> None:
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    sub_node_path = "sub_group"
    attributes = {"foo": 999}
    subnode = await agroup.create_group(path=sub_node_path, attributes=attributes)

    assert isinstance(subnode, AsyncGroup)
    assert subnode.attrs == attributes
    assert subnode.store_path.path == sub_node_path
    assert subnode.store_path.store == store
    assert subnode.metadata.zarr_format == zarr_format


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_create_array(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
) -> None:
    """
    Test that the AsyncGroup.create_array method works correctly. We ensure that array properties
    specified in create_array are present on the resulting array.
    """

    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    shape = (10,)
    dtype = "uint8"
    chunk_shape = (4,)
    attributes = {"foo": 100}

    sub_node_path = "sub_array"
    subnode = await agroup.create_array(
        path=sub_node_path,
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_shape,
        attributes=attributes,
    )
    assert isinstance(subnode, AsyncArray)
    assert subnode.attrs == attributes
    assert subnode.store_path.path == sub_node_path
    assert subnode.store_path.store == store
    assert subnode.shape == shape
    assert subnode.dtype == dtype
    # todo: fix the type annotation of array.metadata.chunk_grid so that we get some autocomplete
    # here.
    assert subnode.metadata.chunk_grid.chunk_shape == chunk_shape
    assert subnode.metadata.zarr_format == zarr_format


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_update_attributes(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    """
    Test that the AsyncGroup.update_attributes method works correctly.
    """
    attributes_old = {"foo": 10}
    attributes_new = {"baz": "new"}
    agroup = await AsyncGroup.create(
        store=store, zarr_format=zarr_format, attributes=attributes_old
    )

    agroup_new_attributes = await agroup.update_attributes(attributes_new)
    assert agroup_new_attributes.attrs == attributes_new


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
def test_group_init(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    agroup = sync(AsyncGroup.create(store=store, zarr_format=zarr_format))
    group = Group(agroup)
    assert group._async_group == agroup


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
def test_group_name_properties(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = Group.create(store=store, zarr_format=zarr_format)
    assert root.path == ""
    assert root.name == "/"
    assert root.basename == ""

    foo = root.create_group("foo")
    assert foo.path == "foo"
    assert foo.name == "/foo"
    assert foo.basename == "foo"

    bar = root.create_group("foo/bar")
    assert bar.path == "foo/bar"
    assert bar.name == "/foo/bar"
    assert bar.basename == "bar"
