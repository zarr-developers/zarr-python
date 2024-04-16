from __future__ import annotations
from typing import TYPE_CHECKING, Any

from zarr.v3.store.core import make_store_path

if TYPE_CHECKING:
    from zarr.v3.store import MemoryStore, LocalStore
    from zarr.v3.common import ZarrFormat

import pytest
import numpy as np

from zarr.v3.group import AsyncGroup, Group, GroupMetadata
from zarr.v3.store import StorePath
from zarr.v3.config import RuntimeConfiguration
from zarr.v3.sync import sync


# todo: put RemoteStore in here
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_group_children(store: MemoryStore | LocalStore):
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
    sync(store.set(f"{path}/extra_object", b"000000"))
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates an implicit group called implicit_subgroup
    sync(store.set(f"{path}/implicit_subgroup/extra_object", b"000000"))
    # make the implicit subgroup
    members_expected["implicit_subgroup"] = Group(
        AsyncGroup(
            metadata=GroupMetadata(),
            store_path=StorePath(store=store, path=f"{path}/implicit_subgroup"),
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


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("exists_ok", (True, False))
@pytest.mark.parametrize(
    "runtime_configuration", (RuntimeConfiguration(order="C"), RuntimeConfiguration(order="F"))
)
def test_group_create(
    store: MemoryStore | LocalStore, exists_ok: bool, runtime_configuration: RuntimeConfiguration
):
    """
    Test that `Group.create` works as expected.
    """
    attributes = {"foo": 100}
    group = Group.create(
        store,
        attributes=attributes,
        exists_ok=exists_ok,
        runtime_configuration=runtime_configuration,
    )

    assert group.attrs == attributes
    assert group._async_group.runtime_configuration == runtime_configuration

    if not exists_ok:
        with pytest.raises(AssertionError):
            group = Group.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("exists_ok", (True, False))
@pytest.mark.parametrize(
    "runtime_configuration", (RuntimeConfiguration(order="C"), RuntimeConfiguration(order="F"))
)
async def test_asyncgroup_create(
    store: MemoryStore | LocalStore,
    exists_ok: bool,
    zarr_format: ZarrFormat,
    runtime_configuration: RuntimeConfiguration,
):
    """
    Test that `AsyncGroup.create` works as expected.
    """
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(
        store,
        attributes=attributes,
        exists_ok=exists_ok,
        zarr_format=zarr_format,
        runtime_configuration=runtime_configuration,
    )

    assert agroup.metadata == GroupMetadata(zarr_format=zarr_format, attributes=attributes)
    assert agroup.store_path == make_store_path(store)
    assert agroup.runtime_configuration == runtime_configuration

    if not exists_ok:
        with pytest.raises(AssertionError):
            agroup = await AsyncGroup.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
                runtime_configuration=runtime_configuration,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_attrs(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(store, zarr_format=zarr_format, attributes=attributes)

    assert agroup.attrs == agroup.metadata.attributes == attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_info(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    agroup = await AsyncGroup.create(  # noqa
        store,
        zarr_format=zarr_format,
    )
    pytest.xfail("Info is not implemented for metadata yet")
    # assert agroup.info == agroup.metadata.info


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("runtime_configuration", (RuntimeConfiguration(),))
async def test_asyncgroup_open(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
    runtime_configuration: RuntimeConfiguration,
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
        runtime_configuration=runtime_configuration,
    )

    group_r = await AsyncGroup.open(
        store=store, zarr_format=zarr_format, runtime_configuration=runtime_configuration
    )

    assert group_w.attrs == group_w.attrs == attributes
    assert group_w == group_r


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (pytest.param(2, marks=pytest.mark.xfail), 3))
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
        assert False

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
def test_asyncgroup_from_dict(store: MemoryStore | LocalStore, data: dict[str, Any]):
    """
    Test that we can create an AsyncGroup from a dict
    """
    path = "test"
    store_path = StorePath(store=store, path=path)
    group = AsyncGroup.from_dict(
        store_path, data=data, runtime_configuration=RuntimeConfiguration()
    )

    assert group.metadata.zarr_format == data["zarr_format"]
    assert group.metadata.attributes == data["attributes"]


# todo: replace this with a declarative API where we model a full hierarchy
@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize(
    "zarr_format",
    (pytest.param(2, marks=pytest.mark.xfail), 3),
)
async def test_asyncgroup_getitem(store: LocalStore | MemoryStore, zarr_format: ZarrFormat):
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
@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_delitem(store: LocalStore | MemoryStore, zarr_format: ZarrFormat):
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
        assert False

    sub_group_path = "sub_group"
    _ = await agroup.create_group(sub_group_path, attributes={"foo": 100})
    await agroup.delitem(sub_group_path)
    if zarr_format == 2:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zgroup")
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + ".zattrs")
    elif zarr_format == 3:
        assert not await agroup.store_path.store.exists(sub_array_path + "/" + "zarr.json")
    else:
        assert False


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_asyncgroup_create_group(store: LocalStore | MemoryStore, zarr_format: ZarrFormat):
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    shape = (10,)
    dtype = "uint8"
    chunk_shape = (4,)
    attributes = {"foo": 100}

    sub_array_path = "sub_array"
    array = await agroup.create_array(
        path=sub_array_path,
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_shape,
        attributes=attributes,
    )

    assert array.shape == shape
    assert array.dtype == dtype
    # todo: fix this
    assert array.metadata.chunk_grid.chunk_shape == chunk_shape
