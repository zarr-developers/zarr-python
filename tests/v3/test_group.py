from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest
from _pytest.compat import LEGACY_PATH

from zarr.array import Array, AsyncArray
from zarr.buffer import Buffer
from zarr.common import ZarrFormat
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.group import AsyncGroup, Group, GroupMetadata
from zarr.store import LocalStore, MemoryStore, StorePath
from zarr.store.core import make_store_path
from zarr.sync import sync

from .conftest import parse_store


@pytest.fixture(params=["local", "memory"])
async def store(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> LocalStore | MemoryStore:
    result = await parse_store(request.param, str(tmpdir))
    if not isinstance(result, MemoryStore | LocalStore):
        raise TypeError("Wrong store class returned by test fixture! got " + result + " instead")
    return result


@pytest.fixture(params=[True, False])
def exists_ok(request: pytest.FixtureRequest) -> bool:
    result = request.param
    if not isinstance(result, bool):
        raise TypeError("Wrong type returned by test fixture.")
    return result


@pytest.fixture(params=[2, 3], ids=["zarr2", "zarr3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    result = request.param
    if result not in (2, 3):
        raise ValueError("Wrong value returned from test fixture.")
    return cast(ZarrFormat, result)


def test_group_init(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    """
    Test that initializing a group from an asyncgroup works.
    """
    agroup = sync(AsyncGroup.create(store=store, zarr_format=zarr_format))
    group = Group(agroup)
    assert group._async_group == agroup


def test_group_name_properties(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    """
    Test basic properties of groups
    """
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


def test_group_members(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test that `Group.members` returns correct values, i.e. the arrays and groups
    (explicit and implicit) contained in that group.
    """

    path = "group"
    agroup = AsyncGroup(
        metadata=GroupMetadata(zarr_format=zarr_format),
        store_path=StorePath(store=store, path=path),
    )
    group = Group(agroup)
    members_expected: dict[str, Array | Group] = {}

    members_expected["subgroup"] = group.create_group("subgroup")
    # make a sub-sub-subgroup, to ensure that the children calculation doesn't go
    # too deep in the hierarchy
    _ = members_expected["subgroup"].create_group("subsubgroup")  # type: ignore

    members_expected["subarray"] = group.create_array(
        "subarray", shape=(100,), dtype="uint8", chunk_shape=(10,), exists_ok=True
    )

    # add an extra object to the domain of the group.
    # the list of children should ignore this object.
    sync(store.set(f"{path}/extra_object-1", Buffer.from_bytes(b"000000")))
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates a directory with a random key in it
    # this should not show up as a member
    sync(store.set(f"{path}/extra_directory/extra_object-2", Buffer.from_bytes(b"000000")))
    members_observed = group.members
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)


def test_group(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test basic Group routines.
    """
    store_path = StorePath(store)
    agroup = AsyncGroup(metadata=GroupMetadata(zarr_format=zarr_format), store_path=store_path)
    group = Group(agroup)
    assert agroup.metadata is group.metadata
    assert agroup.store_path == group.store_path == store_path

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


def test_group_create(
    store: MemoryStore | LocalStore, exists_ok: bool, zarr_format: ZarrFormat
) -> None:
    """
    Test that `Group.create` works as expected.
    """
    attributes = {"foo": 100}
    group = Group.create(store, attributes=attributes, zarr_format=zarr_format, exists_ok=exists_ok)

    assert group.attrs == attributes

    if not exists_ok:
        with pytest.raises(ContainsGroupError):
            group = Group.create(
                store, attributes=attributes, exists_ok=exists_ok, zarr_format=zarr_format
            )


def test_group_open(
    store: MemoryStore | LocalStore, zarr_format: ZarrFormat, exists_ok: bool
) -> None:
    """
    Test the `Group.open` method.
    """
    spath = StorePath(store)
    # attempt to open a group that does not exist
    with pytest.raises(FileNotFoundError):
        Group.open(store)

    # create the group
    attrs = {"path": "foo"}
    group_created = Group.create(
        store, attributes=attrs, zarr_format=zarr_format, exists_ok=exists_ok
    )
    assert group_created.attrs == attrs
    assert group_created.metadata.zarr_format == zarr_format
    assert group_created.store_path == spath

    # attempt to create a new group in place, to test exists_ok
    new_attrs = {"path": "bar"}
    if not exists_ok:
        with pytest.raises(ContainsGroupError):
            Group.create(store, attributes=attrs, zarr_format=zarr_format, exists_ok=exists_ok)
    else:
        group_created_again = Group.create(
            store, attributes=new_attrs, zarr_format=zarr_format, exists_ok=exists_ok
        )
        assert group_created_again.attrs == new_attrs
        assert group_created_again.metadata.zarr_format == zarr_format
        assert group_created_again.store_path == spath


def test_group_getitem(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__getitem__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    subgroup = group.create_group(name="subgroup")
    subarray = group.create_array(name="subarray", shape=(10,), chunk_shape=(10,))

    assert group["subgroup"] == subgroup
    assert group["subarray"] == subarray
    with pytest.raises(KeyError):
        group["nope"]


def test_group_delitem(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__delitem__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    subgroup = group.create_group(name="subgroup")
    subarray = group.create_array(name="subarray", shape=(10,), chunk_shape=(10,))

    assert group["subgroup"] == subgroup
    assert group["subarray"] == subarray

    del group["subgroup"]
    with pytest.raises(KeyError):
        group["subgroup"]

    del group["subarray"]
    with pytest.raises(KeyError):
        group["subarray"]


def test_group_iter(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__iter__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        [x for x in group]  # type: ignore


def test_group_len(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__len__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        len(group)  # type: ignore


def test_group_setitem(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__setitem__` method.
    """
    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        group["key"] = 10


def test_group_contains(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__contains__` method
    """
    group = Group.create(store, zarr_format=zarr_format)
    assert "foo" not in group
    _ = group.create_group(name="foo")
    assert "foo" in group


def test_group_subgroups(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group` methods for accessing subgroups, namely `Group.group_keys` and `Group.groups`
    """
    group = Group.create(store, zarr_format=zarr_format)
    keys = ("foo", "bar")
    subgroups_expected = tuple(group.create_group(k) for k in keys)
    # create a sub-array as well
    _ = group.create_array("array", shape=(10,))
    subgroups_observed = group.groups()
    assert set(group.group_keys()) == set(keys)
    assert len(subgroups_observed) == len(subgroups_expected)
    assert all(a in subgroups_observed for a in subgroups_expected)


def test_group_subarrays(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group` methods for accessing subgroups, namely `Group.group_keys` and `Group.groups`
    """
    group = Group.create(store, zarr_format=zarr_format)
    keys = ("foo", "bar")
    subarrays_expected = tuple(group.create_array(k, shape=(10,)) for k in keys)
    # create a sub-group as well
    _ = group.create_group("group")
    subarrays_observed = group.arrays()
    assert set(group.array_keys()) == set(keys)
    assert len(subarrays_observed) == len(subarrays_expected)
    assert all(a in subarrays_observed for a in subarrays_expected)


def test_group_update_attributes(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group.update_attributes`
    """
    attrs = {"foo": 100}
    group = Group.create(store, zarr_format=zarr_format, attributes=attrs)
    assert group.attrs == attrs
    new_attrs = {"bar": 100}
    new_group = group.update_attributes(new_attrs)
    assert new_group.attrs == new_attrs


async def test_group_update_attributes_async(
    store: MemoryStore | LocalStore, zarr_format: ZarrFormat
) -> None:
    """
    Test the behavior of `Group.update_attributes_async`
    """
    attrs = {"foo": 100}
    group = Group.create(store, zarr_format=zarr_format, attributes=attrs)
    assert group.attrs == attrs
    new_attrs = {"bar": 100}
    new_group = await group.update_attributes_async(new_attrs)
    assert new_group.attrs == new_attrs


@pytest.mark.parametrize("method", ["create_array", "array"])
def test_group_create_array(
    store: MemoryStore | LocalStore,
    zarr_format: ZarrFormat,
    exists_ok: bool,
    method: Literal["create_array", "array"],
) -> None:
    """
    Test `Group.create_array`
    """
    group = Group.create(store, zarr_format=zarr_format)
    shape = (10, 10)
    dtype = "uint8"
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)

    if method == "create_array":
        array = group.create_array(name="array", shape=shape, dtype=dtype, data=data)
    elif method == "array":
        array = group.array(name="array", shape=shape, dtype=dtype, data=data)
    else:
        raise AssertionError

    if not exists_ok:
        if method == "create_array":
            with pytest.raises(ContainsArrayError):
                group.create_array(name="array", shape=shape, dtype=dtype, data=data)
        elif method == "array":
            with pytest.raises(ContainsArrayError):
                group.array(name="array", shape=shape, dtype=dtype, data=data)
    assert array.shape == shape
    assert array.dtype == np.dtype(dtype)
    assert np.array_equal(array[:], data)


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("exists_ok", [True, False])
@pytest.mark.parametrize("extant_node", ["array", "group"])
def test_group_creation_existing_node(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
    exists_ok: bool,
    extant_node: Literal["array", "group"],
) -> None:
    """
    Check that an existing array or group is handled as expected during group creation.
    """
    spath = StorePath(store)
    group = Group.create(spath, zarr_format=zarr_format)
    expected_exception: type[ContainsArrayError] | type[ContainsGroupError]
    attributes = {"old": True}

    if extant_node == "array":
        expected_exception = ContainsArrayError
        _ = group.create_array("extant", shape=(10,), dtype="uint8", attributes=attributes)
    elif extant_node == "group":
        expected_exception = ContainsGroupError
        _ = group.create_group("extant", attributes=attributes)
    else:
        raise AssertionError

    new_attributes = {"new": True}

    if exists_ok:
        node_new = Group.create(
            spath / "extant",
            attributes=new_attributes,
            zarr_format=zarr_format,
            exists_ok=exists_ok,
        )
        assert node_new.attrs == new_attributes
    else:
        with pytest.raises(expected_exception):
            node_new = Group.create(
                spath / "extant",
                attributes=new_attributes,
                zarr_format=zarr_format,
                exists_ok=exists_ok,
            )


async def test_asyncgroup_create(
    store: MemoryStore | LocalStore,
    exists_ok: bool,
    zarr_format: ZarrFormat,
) -> None:
    """
    Test that `AsyncGroup.create` works as expected.
    """
    spath = StorePath(store=store)
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(
        store,
        attributes=attributes,
        exists_ok=exists_ok,
        zarr_format=zarr_format,
    )

    assert agroup.metadata == GroupMetadata(zarr_format=zarr_format, attributes=attributes)
    assert agroup.store_path == await make_store_path(store)

    if not exists_ok:
        with pytest.raises(ContainsGroupError):
            agroup = await AsyncGroup.create(
                spath,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
            )
        # create an array at our target path
        collision_name = "foo"
        _ = await AsyncArray.create(
            spath / collision_name, shape=(10,), dtype="uint8", zarr_format=zarr_format
        )
        with pytest.raises(ContainsArrayError):
            _ = await AsyncGroup.create(
                StorePath(store=store) / collision_name,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
            )


async def test_asyncgroup_attrs(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(store, zarr_format=zarr_format, attributes=attributes)

    assert agroup.attrs == agroup.metadata.attributes == attributes


async def test_asyncgroup_info(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    agroup = await AsyncGroup.create(  # noqa
        store,
        zarr_format=zarr_format,
    )
    pytest.xfail("Info is not implemented for metadata yet")
    # assert agroup.info == agroup.metadata.info


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


async def test_asyncgroup_open_wrong_format(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
) -> None:
    _ = await AsyncGroup.create(store=store, exists_ok=False, zarr_format=zarr_format)
    zarr_format_wrong: ZarrFormat
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


async def test_asyncgroup_create_array(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat, exists_ok: bool
) -> None:
    """
    Test that the AsyncGroup.create_array method works correctly. We ensure that array properties
    specified in create_array are present on the resulting array.
    """

    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    if not exists_ok:
        with pytest.raises(ContainsGroupError):
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
