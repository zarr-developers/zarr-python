from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

import zarr
from zarr import Array, AsyncArray, AsyncGroup, Group
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import JSON, ZarrFormat
from zarr.core.group import GroupMetadata
from zarr.core.sync import sync
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.store import LocalStore, MemoryStore, StorePath
from zarr.store.common import make_store_path

from .conftest import parse_store

if TYPE_CHECKING:
    from _pytest.compat import LEGACY_PATH


@pytest.fixture(params=["local", "memory", "zip"])
async def store(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> Store:
    result = await parse_store(request.param, str(tmpdir))
    if not isinstance(result, Store):
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


def test_group_init(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test that initializing a group from an asyncgroup works.
    """
    agroup = sync(AsyncGroup.create(store=store, zarr_format=zarr_format))
    group = Group(agroup)
    assert group._async_group == agroup


def test_group_name_properties(store: Store, zarr_format: ZarrFormat) -> None:
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


def test_group_members(store: Store, zarr_format: ZarrFormat) -> None:
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
    subsubgroup = members_expected["subgroup"].create_group("subsubgroup")  # type: ignore
    subsubsubgroup = subsubgroup.create_group("subsubsubgroup")  # type: ignore

    members_expected["subarray"] = group.create_array(
        "subarray", shape=(100,), dtype="uint8", chunk_shape=(10,), exists_ok=True
    )

    # add an extra object to the domain of the group.
    # the list of children should ignore this object.
    sync(
        store.set(f"{path}/extra_object-1", default_buffer_prototype().buffer.from_bytes(b"000000"))
    )
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates a directory with a random key in it
    # this should not show up as a member
    sync(
        store.set(
            f"{path}/extra_directory/extra_object-2",
            default_buffer_prototype().buffer.from_bytes(b"000000"),
        )
    )
    members_observed = group.members()
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)

    # partial
    members_observed = group.members(max_depth=1)
    members_expected["subgroup/subsubgroup"] = subsubgroup
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)

    # total
    members_observed = group.members(max_depth=None)
    members_expected["subgroup/subsubgroup/subsubsubgroup"] = subsubsubgroup
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)

    with pytest.raises(ValueError, match="max_depth"):
        members_observed = group.members(max_depth=-1)


def test_group(store: Store, zarr_format: ZarrFormat) -> None:
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


def test_group_create(store: Store, exists_ok: bool, zarr_format: ZarrFormat) -> None:
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


def test_group_open(store: Store, zarr_format: ZarrFormat, exists_ok: bool) -> None:
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


def test_group_getitem(store: Store, zarr_format: ZarrFormat) -> None:
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


def test_group_delitem(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__delitem__` method.
    """
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")

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


def test_group_iter(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__iter__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        [x for x in group]  # type: ignore


def test_group_len(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__len__` method.
    """

    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        len(group)  # type: ignore


def test_group_setitem(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__setitem__` method.
    """
    group = Group.create(store, zarr_format=zarr_format)
    with pytest.raises(NotImplementedError):
        group["key"] = 10


def test_group_contains(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__contains__` method
    """
    group = Group.create(store, zarr_format=zarr_format)
    assert "foo" not in group
    _ = group.create_group(name="foo")
    assert "foo" in group


def test_group_subgroups(store: Store, zarr_format: ZarrFormat) -> None:
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


def test_group_subarrays(store: Store, zarr_format: ZarrFormat) -> None:
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


def test_group_update_attributes(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group.update_attributes`
    """
    attrs = {"foo": 100}
    group = Group.create(store, zarr_format=zarr_format, attributes=attrs)
    assert group.attrs == attrs
    new_attrs = {"bar": 100}
    new_group = group.update_attributes(new_attrs)
    assert new_group.attrs == new_attrs


async def test_group_update_attributes_async(store: Store, zarr_format: ZarrFormat) -> None:
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
    store: Store,
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
        with pytest.warns(DeprecationWarning):
            array = group.array(name="array", shape=shape, dtype=dtype, data=data)
    else:
        raise AssertionError

    if not exists_ok:
        if method == "create_array":
            with pytest.raises(ContainsArrayError):
                group.create_array(name="array", shape=shape, dtype=dtype, data=data)
        elif method == "array":
            with pytest.raises(ContainsArrayError), pytest.warns(DeprecationWarning):
                group.array(name="array", shape=shape, dtype=dtype, data=data)
    assert array.shape == shape
    assert array.dtype == np.dtype(dtype)
    assert np.array_equal(array[:], data)


@pytest.mark.parametrize("store", ("local", "memory", "zip"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("exists_ok", [True, False])
@pytest.mark.parametrize("extant_node", ["array", "group"])
def test_group_creation_existing_node(
    store: Store,
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
    attributes: dict[str, JSON] = {"old": True}

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
    store: Store,
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


async def test_asyncgroup_attrs(store: Store, zarr_format: ZarrFormat) -> None:
    attributes = {"foo": 100}
    agroup = await AsyncGroup.create(store, zarr_format=zarr_format, attributes=attributes)

    assert agroup.attrs == agroup.metadata.attributes == attributes


async def test_asyncgroup_info(store: Store, zarr_format: ZarrFormat) -> None:
    agroup = await AsyncGroup.create(  # noqa
        store,
        zarr_format=zarr_format,
    )
    pytest.xfail("Info is not implemented for metadata yet")
    # assert agroup.info == agroup.metadata.info


async def test_asyncgroup_open(
    store: Store,
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
    store: Store,
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
def test_asyncgroup_from_dict(store: Store, data: dict[str, Any]) -> None:
    """
    Test that we can create an AsyncGroup from a dict
    """
    path = "test"
    store_path = StorePath(store=store, path=path)
    group = AsyncGroup.from_dict(store_path, data=data)

    assert group.metadata.zarr_format == data["zarr_format"]
    assert group.metadata.attributes == data["attributes"]


# todo: replace this with a declarative API where we model a full hierarchy


async def test_asyncgroup_getitem(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Create an `AsyncGroup`, then create members of that group, and ensure that we can access those
    members via the `AsyncGroup.getitem` method.
    """
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    array_name = "sub_array"
    sub_array = await agroup.create_array(
        name=array_name, shape=(10,), dtype="uint8", chunk_shape=(2,)
    )
    assert await agroup.getitem(array_name) == sub_array

    sub_group_path = "sub_group"
    sub_group = await agroup.create_group(sub_group_path, attributes={"foo": 100})
    assert await agroup.getitem(sub_group_path) == sub_group

    # check that asking for a nonexistent key raises KeyError
    with pytest.raises(KeyError):
        await agroup.getitem("foo")


async def test_asyncgroup_delitem(store: Store, zarr_format: ZarrFormat) -> None:
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")

    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    array_name = "sub_array"
    _ = await agroup.create_array(
        name=array_name, shape=(10,), dtype="uint8", chunk_shape=(2,), attributes={"foo": 100}
    )
    await agroup.delitem(array_name)

    #  todo: clean up the code duplication here
    if zarr_format == 2:
        assert not await agroup.store_path.store.exists(array_name + "/" + ".zarray")
        assert not await agroup.store_path.store.exists(array_name + "/" + ".zattrs")
    elif zarr_format == 3:
        assert not await agroup.store_path.store.exists(array_name + "/" + "zarr.json")
    else:
        raise AssertionError

    sub_group_path = "sub_group"
    _ = await agroup.create_group(sub_group_path, attributes={"foo": 100})
    await agroup.delitem(sub_group_path)
    if zarr_format == 2:
        assert not await agroup.store_path.store.exists(array_name + "/" + ".zgroup")
        assert not await agroup.store_path.store.exists(array_name + "/" + ".zattrs")
    elif zarr_format == 3:
        assert not await agroup.store_path.store.exists(array_name + "/" + "zarr.json")
    else:
        raise AssertionError


async def test_asyncgroup_create_group(
    store: Store,
    zarr_format: ZarrFormat,
) -> None:
    agroup = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    sub_node_path = "sub_group"
    attributes = {"foo": 999}
    subnode = await agroup.create_group(name=sub_node_path, attributes=attributes)

    assert isinstance(subnode, AsyncGroup)
    assert subnode.attrs == attributes
    assert subnode.store_path.path == sub_node_path
    assert subnode.store_path.store == store
    assert subnode.metadata.zarr_format == zarr_format


async def test_asyncgroup_create_array(
    store: Store, zarr_format: ZarrFormat, exists_ok: bool
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
    attributes: dict[str, JSON] = {"foo": 100}

    sub_node_path = "sub_array"
    subnode = await agroup.create_array(
        name=sub_node_path,
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


async def test_asyncgroup_update_attributes(store: Store, zarr_format: ZarrFormat) -> None:
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


@pytest.mark.parametrize("store", ("local",), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_serializable_async_group(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = await AsyncGroup.create(
        store=store, attributes={"foo": 999}, zarr_format=zarr_format
    )
    p = pickle.dumps(expected)
    actual = pickle.loads(p)
    assert actual == expected


@pytest.mark.parametrize("store", ("local",), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
def test_serializable_sync_group(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = Group.create(store=store, attributes={"foo": 999}, zarr_format=zarr_format)
    p = pickle.dumps(expected)
    actual = pickle.loads(p)

    assert actual == expected


async def test_group_members_async(store: LocalStore | MemoryStore) -> None:
    group = AsyncGroup(
        GroupMetadata(),
        store_path=StorePath(store=store, path="root"),
    )
    a0 = await group.create_array("a0", shape=(1,))
    g0 = await group.create_group("g0")
    a1 = await g0.create_array("a1", shape=(1,))
    g1 = await g0.create_group("g1")
    a2 = await g1.create_array("a2", shape=(1,))
    g2 = await g1.create_group("g2")

    # immediate children
    children = sorted([x async for x in group.members()], key=lambda x: x[0])
    assert children == [
        ("a0", a0),
        ("g0", g0),
    ]

    nmembers = await group.nmembers()
    assert nmembers == 2

    # partial
    children = sorted([x async for x in group.members(max_depth=1)], key=lambda x: x[0])
    expected = [
        ("a0", a0),
        ("g0", g0),
        ("g0/a1", a1),
        ("g0/g1", g1),
    ]
    assert children == expected
    nmembers = await group.nmembers(max_depth=1)
    assert nmembers == 4

    # all children
    all_children = sorted([x async for x in group.members(max_depth=None)], key=lambda x: x[0])
    expected = [
        ("a0", a0),
        ("g0", g0),
        ("g0/a1", a1),
        ("g0/g1", g1),
        ("g0/g1/a2", a2),
        ("g0/g1/g2", g2),
    ]
    assert all_children == expected

    nmembers = await group.nmembers(max_depth=None)
    assert nmembers == 6

    with pytest.raises(ValueError, match="max_depth"):
        [x async for x in group.members(max_depth=-1)]


async def test_require_group(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.create(store=store, zarr_format=zarr_format)

    # create foo group
    _ = await root.create_group("foo", attributes={"foo": 100})

    # test that we can get the group using require_group
    foo_group = await root.require_group("foo")
    assert foo_group.attrs == {"foo": 100}

    # test that we can get the group using require_group and overwrite=True
    foo_group = await root.require_group("foo", overwrite=True)

    _ = await foo_group.create_array(
        "bar", shape=(10,), dtype="uint8", chunk_shape=(2,), attributes={"foo": 100}
    )

    # test that overwriting a group w/ children fails
    # TODO: figure out why ensure_no_existing_node is not catching the foo.bar array
    #
    # with pytest.raises(ContainsArrayError):
    #     await root.require_group("foo", overwrite=True)

    # test that requiring a group where an array is fails
    with pytest.raises(TypeError):
        await foo_group.require_group("bar")


async def test_require_groups(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    # create foo group
    _ = await root.create_group("foo", attributes={"foo": 100})
    # create bar group
    _ = await root.create_group("bar", attributes={"bar": 200})

    foo_group, bar_group = await root.require_groups("foo", "bar")
    assert foo_group.attrs == {"foo": 100}
    assert bar_group.attrs == {"bar": 200}

    # get a mix of existing and new groups
    foo_group, spam_group = await root.require_groups("foo", "spam")
    assert foo_group.attrs == {"foo": 100}
    assert spam_group.attrs == {}

    # no names
    no_group = await root.require_groups()
    assert no_group == ()


async def test_create_dataset(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    with pytest.warns(DeprecationWarning):
        foo = await root.create_dataset("foo", shape=(10,), dtype="uint8")
    assert foo.shape == (10,)

    with pytest.raises(ContainsArrayError), pytest.warns(DeprecationWarning):
        await root.create_dataset("foo", shape=(100,), dtype="int8")

    _ = await root.create_group("bar")
    with pytest.raises(ContainsGroupError), pytest.warns(DeprecationWarning):
        await root.create_dataset("bar", shape=(100,), dtype="int8")


async def test_require_array(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.create(store=store, zarr_format=zarr_format)
    foo1 = await root.require_array("foo", shape=(10,), dtype="i8", attributes={"foo": 101})
    assert foo1.attrs == {"foo": 101}
    foo2 = await root.require_array("foo", shape=(10,), dtype="i8")
    assert foo2.attrs == {"foo": 101}

    # exact = False
    _ = await root.require_array("foo", shape=10, dtype="f8")

    # errors w/ exact True
    with pytest.raises(TypeError, match="Incompatible dtype"):
        await root.require_array("foo", shape=(10,), dtype="f8", exact=True)

    with pytest.raises(TypeError, match="Incompatible shape"):
        await root.require_array("foo", shape=(100, 100), dtype="i8")

    with pytest.raises(TypeError, match="Incompatible dtype"):
        await root.require_array("foo", shape=(10,), dtype="f4")

    _ = await root.create_group("bar")
    with pytest.raises(TypeError, match="Incompatible object"):
        await root.require_array("bar", shape=(10,), dtype="int8")


async def test_open_mutable_mapping():
    group = await zarr.api.asynchronous.open_group(store={}, mode="w")
    assert isinstance(group.store_path.store, MemoryStore)


def test_open_mutable_mapping_sync():
    group = zarr.open_group(store={}, mode="w")
    assert isinstance(group.store_path.store, MemoryStore)
