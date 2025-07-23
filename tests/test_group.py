from __future__ import annotations

import contextlib
import inspect
import operator
import pickle
import re
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from numcodecs import Blosc

import zarr
import zarr.api.asynchronous
import zarr.api.synchronous
import zarr.storage
from zarr import Array, AsyncArray, AsyncGroup, Group
from zarr.abc.store import Store
from zarr.core import sync_group
from zarr.core._info import GroupInfo
from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import config as zarr_config
from zarr.core.dtype.common import unpack_dtype_json
from zarr.core.dtype.npy.int import UInt8
from zarr.core.group import (
    ConsolidatedMetadata,
    GroupMetadata,
    ImplicitGroupMarker,
    _build_metadata_v3,
    _get_roots,
    _parse_hierarchy_dict,
    create_hierarchy,
    create_nodes,
    create_rooted_hierarchy,
    get_node,
)
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import _collect_aiterator, sync
from zarr.errors import ContainsArrayError, ContainsGroupError, MetadataValidationError
from zarr.storage import LocalStore, MemoryStore, StorePath, ZipStore
from zarr.storage._common import make_store_path
from zarr.storage._utils import _join_paths, normalize_path
from zarr.testing.store import LatencyStore

from .conftest import meta_from_array, parse_store

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.compat import LEGACY_PATH

    from zarr.core.common import JSON, ZarrFormat


@pytest.fixture(params=["local", "memory", "zip"])
async def store(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> Store:
    result = await parse_store(request.param, str(tmpdir))
    if not isinstance(result, Store):
        raise TypeError("Wrong store class returned by test fixture! got " + result + " instead")
    return result


@pytest.fixture(params=[True, False])
def overwrite(request: pytest.FixtureRequest) -> bool:
    result = request.param
    if not isinstance(result, bool):
        raise TypeError("Wrong type returned by test fixture.")
    return result


def test_group_init(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test that initializing a group from an asyncgroup works.
    """
    agroup = sync(AsyncGroup.from_store(store=store, zarr_format=zarr_format))
    group = Group(agroup)
    assert group._async_group == agroup


async def test_create_creates_parents(store: Store, zarr_format: ZarrFormat) -> None:
    # prepare a root node, with some data set
    await zarr.api.asynchronous.open_group(
        store=store, path="a", zarr_format=zarr_format, attributes={"key": "value"}
    )
    objs = {x async for x in store.list()}
    if zarr_format == 2:
        assert objs == {".zgroup", ".zattrs", "a/.zgroup", "a/.zattrs"}
    else:
        assert objs == {"zarr.json", "a/zarr.json"}

    # test that root group node was created
    root = await zarr.api.asynchronous.open_group(
        store=store,
    )
    agroup = await root.getitem("a")
    assert agroup.attrs == {"key": "value"}

    # create a child node with a couple intermediates
    await zarr.api.asynchronous.open_group(store=store, path="a/b/c/d", zarr_format=zarr_format)
    parts = ["a", "a/b", "a/b/c"]

    if zarr_format == 2:
        files = [".zattrs", ".zgroup"]
    else:
        files = ["zarr.json"]

    expected = [f"{part}/{file}" for file in files for part in parts]

    if zarr_format == 2:
        expected.extend([".zgroup", ".zattrs", "a/b/c/d/.zgroup", "a/b/c/d/.zattrs"])
    else:
        expected.extend(["zarr.json", "a/b/c/d/zarr.json"])

    expected = sorted(expected)

    result = sorted([x async for x in store.list_prefix("")])

    assert result == expected

    paths = ["a", "a/b", "a/b/c"]
    for path in paths:
        g = await zarr.api.asynchronous.open_group(store=store, path=path)
        assert isinstance(g, AsyncGroup)

        if path == "a":
            # ensure we didn't overwrite the root attributes
            assert g.attrs == {"key": "value"}
        else:
            assert g.attrs == {}


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("root_name", ["", "/", "a", "/a"])
@pytest.mark.parametrize("branch_name", ["foo", "/foo", "foo/bar", "/foo/bar"])
def test_group_name_properties(
    store: Store, zarr_format: ZarrFormat, root_name: str, branch_name: str
) -> None:
    """
    Test that the path, name, and basename attributes of a group and its subgroups are consistent
    """
    root = Group.from_store(store=StorePath(store=store, path=root_name), zarr_format=zarr_format)
    assert root.path == normalize_path(root_name)
    assert root.name == "/" + root.path
    assert root.basename == root.path

    branch = root.create_group(branch_name)
    if root.path == "":
        assert branch.path == normalize_path(branch_name)
    else:
        assert branch.path == "/".join([root.path, normalize_path(branch_name)])
    assert branch.name == "/" + branch.path
    assert branch.basename == branch_name.split("/")[-1]


@pytest.mark.parametrize("consolidated_metadata", [True, False])
def test_group_members(store: Store, zarr_format: ZarrFormat, consolidated_metadata: bool) -> None:
    """
    Test that `Group.members` returns correct values, i.e. the arrays and groups
    (explicit and implicit) contained in that group.
    """
    # group/
    #   subgroup/
    #     subsubgroup/
    #       subsubsubgroup
    #   subarray

    path = "group"
    group = Group.from_store(
        store=store,
        zarr_format=zarr_format,
    )
    members_expected: dict[str, Array | Group] = {}

    members_expected["subgroup"] = group.create_group("subgroup")
    # make a sub-sub-subgroup, to ensure that the children calculation doesn't go
    # too deep in the hierarchy
    subsubgroup = members_expected["subgroup"].create_group("subsubgroup")
    subsubsubgroup = subsubgroup.create_group("subsubsubgroup")

    members_expected["subarray"] = group.create_array(
        "subarray", shape=(100,), dtype="uint8", chunks=(10,), overwrite=True
    )
    # add an extra object to the domain of the group.
    # the list of children should ignore this object.
    sync(
        store.set(
            f"{path}/extra_object-1",
            default_buffer_prototype().buffer.from_bytes(b"000000"),
        )
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

    # this warning shows up when extra objects show up in the hierarchy
    warn_context = pytest.warns(
        UserWarning, match=r"Object at .* is not recognized as a component of a Zarr hierarchy."
    )
    if consolidated_metadata:
        with warn_context:
            zarr.consolidate_metadata(store=store, zarr_format=zarr_format)
        # now that we've consolidated the store, we shouldn't get the warnings from the unrecognized objects anymore
        # we use a nullcontext to handle these cases
        warn_context = contextlib.nullcontext()
        group = zarr.open_consolidated(store=store, zarr_format=zarr_format)

    with warn_context:
        members_observed = group.members()
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)

    # partial
    with warn_context:
        members_observed = group.members(max_depth=1)
    members_expected["subgroup/subsubgroup"] = subsubgroup
    # members are not guaranteed to be ordered, so sort before comparing
    assert sorted(dict(members_observed)) == sorted(members_expected)

    # total
    with warn_context:
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
    arr = bar.create_array("baz", shape=data.shape, dtype=data.dtype, chunks=(2, 2), overwrite=True)
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


def test_group_create(store: Store, overwrite: bool, zarr_format: ZarrFormat) -> None:
    """
    Test that `Group.from_store` works as expected.
    """
    attributes = {"foo": 100}
    group = Group.from_store(
        store, attributes=attributes, zarr_format=zarr_format, overwrite=overwrite
    )

    assert group.attrs == attributes

    if not overwrite:
        with pytest.raises(ContainsGroupError):
            _ = Group.from_store(store, overwrite=overwrite, zarr_format=zarr_format)


def test_group_open(store: Store, zarr_format: ZarrFormat, overwrite: bool) -> None:
    """
    Test the `Group.open` method.
    """
    spath = StorePath(store)
    # attempt to open a group that does not exist
    with pytest.raises(FileNotFoundError):
        Group.open(store)

    # create the group
    attrs = {"path": "foo"}
    group_created = Group.from_store(
        store, attributes=attrs, zarr_format=zarr_format, overwrite=overwrite
    )
    assert group_created.attrs == attrs
    assert group_created.metadata.zarr_format == zarr_format
    assert group_created.store_path == spath

    # attempt to create a new group in place, to test overwrite
    new_attrs = {"path": "bar"}
    if not overwrite:
        with pytest.raises(ContainsGroupError):
            Group.from_store(store, attributes=attrs, zarr_format=zarr_format, overwrite=overwrite)
    else:
        if not store.supports_deletes:
            pytest.skip(
                "Store does not support deletes but `overwrite` is True, requiring deletes to override a group"
            )
        group_created_again = Group.from_store(
            store, attributes=new_attrs, zarr_format=zarr_format, overwrite=overwrite
        )
        assert group_created_again.attrs == new_attrs
        assert group_created_again.metadata.zarr_format == zarr_format
        assert group_created_again.store_path == spath


@pytest.mark.parametrize("consolidated", [True, False])
def test_group_getitem(store: Store, zarr_format: ZarrFormat, consolidated: bool) -> None:
    """
    Test the `Group.__getitem__` method.
    """

    group = Group.from_store(store, zarr_format=zarr_format)
    subgroup = group.create_group(name="subgroup")
    subarray = group.create_array(name="subarray", shape=(10,), chunks=(10,), dtype="uint8")
    subsubarray = subgroup.create_array(name="subarray", shape=(10,), chunks=(10,), dtype="uint8")

    if consolidated:
        group = zarr.api.synchronous.consolidate_metadata(store=store, zarr_format=zarr_format)
        # we're going to assume that `group.metadata` is correct, and reuse that to focus
        # on indexing in this test. Other tests verify the correctness of group.metadata
        object.__setattr__(
            subgroup.metadata,
            "consolidated_metadata",
            ConsolidatedMetadata(
                metadata={"subarray": group.metadata.consolidated_metadata.metadata["subarray"]}
            ),
        )

    assert group["subgroup"] == subgroup
    assert group["subarray"] == subarray
    assert group["subgroup"]["subarray"] == subsubarray
    assert group["subgroup/subarray"] == subsubarray

    with pytest.raises(KeyError):
        group["nope"]

    with pytest.raises(KeyError, match="subarray/subsubarray"):
        group["subarray/subsubarray"]

    # Now test the mixed case
    if consolidated:
        object.__setattr__(
            group.metadata.consolidated_metadata.metadata["subgroup"],
            "consolidated_metadata",
            None,
        )

        # test the implementation directly
        with pytest.raises(KeyError):
            group._async_group._getitem_consolidated(
                group.store_path, "subgroup/subarray", prefix="/"
            )

        with pytest.raises(KeyError):
            # We've chosen to trust the consolidated metadata, which doesn't
            # contain this array
            group["subgroup/subarray"]

        with pytest.raises(KeyError, match="subarray/subsubarray"):
            group["subarray/subsubarray"]


def test_group_get_with_default(store: Store, zarr_format: ZarrFormat) -> None:
    group = Group.from_store(store, zarr_format=zarr_format)

    # default behavior
    result = group.get("subgroup")
    assert result is None

    # custom default
    result = group.get("subgroup", 8)
    assert result == 8

    # now with a group
    subgroup = group.require_group("subgroup")
    subgroup.attrs["foo"] = "bar"

    result = group.get("subgroup", 8)
    assert result.attrs["foo"] == "bar"


@pytest.mark.parametrize("consolidated", [True, False])
def test_group_delitem(store: Store, zarr_format: ZarrFormat, consolidated: bool) -> None:
    """
    Test the `Group.__delitem__` method.
    """
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")

    group = Group.from_store(store, zarr_format=zarr_format)
    subgroup = group.create_group(name="subgroup")
    subarray = group.create_array(name="subarray", shape=(10,), chunks=(10,), dtype="uint8")

    if consolidated:
        group = zarr.api.synchronous.consolidate_metadata(store=store, zarr_format=zarr_format)
        object.__setattr__(
            subgroup.metadata, "consolidated_metadata", ConsolidatedMetadata(metadata={})
        )

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

    group = Group.from_store(store, zarr_format=zarr_format)
    assert list(group) == []


def test_group_len(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__len__` method.
    """

    group = Group.from_store(store, zarr_format=zarr_format)
    assert len(group) == 0


def test_group_setitem(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__setitem__` method.
    """
    group = Group.from_store(store, zarr_format=zarr_format)
    arr = np.ones((2, 4))
    group["key"] = arr
    assert list(group.array_keys()) == ["key"]
    assert group["key"].shape == (2, 4)
    np.testing.assert_array_equal(group["key"][:], arr)

    if store.supports_deletes:
        key = "key"
    else:
        # overwriting with another array requires deletes
        # for stores that don't support this, we just use a new key
        key = "key2"

    # overwrite with another array
    arr = np.zeros((3, 5))
    group[key] = arr
    assert key in list(group.array_keys())
    assert group[key].shape == (3, 5)
    np.testing.assert_array_equal(group[key], arr)


def test_group_contains(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the `Group.__contains__` method
    """
    group = Group.from_store(store, zarr_format=zarr_format)
    assert "foo" not in group
    _ = group.create_group(name="foo")
    assert "foo" in group


@pytest.mark.parametrize("consolidate", [True, False])
def test_group_child_iterators(store: Store, zarr_format: ZarrFormat, consolidate: bool):
    group = Group.from_store(store, zarr_format=zarr_format)
    expected_group_keys = ["g0", "g1"]
    expected_group_values = [group.create_group(name=name) for name in expected_group_keys]
    expected_groups = list(zip(expected_group_keys, expected_group_values, strict=False))

    fill_value = 3
    dtype = UInt8()

    expected_group_values[0].create_group("subgroup")
    expected_group_values[0].create_array(
        "subarray", shape=(1,), dtype=dtype, fill_value=fill_value
    )

    expected_array_keys = ["a0", "a1"]

    expected_array_values = [
        group.create_array(name=name, shape=(1,), dtype=dtype, fill_value=fill_value)
        for name in expected_array_keys
    ]
    expected_arrays = list(zip(expected_array_keys, expected_array_values, strict=False))

    if consolidate:
        group = zarr.consolidate_metadata(store)
        if zarr_format == 2:
            metadata = {
                "subarray": {
                    "attributes": {},
                    "dtype": unpack_dtype_json(dtype.to_json(zarr_format=zarr_format)),
                    "fill_value": fill_value,
                    "shape": (1,),
                    "chunks": (1,),
                    "order": "C",
                    "filters": None,
                    "compressor": Blosc(),
                    "zarr_format": zarr_format,
                },
                "subgroup": {
                    "attributes": {},
                    "consolidated_metadata": {
                        "metadata": {},
                        "kind": "inline",
                        "must_understand": False,
                    },
                    "node_type": "group",
                    "zarr_format": zarr_format,
                },
            }
        else:
            metadata = {
                "subarray": {
                    "attributes": {},
                    "chunk_grid": {
                        "configuration": {"chunk_shape": (1,)},
                        "name": "regular",
                    },
                    "chunk_key_encoding": {
                        "configuration": {"separator": "/"},
                        "name": "default",
                    },
                    "codecs": (
                        {"configuration": {"endian": "little"}, "name": "bytes"},
                        {"configuration": {}, "name": "zstd"},
                    ),
                    "data_type": unpack_dtype_json(dtype.to_json(zarr_format=zarr_format)),
                    "fill_value": fill_value,
                    "node_type": "array",
                    "shape": (1,),
                    "zarr_format": zarr_format,
                },
                "subgroup": {
                    "attributes": {},
                    "consolidated_metadata": {
                        "metadata": {},
                        "kind": "inline",
                        "must_understand": False,
                    },
                    "node_type": "group",
                    "zarr_format": zarr_format,
                },
            }

        object.__setattr__(
            expected_group_values[0].metadata,
            "consolidated_metadata",
            ConsolidatedMetadata.from_dict(
                {
                    "kind": "inline",
                    "metadata": metadata,
                    "must_understand": False,
                }
            ),
        )
        object.__setattr__(
            expected_group_values[1].metadata,
            "consolidated_metadata",
            ConsolidatedMetadata(metadata={}),
        )

    result = sorted(group.groups(), key=operator.itemgetter(0))
    assert result == expected_groups

    assert sorted(group.groups(), key=operator.itemgetter(0)) == expected_groups
    assert sorted(group.group_keys()) == expected_group_keys
    assert sorted(group.group_values(), key=lambda x: x.name) == expected_group_values

    assert sorted(group.arrays(), key=operator.itemgetter(0)) == expected_arrays
    assert sorted(group.array_keys()) == expected_array_keys
    assert sorted(group.array_values(), key=lambda x: x.name) == expected_array_values


def test_group_update_attributes(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group.update_attributes`
    """
    attrs = {"foo": 100}
    group = Group.from_store(store, zarr_format=zarr_format, attributes=attrs)
    assert group.attrs == attrs
    new_attrs = {"bar": 100}
    new_group = group.update_attributes(new_attrs)

    updated_attrs = attrs.copy()
    updated_attrs.update(new_attrs)
    assert new_group.attrs == updated_attrs


async def test_group_update_attributes_async(store: Store, zarr_format: ZarrFormat) -> None:
    """
    Test the behavior of `Group.update_attributes_async`
    """
    attrs = {"foo": 100}
    group = Group.from_store(store, zarr_format=zarr_format, attributes=attrs)
    assert group.attrs == attrs
    new_attrs = {"bar": 100}
    new_group = await group.update_attributes_async(new_attrs)
    assert new_group.attrs == new_attrs


@pytest.mark.parametrize("method", ["create_array", "array"])
@pytest.mark.parametrize("name", ["a", "/a"])
def test_group_create_array(
    store: Store,
    zarr_format: ZarrFormat,
    overwrite: bool,
    method: Literal["create_array", "array"],
    name: str,
) -> None:
    """
    Test `Group.from_store`
    """
    group = Group.from_store(store, zarr_format=zarr_format)
    shape = (10, 10)
    dtype = "uint8"
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)

    if method == "create_array":
        array = group.create_array(name=name, shape=shape, dtype=dtype)
        array[:] = data
    elif method == "array":
        with pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."):
            array = group.array(name=name, data=data, shape=shape, dtype=dtype)
    else:
        raise AssertionError

    if not overwrite:
        if method == "create_array":
            with pytest.raises(ContainsArrayError):  # noqa: PT012
                a = group.create_array(name=name, shape=shape, dtype=dtype)
                a[:] = data
        elif method == "array":
            with pytest.raises(ContainsArrayError):  # noqa: PT012
                with pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."):
                    a = group.array(name=name, shape=shape, dtype=dtype)
                a[:] = data

    assert array.path == normalize_path(name)
    assert array.name == "/" + array.path
    assert array.shape == shape
    assert array.dtype == np.dtype(dtype)
    assert np.array_equal(array[:], data)


def test_group_array_creation(
    store: Store,
    zarr_format: ZarrFormat,
):
    group = Group.from_store(store, zarr_format=zarr_format)
    shape = (10, 10)
    empty_array = group.empty(name="empty", shape=shape)
    assert isinstance(empty_array, Array)
    assert empty_array.fill_value == 0
    assert empty_array.shape == shape
    assert empty_array.store_path.store == store
    assert empty_array.store_path.path == "empty"

    empty_like_array = group.empty_like(name="empty_like", data=empty_array)
    assert isinstance(empty_like_array, Array)
    assert empty_like_array.fill_value == 0
    assert empty_like_array.shape == shape
    assert empty_like_array.store_path.store == store

    empty_array_bool = group.empty(name="empty_bool", shape=shape, dtype=np.dtype("bool"))
    assert isinstance(empty_array_bool, Array)
    assert not empty_array_bool.fill_value
    assert empty_array_bool.shape == shape
    assert empty_array_bool.store_path.store == store

    empty_like_array_bool = group.empty_like(name="empty_like_bool", data=empty_array_bool)
    assert isinstance(empty_like_array_bool, Array)
    assert not empty_like_array_bool.fill_value
    assert empty_like_array_bool.shape == shape
    assert empty_like_array_bool.store_path.store == store

    zeros_array = group.zeros(name="zeros", shape=shape)
    assert isinstance(zeros_array, Array)
    assert zeros_array.fill_value == 0
    assert zeros_array.shape == shape
    assert zeros_array.store_path.store == store

    zeros_like_array = group.zeros_like(name="zeros_like", data=zeros_array)
    assert isinstance(zeros_like_array, Array)
    assert zeros_like_array.fill_value == 0
    assert zeros_like_array.shape == shape
    assert zeros_like_array.store_path.store == store

    ones_array = group.ones(name="ones", shape=shape)
    assert isinstance(ones_array, Array)
    assert ones_array.fill_value == 1
    assert ones_array.shape == shape
    assert ones_array.store_path.store == store

    ones_like_array = group.ones_like(name="ones_like", data=ones_array)
    assert isinstance(ones_like_array, Array)
    assert ones_like_array.fill_value == 1
    assert ones_like_array.shape == shape
    assert ones_like_array.store_path.store == store

    full_array = group.full(name="full", shape=shape, fill_value=42)
    assert isinstance(full_array, Array)
    assert full_array.fill_value == 42
    assert full_array.shape == shape
    assert full_array.store_path.store == store

    full_like_array = group.full_like(name="full_like", data=full_array, fill_value=43)
    assert isinstance(full_like_array, Array)
    assert full_like_array.fill_value == 43
    assert full_like_array.shape == shape
    assert full_like_array.store_path.store == store


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("extant_node", ["array", "group"])
def test_group_creation_existing_node(
    store: Store,
    zarr_format: ZarrFormat,
    overwrite: bool,
    extant_node: Literal["array", "group"],
) -> None:
    """
    Check that an existing array or group is handled as expected during group creation.
    """
    spath = StorePath(store)
    group = Group.from_store(spath, zarr_format=zarr_format)
    expected_exception: type[ContainsArrayError | ContainsGroupError]
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

    if overwrite:
        if not store.supports_deletes:
            pytest.skip("store does not support deletes but overwrite is True")
        node_new = Group.from_store(
            spath / "extant",
            attributes=new_attributes,
            zarr_format=zarr_format,
            overwrite=overwrite,
        )
        assert node_new.attrs == new_attributes
    else:
        with pytest.raises(expected_exception):
            node_new = Group.from_store(
                spath / "extant",
                attributes=new_attributes,
                zarr_format=zarr_format,
                overwrite=overwrite,
            )


async def test_asyncgroup_create(
    store: Store,
    overwrite: bool,
    zarr_format: ZarrFormat,
) -> None:
    """
    Test that `AsyncGroup.from_store` works as expected.
    """
    spath = StorePath(store=store)
    attributes = {"foo": 100}
    agroup = await AsyncGroup.from_store(
        store,
        attributes=attributes,
        overwrite=overwrite,
        zarr_format=zarr_format,
    )

    assert agroup.metadata == GroupMetadata(zarr_format=zarr_format, attributes=attributes)
    assert agroup.store_path == await make_store_path(store)

    if not overwrite:
        with pytest.raises(ContainsGroupError):
            agroup = await AsyncGroup.from_store(
                spath,
                attributes=attributes,
                overwrite=overwrite,
                zarr_format=zarr_format,
            )
        # create an array at our target path
        collision_name = "foo"
        _ = await zarr.api.asynchronous.create_array(
            spath / collision_name, shape=(10,), dtype="uint8", zarr_format=zarr_format
        )
        with pytest.raises(ContainsArrayError):
            _ = await AsyncGroup.from_store(
                StorePath(store=store) / collision_name,
                attributes=attributes,
                overwrite=overwrite,
                zarr_format=zarr_format,
            )


async def test_asyncgroup_attrs(store: Store, zarr_format: ZarrFormat) -> None:
    attributes = {"foo": 100}
    agroup = await AsyncGroup.from_store(store, zarr_format=zarr_format, attributes=attributes)

    assert agroup.attrs == agroup.metadata.attributes == attributes


async def test_asyncgroup_open(
    store: Store,
    zarr_format: ZarrFormat,
) -> None:
    """
    Create an `AsyncGroup`, then ensure that we can open it using `AsyncGroup.open`
    """
    attributes = {"foo": 100}
    group_w = await AsyncGroup.from_store(
        store=store,
        attributes=attributes,
        overwrite=False,
        zarr_format=zarr_format,
    )

    group_r = await AsyncGroup.open(store=store, zarr_format=zarr_format)

    assert group_w.attrs == group_w.attrs == attributes
    assert group_w == group_r


async def test_asyncgroup_open_wrong_format(
    store: Store,
    zarr_format: ZarrFormat,
) -> None:
    _ = await AsyncGroup.from_store(store=store, overwrite=False, zarr_format=zarr_format)
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
    [
        {"zarr_format": 3, "node_type": "group", "attributes": {"foo": 100}},
        {"zarr_format": 2, "attributes": {"foo": 100}},
    ],
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
    agroup = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)

    array_name = "sub_array"
    sub_array = await agroup.create_array(name=array_name, shape=(10,), dtype="uint8", chunks=(2,))
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

    agroup = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)
    array_name = "sub_array"
    _ = await agroup.create_array(
        name=array_name,
        shape=(10,),
        dtype="uint8",
        chunks=(2,),
        attributes={"foo": 100},
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


@pytest.mark.parametrize("name", ["a", "/a"])
async def test_asyncgroup_create_group(
    store: Store,
    name: str,
    zarr_format: ZarrFormat,
) -> None:
    agroup = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)
    attributes = {"foo": 999}
    subgroup = await agroup.create_group(name=name, attributes=attributes)

    assert isinstance(subgroup, AsyncGroup)
    assert subgroup.path == normalize_path(name)
    assert subgroup.name == "/" + subgroup.path
    assert subgroup.attrs == attributes
    assert subgroup.store_path.path == subgroup.path
    assert subgroup.store_path.store == store
    assert subgroup.metadata.zarr_format == zarr_format


async def test_asyncgroup_create_array(
    store: Store, zarr_format: ZarrFormat, overwrite: bool
) -> None:
    """
    Test that the AsyncGroup.create_array method works correctly. We ensure that array properties
    specified in create_array are present on the resulting array.
    """

    agroup = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)

    if not overwrite:
        with pytest.raises(ContainsGroupError):
            agroup = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)

    shape = (10,)
    dtype = "uint8"
    chunk_shape = (4,)
    attributes: dict[str, JSON] = {"foo": 100}

    sub_node_path = "sub_array"
    subnode = await agroup.create_array(
        name=sub_node_path,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
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
    agroup = await AsyncGroup.from_store(
        store=store, zarr_format=zarr_format, attributes=attributes_old
    )

    agroup_new_attributes = await agroup.update_attributes(attributes_new)
    attributes_updated = attributes_old.copy()
    attributes_updated.update(attributes_new)
    assert agroup_new_attributes.attrs == attributes_updated


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_serializable_async_group(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = await AsyncGroup.from_store(
        store=store, attributes={"foo": 999}, zarr_format=zarr_format
    )
    p = pickle.dumps(expected)
    actual = pickle.loads(p)
    assert actual == expected


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_serializable_sync_group(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = Group.from_store(store=store, attributes={"foo": 999}, zarr_format=zarr_format)
    p = pickle.dumps(expected)
    actual = pickle.loads(p)
    assert actual == expected


@pytest.mark.parametrize("consolidated_metadata", [True, False])
async def test_group_members_async(store: Store, consolidated_metadata: bool) -> None:
    group = await AsyncGroup.from_store(
        store=store,
    )
    a0 = await group.create_array("a0", shape=(1,), dtype="uint8")
    g0 = await group.create_group("g0")
    a1 = await g0.create_array("a1", shape=(1,), dtype="uint8")
    g1 = await g0.create_group("g1")
    a2 = await g1.create_array("a2", shape=(1,), dtype="uint8")
    g2 = await g1.create_group("g2")

    # immediate children
    children = sorted([x async for x in group.members()], key=operator.itemgetter(0))
    assert children == [
        ("a0", a0),
        ("g0", g0),
    ]

    nmembers = await group.nmembers()
    assert nmembers == 2

    # partial
    children = sorted([x async for x in group.members(max_depth=1)], key=operator.itemgetter(0))
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
    all_children = sorted(
        [x async for x in group.members(max_depth=None)], key=operator.itemgetter(0)
    )
    expected = [
        ("a0", a0),
        ("g0", g0),
        ("g0/a1", a1),
        ("g0/g1", g1),
        ("g0/g1/a2", a2),
        ("g0/g1/g2", g2),
    ]
    assert all_children == expected

    if consolidated_metadata:
        await zarr.api.asynchronous.consolidate_metadata(store=store)
        group = await zarr.api.asynchronous.open_group(store=store)

    nmembers = await group.nmembers(max_depth=None)
    assert nmembers == 6

    with pytest.raises(ValueError, match="max_depth"):
        [x async for x in group.members(max_depth=-1)]

    if consolidated_metadata:
        # test for mixed known and unknown metadata.
        # For now, we trust the consolidated metadata.
        object.__setattr__(
            group.metadata.consolidated_metadata.metadata["g0"].consolidated_metadata.metadata[
                "g1"
            ],
            "consolidated_metadata",
            None,
        )
        # test depth=0
        nmembers = await group.nmembers(max_depth=0)
        assert nmembers == 2
        # test depth=1
        nmembers = await group.nmembers(max_depth=1)
        assert nmembers == 4
        # test depth=None
        all_children = sorted(
            [x async for x in group.members(max_depth=None)], key=operator.itemgetter(0)
        )
        assert len(all_children) == 4
        nmembers = await group.nmembers(max_depth=None)
        assert nmembers == 4
        # test depth<0
        with pytest.raises(ValueError, match="max_depth"):
            await group.nmembers(max_depth=-1)


async def test_require_group(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)

    # create foo group
    _ = await root.create_group("foo", attributes={"foo": 100})

    # test that we can get the group using require_group
    foo_group = await root.require_group("foo")
    assert foo_group.attrs == {"foo": 100}

    # test that we can get the group using require_group and overwrite=True
    if store.supports_deletes:
        foo_group = await root.require_group("foo", overwrite=True)
        assert foo_group.attrs == {}

    _ = await foo_group.create_array(
        "bar", shape=(10,), dtype="uint8", chunks=(2,), attributes={"foo": 100}
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
    root = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)
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


def test_create_dataset_with_data(store: Store, zarr_format: ZarrFormat) -> None:
    """Check that deprecated create_dataset method allows input data.

    See https://github.com/zarr-developers/zarr-python/issues/2631.
    """
    root = Group.from_store(store=store, zarr_format=zarr_format)
    arr = np.random.random((5, 5))
    with pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."):
        data = root.create_dataset("random", data=arr, shape=arr.shape)
    np.testing.assert_array_equal(np.asarray(data), arr)


async def test_create_dataset(store: Store, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)
    with pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."):
        foo = await root.create_dataset("foo", shape=(10,), dtype="uint8")
    assert foo.shape == (10,)

    with (
        pytest.raises(ContainsArrayError),
        pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."),
    ):
        await root.create_dataset("foo", shape=(100,), dtype="int8")

    _ = await root.create_group("bar")
    with (
        pytest.raises(ContainsGroupError),
        pytest.warns(DeprecationWarning, match=r"Group\.create_array instead\."),
    ):
        await root.create_dataset("bar", shape=(100,), dtype="int8")


async def test_require_array(store: Store, zarr_format: ZarrFormat) -> None:
    root = await AsyncGroup.from_store(store=store, zarr_format=zarr_format)
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


@pytest.mark.parametrize("consolidate", [True, False])
async def test_members_name(store: Store, consolidate: bool, zarr_format: ZarrFormat):
    group = Group.from_store(store=store, zarr_format=zarr_format)
    a = group.create_group(name="a")
    a.create_array("array", shape=(1,), dtype="uint8")
    b = a.create_group(name="b")
    b.create_array("array", shape=(1,), dtype="uint8")

    if consolidate:
        group = zarr.api.synchronous.consolidate_metadata(store)

    result = group["a"]["b"]
    assert result.name == "/a/b"

    paths = sorted(x.name for _, x in group.members(max_depth=None))
    expected = ["/a", "/a/array", "/a/b", "/a/b/array"]
    assert paths == expected

    # regression test for https://github.com/zarr-developers/zarr-python/pull/2356
    g = zarr.open_group(store, use_consolidated=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(g)


async def test_open_mutable_mapping():
    group = await zarr.api.asynchronous.open_group(
        store={},
    )
    assert isinstance(group.store_path.store, MemoryStore)


def test_open_mutable_mapping_sync():
    group = zarr.open_group(
        store={},
    )
    assert isinstance(group.store_path.store, MemoryStore)


class TestConsolidated:
    async def test_group_getitem_consolidated(self, store: Store) -> None:
        root = await AsyncGroup.from_store(store=store)
        # Set up the test structure with
        # /
        #  g0/      # group /g0
        #    g1/    # group /g0/g1
        #      g2/  # group /g0/g1/g2
        #  x1/      # group /x0
        #    x2/    # group /x0/x1
        #      x3/  # group /x0/x1/x2

        g0 = await root.create_group("g0")
        g1 = await g0.create_group("g1")
        await g1.create_group("g2")

        x0 = await root.create_group("x0")
        x1 = await x0.create_group("x1")
        await x1.create_group("x2")

        await zarr.api.asynchronous.consolidate_metadata(store)

        # On disk, we've consolidated all the metadata in the root zarr.json
        group = await zarr.api.asynchronous.open(store=store)
        rg0 = await group.getitem("g0")

        expected = ConsolidatedMetadata(
            metadata={
                "g1": GroupMetadata(
                    attributes={},
                    zarr_format=3,
                    consolidated_metadata=ConsolidatedMetadata(
                        metadata={
                            "g2": GroupMetadata(
                                attributes={},
                                zarr_format=3,
                                consolidated_metadata=ConsolidatedMetadata(metadata={}),
                            )
                        }
                    ),
                ),
            }
        )
        assert rg0.metadata.consolidated_metadata == expected

        rg1 = await rg0.getitem("g1")
        assert rg1.metadata.consolidated_metadata == expected.metadata["g1"].consolidated_metadata

        rg2 = await rg1.getitem("g2")
        assert rg2.metadata.consolidated_metadata == ConsolidatedMetadata(metadata={})

    async def test_group_delitem_consolidated(self, store: Store) -> None:
        if isinstance(store, ZipStore):
            raise pytest.skip("Not implemented")

        root = await AsyncGroup.from_store(store=store)
        # Set up the test structure with
        # /
        #  g0/         # group /g0
        #    g1/       # group /g0/g1
        #      g2/     # group /g0/g1/g2
        #        data  # array
        #  x1/         # group /x0
        #    x2/       # group /x0/x1
        #      x3/     # group /x0/x1/x2
        #        data  # array

        g0 = await root.create_group("g0")
        g1 = await g0.create_group("g1")
        g2 = await g1.create_group("g2")
        await g2.create_array("data", shape=(1,), dtype="uint8")

        x0 = await root.create_group("x0")
        x1 = await x0.create_group("x1")
        x2 = await x1.create_group("x2")
        await x2.create_array("data", shape=(1,), dtype="uint8")

        await zarr.api.asynchronous.consolidate_metadata(store)

        group = await zarr.api.asynchronous.open_consolidated(store=store)
        assert len(group.metadata.consolidated_metadata.metadata) == 2
        assert "g0" in group.metadata.consolidated_metadata.metadata

        await group.delitem("g0")
        assert len(group.metadata.consolidated_metadata.metadata) == 1
        assert "g0" not in group.metadata.consolidated_metadata.metadata

    def test_open_consolidated_raises(self, store: Store) -> None:
        if isinstance(store, ZipStore):
            raise pytest.skip("Not implemented")

        root = Group.from_store(store=store)

        # fine to be missing by default
        zarr.open_group(store=store)

        with pytest.raises(ValueError, match="Consolidated metadata requested."):
            zarr.open_group(store=store, use_consolidated=True)

        # Now create consolidated metadata...
        root.create_group("g0")
        zarr.consolidate_metadata(store)

        # and explicitly ignore it.
        group = zarr.open_group(store=store, use_consolidated=False)
        assert group.metadata.consolidated_metadata is None

    async def test_open_consolidated_raises_async(self, store: Store) -> None:
        if isinstance(store, ZipStore):
            raise pytest.skip("Not implemented")

        root = await AsyncGroup.from_store(store=store)

        # fine to be missing by default
        await zarr.api.asynchronous.open_group(store=store)

        with pytest.raises(ValueError, match="Consolidated metadata requested."):
            await zarr.api.asynchronous.open_group(store=store, use_consolidated=True)

        # Now create consolidated metadata...
        await root.create_group("g0")
        await zarr.api.asynchronous.consolidate_metadata(store)

        # and explicitly ignore it.
        group = await zarr.api.asynchronous.open_group(store=store, use_consolidated=False)
        assert group.metadata.consolidated_metadata is None


class TestGroupMetadata:
    def test_from_dict_extra_fields(self):
        data = {
            "attributes": {"key": "value"},
            "_nczarr_superblock": {"version": "2.0.0"},
            "zarr_format": 2,
        }
        result = GroupMetadata.from_dict(data)
        expected = GroupMetadata(attributes={"key": "value"}, zarr_format=2)
        assert result == expected


class TestInfo:
    def test_info(self):
        store = zarr.storage.MemoryStore()
        A = zarr.group(store=store, path="A")
        B = A.create_group(name="B")

        B.create_array(name="x", shape=(1,), dtype="uint8")
        B.create_array(name="y", shape=(2,), dtype="uint8")

        result = A.info
        expected = GroupInfo(
            _name="A",
            _read_only=False,
            _store_type="MemoryStore",
            _zarr_format=3,
        )
        assert result == expected

        result = A.info_complete()
        expected = GroupInfo(
            _name="A",
            _read_only=False,
            _store_type="MemoryStore",
            _zarr_format=3,
            _count_members=3,
            _count_arrays=2,
            _count_groups=1,
        )
        assert result == expected


def test_update_attrs() -> None:
    # regression test for https://github.com/zarr-developers/zarr-python/issues/2328
    root = Group.from_store(
        MemoryStore(),
    )
    root.attrs["foo"] = "bar"
    assert root.attrs["foo"] == "bar"


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_delitem_removes_children(store: Store, zarr_format: ZarrFormat) -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2191
    g1 = zarr.group(store=store, zarr_format=zarr_format)
    g1.create_group("0")
    g1.create_group("0/0")
    arr = g1.create_array("0/0/0", shape=(1,), dtype="uint8")
    arr[:] = 1
    del g1["0"]
    with pytest.raises(KeyError):
        g1["0/0"]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_nodes(
    impl: Literal["async", "sync"], store: Store, zarr_format: ZarrFormat
) -> None:
    """
    Ensure that ``create_nodes`` can create a zarr hierarchy from a model of that
    hierarchy in dict form. Note that this creates an incomplete Zarr hierarchy.
    """
    node_spec = {
        "group": GroupMetadata(attributes={"foo": 10}),
        "group/array_0": meta_from_array(np.arange(3), zarr_format=zarr_format),
        "group/array_1": meta_from_array(np.arange(4), zarr_format=zarr_format),
        "group/subgroup/array_0": meta_from_array(np.arange(4), zarr_format=zarr_format),
        "group/subgroup/array_1": meta_from_array(np.arange(5), zarr_format=zarr_format),
    }
    if impl == "sync":
        observed_nodes = dict(sync_group.create_nodes(store=store, nodes=node_spec))
    elif impl == "async":
        observed_nodes = dict(await _collect_aiterator(create_nodes(store=store, nodes=node_spec)))
    else:
        raise ValueError(f"Invalid impl: {impl}")

    assert node_spec == {k: v.metadata for k, v in observed_nodes.items()}


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_create_nodes_concurrency_limit(store: MemoryStore) -> None:
    """
    Test that the execution time of create_nodes can be constrained by the async concurrency
    configuration setting.
    """
    set_latency = 0.02
    num_groups = 10
    groups = {str(idx): GroupMetadata() for idx in range(num_groups)}

    latency_store = LatencyStore(store, set_latency=set_latency)

    # check how long it takes to iterate over the groups
    # if create_nodes is sensitive to IO latency,
    # this should take (num_groups * get_latency) seconds
    # otherwise, it should take only marginally more than get_latency seconds
    with zarr_config.set({"async.concurrency": 1}):
        start = time.time()
        _ = tuple(sync_group.create_nodes(store=latency_store, nodes=groups))
        elapsed = time.time() - start
        assert elapsed > num_groups * set_latency


@pytest.mark.parametrize(
    ("a_func", "b_func"),
    [
        (zarr.core.group.AsyncGroup.create_array, zarr.core.group.Group.create_array),
        (zarr.core.group.AsyncGroup.create_hierarchy, zarr.core.group.Group.create_hierarchy),
        (zarr.core.group.create_hierarchy, zarr.core.sync_group.create_hierarchy),
        (zarr.core.group.create_nodes, zarr.core.sync_group.create_nodes),
        (zarr.core.group.create_rooted_hierarchy, zarr.core.sync_group.create_rooted_hierarchy),
        (zarr.core.group.get_node, zarr.core.sync_group.get_node),
    ],
)
def test_consistent_signatures(
    a_func: Callable[[object], object], b_func: Callable[[object], object]
) -> None:
    """
    Ensure that pairs of functions have consistent signatures
    """
    base_sig = inspect.signature(a_func)
    test_sig = inspect.signature(b_func)
    wrong: dict[str, list[object]] = {
        "missing_from_test": [],
        "missing_from_base": [],
        "wrong_type": [],
    }
    for key, value in base_sig.parameters.items():
        if key not in test_sig.parameters:
            wrong["missing_from_test"].append((key, value))
    for key, value in test_sig.parameters.items():
        if key not in base_sig.parameters:
            wrong["missing_from_base"].append((key, value))
        if base_sig.parameters[key] != value:
            wrong["wrong_type"].append({key: {"test": value, "base": base_sig.parameters[key]}})
    assert wrong["missing_from_base"] == []
    assert wrong["missing_from_test"] == []
    assert wrong["wrong_type"] == []


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_hierarchy(
    impl: Literal["async", "sync"], store: Store, overwrite: bool, zarr_format: ZarrFormat
) -> None:
    """
    Test that ``create_hierarchy`` can create a complete Zarr hierarchy, even if the input describes
    an incomplete one.
    """

    hierarchy_spec = {
        "group": GroupMetadata(attributes={"path": "group"}, zarr_format=zarr_format),
        "group/array_0": meta_from_array(
            np.arange(3), attributes={"path": "group/array_0"}, zarr_format=zarr_format
        ),
        "group/subgroup/array_0": meta_from_array(
            np.arange(4), attributes={"path": "group/subgroup/array_0"}, zarr_format=zarr_format
        ),
    }
    pre_existing_nodes = {
        "group/extra": GroupMetadata(zarr_format=zarr_format, attributes={"path": "group/extra"}),
        "": GroupMetadata(zarr_format=zarr_format, attributes={"name": "root"}),
    }
    # we expect create_hierarchy to insert a group that was missing from the hierarchy spec
    expected_meta = hierarchy_spec | {"group/subgroup": GroupMetadata(zarr_format=zarr_format)}

    # initialize the group with some nodes
    _ = dict(sync_group.create_nodes(store=store, nodes=pre_existing_nodes))

    if impl == "sync":
        created = dict(
            sync_group.create_hierarchy(store=store, nodes=hierarchy_spec, overwrite=overwrite)
        )
    elif impl == "async":
        created = {
            k: v
            async for k, v in create_hierarchy(
                store=store, nodes=hierarchy_spec, overwrite=overwrite
            )
        }
    else:
        raise ValueError(f"Invalid impl: {impl}")
    if not overwrite:
        extra_group = sync_group.get_node(store=store, path="group/extra", zarr_format=zarr_format)
        assert extra_group.metadata.attributes == {"path": "group/extra"}
    else:
        with pytest.raises(FileNotFoundError):
            await get_node(store=store, path="group/extra", zarr_format=zarr_format)
    assert expected_meta == {k: v.metadata for k, v in created.items()}


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("extant_node", ["array", "group"])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_hierarchy_existing_nodes(
    impl: Literal["async", "sync"],
    store: Store,
    extant_node: Literal["array", "group"],
    zarr_format: ZarrFormat,
) -> None:
    """
    Test that create_hierarchy with overwrite = False will not overwrite an existing array or group,
    and raises an exception instead.
    """
    extant_node_path = "node"

    if extant_node == "array":
        extant_metadata = meta_from_array(
            np.zeros(4), zarr_format=zarr_format, attributes={"extant": True}
        )
        new_metadata = meta_from_array(np.zeros(4), zarr_format=zarr_format)
        err_cls = ContainsArrayError
    else:
        extant_metadata = GroupMetadata(zarr_format=zarr_format, attributes={"extant": True})
        new_metadata = GroupMetadata(zarr_format=zarr_format)
        err_cls = ContainsGroupError

    # write the extant metadata
    tuple(sync_group.create_nodes(store=store, nodes={extant_node_path: extant_metadata}))

    msg = f"{extant_node} exists in store {store!r} at path {extant_node_path!r}."
    # ensure that we cannot invoke create_hierarchy with overwrite=False here
    if impl == "sync":
        with pytest.raises(err_cls, match=re.escape(msg)):
            tuple(
                sync_group.create_hierarchy(
                    store=store, nodes={"node": new_metadata}, overwrite=False
                )
            )
    elif impl == "async":
        with pytest.raises(err_cls, match=re.escape(msg)):
            tuple(
                [
                    x
                    async for x in create_hierarchy(
                        store=store, nodes={"node": new_metadata}, overwrite=False
                    )
                ]
            )
    else:
        raise ValueError(f"Invalid impl: {impl}")

    # ensure that the extant metadata was not overwritten
    assert (
        await get_node(store=store, path=extant_node_path, zarr_format=zarr_format)
    ).metadata.attributes == {"extant": True}


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("group_path", ["", "foo"])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_group_create_hierarchy(
    store: Store,
    zarr_format: ZarrFormat,
    overwrite: bool,
    group_path: str,
    impl: Literal["async", "sync"],
) -> None:
    """
    Test that the Group.create_hierarchy method creates specified nodes and returns them in a dict.
    Also test that off-target nodes are not deleted, and that the root group is not deleted
    """
    root_attrs = {"root": True}
    g = sync_group.create_rooted_hierarchy(
        store=store,
        nodes={group_path: GroupMetadata(zarr_format=zarr_format, attributes=root_attrs)},
    )
    node_spec = {
        "a": GroupMetadata(zarr_format=zarr_format, attributes={"name": "a"}),
        "a/b": GroupMetadata(zarr_format=zarr_format, attributes={"name": "a/b"}),
        "a/b/c": meta_from_array(
            np.zeros(5), zarr_format=zarr_format, attributes={"name": "a/b/c"}
        ),
    }
    # This node should be kept if overwrite is True
    extant_spec = {"b": GroupMetadata(zarr_format=zarr_format, attributes={"name": "b"})}
    if impl == "async":
        extant_created = dict(
            await _collect_aiterator(g._async_group.create_hierarchy(extant_spec, overwrite=False))
        )
        nodes_created = dict(
            await _collect_aiterator(
                g._async_group.create_hierarchy(node_spec, overwrite=overwrite)
            )
        )
    elif impl == "sync":
        extant_created = dict(g.create_hierarchy(extant_spec, overwrite=False))
        nodes_created = dict(g.create_hierarchy(node_spec, overwrite=overwrite))

    all_members = dict(g.members(max_depth=None))
    for k, v in node_spec.items():
        assert all_members[k].metadata == v == nodes_created[k].metadata

    # if overwrite is True, the extant nodes should be erased
    for k, v in extant_spec.items():
        if overwrite:
            assert k in all_members
        else:
            assert all_members[k].metadata == v == extant_created[k].metadata
    # ensure that we left the root group as-is
    assert (
        sync_group.get_node(store=store, path=group_path, zarr_format=zarr_format).attrs.asdict()
        == root_attrs
    )


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("overwrite", [True, False])
def test_group_create_hierarchy_no_root(
    store: Store, zarr_format: ZarrFormat, overwrite: bool
) -> None:
    """
    Test that the Group.create_hierarchy method will error if the dict provided contains a root.
    """
    g = Group.from_store(store, zarr_format=zarr_format)
    tree = {
        "": GroupMetadata(zarr_format=zarr_format, attributes={"name": "a"}),
    }
    with pytest.raises(
        ValueError, match="It is an error to use this method to create a root node. "
    ):
        _ = dict(g.create_hierarchy(tree, overwrite=overwrite))


class TestParseHierarchyDict:
    """
    Tests for the function that parses dicts of str : Metadata pairs, ensuring that the output models a
    valid Zarr hierarchy
    """

    @staticmethod
    def test_normed_keys() -> None:
        """
        Test that keys get normalized properly
        """

        nodes = {
            "a": GroupMetadata(),
            "/b": GroupMetadata(),
            "": GroupMetadata(),
            "/a//c////": GroupMetadata(),
        }
        observed = _parse_hierarchy_dict(data=nodes)
        expected = {normalize_path(k): v for k, v in nodes.items()}
        assert observed == expected

    @staticmethod
    def test_empty() -> None:
        """
        Test that an empty dict passes through
        """
        assert _parse_hierarchy_dict(data={}) == {}

    @staticmethod
    def test_implicit_groups() -> None:
        """
        Test that implicit groups were added as needed.
        """
        requested = {"a/b/c": GroupMetadata()}
        expected = requested | {
            "": ImplicitGroupMarker(),
            "a": ImplicitGroupMarker(),
            "a/b": ImplicitGroupMarker(),
        }
        observed = _parse_hierarchy_dict(data=requested)
        assert observed == expected


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_group_create_hierarchy_invalid_mixed_zarr_format(
    store: Store, zarr_format: ZarrFormat
) -> None:
    """
    Test that ``Group.create_hierarchy`` will raise an error if the zarr_format of the nodes is
    different from the parent group.
    """
    other_format = 2 if zarr_format == 3 else 3
    g = Group.from_store(store, zarr_format=other_format)
    tree = {
        "a": GroupMetadata(zarr_format=zarr_format, attributes={"name": "a"}),
        "a/b": meta_from_array(np.zeros(5), zarr_format=zarr_format, attributes={"name": "a/c"}),
    }

    msg = "The zarr_format of the nodes must be the same as the parent group."
    with pytest.raises(ValueError, match=msg):
        _ = tuple(g.create_hierarchy(tree))


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("defect", ["array/array", "array/group"])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_hierarchy_invalid_nested(
    impl: Literal["async", "sync"], store: Store, defect: tuple[str, str], zarr_format: ZarrFormat
) -> None:
    """
    Test that create_hierarchy will not create a Zarr array that contains a Zarr group
    or Zarr array.
    """
    if defect == "array/array":
        hierarchy_spec = {
            "array_0": meta_from_array(np.arange(3), zarr_format=zarr_format),
            "array_0/subarray": meta_from_array(np.arange(4), zarr_format=zarr_format),
        }
    elif defect == "array/group":
        hierarchy_spec = {
            "array_0": meta_from_array(np.arange(3), zarr_format=zarr_format),
            "array_0/subgroup": GroupMetadata(attributes={"foo": 10}, zarr_format=zarr_format),
        }

    msg = "Only Zarr groups can contain other nodes."
    if impl == "sync":
        with pytest.raises(ValueError, match=msg):
            tuple(sync_group.create_hierarchy(store=store, nodes=hierarchy_spec))
    elif impl == "async":
        with pytest.raises(ValueError, match=msg):
            await _collect_aiterator(create_hierarchy(store=store, nodes=hierarchy_spec))


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_hierarchy_invalid_mixed_format(
    impl: Literal["async", "sync"], store: Store
) -> None:
    """
    Test that create_hierarchy will not create a Zarr group that contains a both Zarr v2 and
    Zarr v3 nodes.
    """
    msg = (
        "Got data with both Zarr v2 and Zarr v3 nodes, which is invalid. "
        "The following keys map to Zarr v2 nodes: ['v2']. "
        "The following keys map to Zarr v3 nodes: ['v3']."
        "Ensure that all nodes have the same Zarr format."
    )
    nodes = {
        "v2": GroupMetadata(zarr_format=2),
        "v3": GroupMetadata(zarr_format=3),
    }
    if impl == "sync":
        with pytest.raises(ValueError, match=re.escape(msg)):
            tuple(
                sync_group.create_hierarchy(
                    store=store,
                    nodes=nodes,
                )
            )
    elif impl == "async":
        with pytest.raises(ValueError, match=re.escape(msg)):
            await _collect_aiterator(
                create_hierarchy(
                    store=store,
                    nodes=nodes,
                )
            )
    else:
        raise ValueError(f"Invalid impl: {impl}")


@pytest.mark.parametrize("store", ["memory", "local"], indirect=True)
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("root_key", ["", "root"])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_rooted_hierarchy_group(
    impl: Literal["async", "sync"], store: Store, zarr_format, root_key: str
) -> None:
    """
    Test that the _create_rooted_hierarchy can create a group.
    """
    root_meta = {root_key: GroupMetadata(zarr_format=zarr_format, attributes={"path": root_key})}
    group_names = ["a", "a/b"]
    array_names = ["a/b/c", "a/b/d"]

    # just to ensure that we don't use the same name twice in tests
    assert set(group_names) & set(array_names) == set()

    groups_expected_meta = {
        _join_paths([root_key, node_name]): GroupMetadata(
            zarr_format=zarr_format, attributes={"path": node_name}
        )
        for node_name in group_names
    }

    arrays_expected_meta = {
        _join_paths([root_key, node_name]): meta_from_array(np.zeros(4), zarr_format=zarr_format)
        for node_name in array_names
    }

    nodes_create = root_meta | groups_expected_meta | arrays_expected_meta
    if impl == "sync":
        g = sync_group.create_rooted_hierarchy(store=store, nodes=nodes_create)
        assert isinstance(g, Group)
        members = g.members(max_depth=None)
    elif impl == "async":
        g = await create_rooted_hierarchy(store=store, nodes=nodes_create)
        assert isinstance(g, AsyncGroup)
        members = await _collect_aiterator(g.members(max_depth=None))
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    assert g.metadata.attributes == {"path": root_key}

    members_observed_meta = {k: v.metadata for k, v in members}
    members_expected_meta_relative = {
        k.removeprefix(root_key).lstrip("/"): v
        for k, v in (groups_expected_meta | arrays_expected_meta).items()
    }
    assert members_observed_meta == members_expected_meta_relative


@pytest.mark.parametrize("store", ["memory", "local"], indirect=True)
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("root_key", ["", "root"])
@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_rooted_hierarchy_array(
    impl: Literal["async", "sync"], store: Store, zarr_format, root_key: str
) -> None:
    """
    Test that _create_rooted_hierarchy can create an array.
    """

    root_meta = {
        root_key: meta_from_array(
            np.arange(3), zarr_format=zarr_format, attributes={"path": root_key}
        )
    }
    nodes_create = root_meta

    if impl == "sync":
        a = sync_group.create_rooted_hierarchy(store=store, nodes=nodes_create, overwrite=True)
        assert isinstance(a, Array)
    elif impl == "async":
        a = await create_rooted_hierarchy(store=store, nodes=nodes_create, overwrite=True)
        assert isinstance(a, AsyncArray)
    else:
        raise ValueError(f"Invalid impl: {impl}")
    assert a.metadata.attributes == {"path": root_key}


@pytest.mark.parametrize("impl", ["async", "sync"])
async def test_create_rooted_hierarchy_invalid(impl: Literal["async", "sync"]) -> None:
    """
    Ensure _create_rooted_hierarchy will raise a ValueError if the input does not contain
    a root node.
    """
    zarr_format = 3
    nodes = {
        "a": GroupMetadata(zarr_format=zarr_format),
        "b": GroupMetadata(zarr_format=zarr_format),
    }
    msg = "The input does not specify a root node. "
    if impl == "sync":
        with pytest.raises(ValueError, match=msg):
            sync_group.create_rooted_hierarchy(store=store, nodes=nodes)
    elif impl == "async":
        with pytest.raises(ValueError, match=msg):
            await create_rooted_hierarchy(store=store, nodes=nodes)
    else:
        raise ValueError(f"Invalid impl: {impl}")


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_group_members_performance(store: Store) -> None:
    """
    Test that the execution time of Group.members is less than the number of members times the
    latency for accessing each member.
    """
    get_latency = 0.1

    # use the input store to create some groups
    group_create = zarr.group(store=store)
    num_groups = 10

    # Create some groups
    for i in range(num_groups):
        group_create.create_group(f"group{i}")

    latency_store = LatencyStore(store, get_latency=get_latency)
    # create a group with some latency on get operations
    group_read = zarr.group(store=latency_store)

    # check how long it takes to iterate over the groups
    # if .members is sensitive to IO latency,
    # this should take (num_groups * get_latency) seconds
    # otherwise, it should take only marginally more than get_latency seconds
    start = time.time()
    _ = group_read.members()
    elapsed = time.time() - start

    assert elapsed < (num_groups * get_latency)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_group_members_concurrency_limit(store: MemoryStore) -> None:
    """
    Test that the execution time of Group.members can be constrained by the async concurrency
    configuration setting.
    """
    get_latency = 0.02

    # use the input store to create some groups
    group_create = zarr.group(store=store)
    num_groups = 10

    # Create some groups
    for i in range(num_groups):
        group_create.create_group(f"group{i}")

    latency_store = LatencyStore(store, get_latency=get_latency)
    # create a group with some latency on get operations
    group_read = zarr.group(store=latency_store)

    # check how long it takes to iterate over the groups
    # if .members is sensitive to IO latency,
    # this should take (num_groups * get_latency) seconds
    # otherwise, it should take only marginally more than get_latency seconds
    with zarr_config.set({"async.concurrency": 1}):
        start = time.time()
        _ = group_read.members()
        elapsed = time.time() - start

        assert elapsed > num_groups * get_latency


@pytest.mark.parametrize("option", ["array", "group", "invalid"])
def test_build_metadata_v3(option: Literal["array", "group", "invalid"]) -> None:
    """
    Test that _build_metadata_v3 returns the correct metadata for a v3 array or group
    """
    match option:
        case "array":
            metadata_dict = meta_from_array(np.arange(10), zarr_format=3).to_dict()
            assert _build_metadata_v3(metadata_dict) == ArrayV3Metadata.from_dict(metadata_dict)
        case "group":
            metadata_dict = GroupMetadata(attributes={"foo": 10}, zarr_format=3).to_dict()
            assert _build_metadata_v3(metadata_dict) == GroupMetadata.from_dict(metadata_dict)
        case "invalid":
            metadata_dict = GroupMetadata(zarr_format=3).to_dict()
            metadata_dict.pop("node_type")
            # TODO: fix the error message
            msg = "Invalid value for 'node_type'. Expected 'array or group'. Got 'nothing (the key is missing)'."
            with pytest.raises(MetadataValidationError, match=re.escape(msg)):
                _build_metadata_v3(metadata_dict)


@pytest.mark.parametrize("roots", [("",), ("a", "b")])
def test_get_roots(roots: tuple[str, ...]):
    root_nodes = {k: GroupMetadata(attributes={"name": k}) for k in roots}
    child_nodes = {
        _join_paths([k, "foo"]): GroupMetadata(attributes={"name": _join_paths([k, "foo"])})
        for k in roots
    }
    data = root_nodes | child_nodes
    assert set(_get_roots(data)) == set(roots)
