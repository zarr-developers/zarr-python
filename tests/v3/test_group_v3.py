from __future__ import annotations
from typing import TYPE_CHECKING

from zarr.v3.store.core import make_store_path

if TYPE_CHECKING:
    from zarr.v3.store import MemoryStore, LocalStore
    from typing import Literal
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
    Test that `Group.children` returns correct values, i.e. the arrays and groups
    (explicit and implicit) contained in that group.
    """

    path = "group"
    agroup = AsyncGroup(
        metadata=GroupMetadata(),
        store_path=StorePath(store=store, path=path),
    )
    group = Group(agroup)

    subgroup = group.create_group("subgroup")
    # make a sub-sub-subgroup, to ensure that the children calculation doesn't go
    # too deep in the hierarchy
    _ = subgroup.create_group("subsubgroup")
    subarray = group.create_array(
        "subarray", shape=(100,), dtype="uint8", chunk_shape=(10,), exists_ok=True
    )

    # add an extra object to the domain of the group.
    # the list of children should ignore this object.
    sync(store.set(f"{path}/extra_object", b"000000"))
    # add an extra object under a directory-like prefix in the domain of the group.
    # this creates an implicit group called implicit_subgroup
    sync(store.set(f"{path}/implicit_subgroup/extra_object", b"000000"))
    # make the implicit subgroup
    implicit_subgroup = Group(
        AsyncGroup(
            metadata=GroupMetadata(),
            store_path=StorePath(store=store, path=f"{path}/implicit_subgroup"),
        )
    )
    # note: these assertions are order-independent, because it is not clear
    # if group.children guarantees a particular order for the children.
    # If order is not guaranteed, then the better version of this test is
    # to compare two sets, but presently neither the group nor array classes are hashable.
    observed = group.children
    assert len(observed) == 3
    assert subarray in observed
    assert implicit_subgroup in observed
    assert subgroup in observed


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
    group = await AsyncGroup.create(
        store,
        attributes=attributes,
        exists_ok=exists_ok,
        zarr_format=zarr_format,
        runtime_configuration=runtime_configuration,
    )

    assert group.metadata == GroupMetadata(zarr_format=zarr_format, attributes=attributes)
    assert group.store_path == make_store_path(store)
    assert group.runtime_configuration == runtime_configuration

    if not exists_ok:
        with pytest.raises(AssertionError):
            group = await AsyncGroup.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
                runtime_configuration=runtime_configuration,
            )


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
        zarr_format=ZarrFormat,
        runtime_configuration=runtime_configuration,
    )

    group_r = AsyncGroup.open(
        store=store, zarr_format=zarr_format, runtime_configuration=runtime_configuration
    )

    assert group_r == group_w


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", ("2", "3"))
def test_getitem(store: MemoryStore | LocalStore, zarr_format: Literal["2", "3"]):
    ...
