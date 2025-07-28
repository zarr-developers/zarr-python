from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numcodecs import Blosc

import zarr.api.asynchronous
import zarr.api.synchronous
import zarr.storage
from zarr.api.asynchronous import (
    AsyncGroup,
    consolidate_metadata,
    group,
    open,
    open_consolidated,
)
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.core.dtype import parse_dtype
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import ZarrFormat


@pytest.fixture
async def memory_store_with_hierarchy(memory_store: Store) -> None:
    g = await group(store=memory_store, attributes={"foo": "bar"})
    dtype = "uint8"
    await g.create_array(name="air", shape=(1, 2, 3), dtype=dtype)
    await g.create_array(name="lat", shape=(1,), dtype=dtype)
    await g.create_array(name="lon", shape=(2,), dtype=dtype)
    await g.create_array(name="time", shape=(3,), dtype=dtype)

    child = await g.create_group("child", attributes={"key": "child"})
    await child.create_array("array", shape=(4, 4), attributes={"key": "child"}, dtype=dtype)

    grandchild = await child.create_group("grandchild", attributes={"key": "grandchild"})
    await grandchild.create_array(
        "array", shape=(4, 4), attributes={"key": "grandchild"}, dtype=dtype
    )
    await grandchild.create_group("empty_group", attributes={"key": "empty"})
    return memory_store


class TestConsolidated:
    async def test_open_consolidated_false_raises(self):
        store = zarr.storage.MemoryStore()
        with pytest.raises(TypeError, match="use_consolidated"):
            await zarr.api.asynchronous.open_consolidated(store, use_consolidated=False)

    def test_open_consolidated_false_raises_sync(self):
        store = zarr.storage.MemoryStore()
        with pytest.raises(TypeError, match="use_consolidated"):
            zarr.open_consolidated(store, use_consolidated=False)

    async def test_consolidated(self, memory_store_with_hierarchy: Store) -> None:
        # TODO: Figure out desired keys in
        # TODO: variety in the hierarchies
        # More nesting
        # arrays under arrays
        # single array
        # etc.
        await consolidate_metadata(memory_store_with_hierarchy)
        group2 = await AsyncGroup.open(memory_store_with_hierarchy)

        array_metadata = {
            "attributes": {},
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default",
            },
            "codecs": (
                {"configuration": {"endian": "little"}, "name": "bytes"},
                {"configuration": {"level": 0, "checksum": False}, "name": "zstd"},
            ),
            "data_type": "uint8",
            "fill_value": 0,
            "node_type": "array",
            # "shape": (1, 2, 3),
            "zarr_format": 3,
        }

        expected = GroupMetadata(
            attributes={"foo": "bar"},
            consolidated_metadata=ConsolidatedMetadata(
                kind="inline",
                must_understand=False,
                metadata={
                    "air": ArrayV3Metadata.from_dict(
                        {
                            "shape": (1, 2, 3),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1, 2, 3)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "lat": ArrayV3Metadata.from_dict(
                        {
                            "shape": (1,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "lon": ArrayV3Metadata.from_dict(
                        {
                            "shape": (2,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (2,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "time": ArrayV3Metadata.from_dict(
                        {
                            "shape": (3,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (3,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "child": GroupMetadata(
                        attributes={"key": "child"},
                        consolidated_metadata=ConsolidatedMetadata(
                            metadata={
                                "array": ArrayV3Metadata.from_dict(
                                    {
                                        **array_metadata,
                                        "attributes": {"key": "child"},
                                        "shape": (4, 4),
                                        "chunk_grid": {
                                            "configuration": {"chunk_shape": (4, 4)},
                                            "name": "regular",
                                        },
                                    }
                                ),
                                "grandchild": GroupMetadata(
                                    attributes={"key": "grandchild"},
                                    consolidated_metadata=ConsolidatedMetadata(
                                        metadata={
                                            # known to be empty child group
                                            "empty_group": GroupMetadata(
                                                consolidated_metadata=ConsolidatedMetadata(
                                                    metadata={}
                                                ),
                                                attributes={"key": "empty"},
                                            ),
                                            "array": ArrayV3Metadata.from_dict(
                                                {
                                                    **array_metadata,
                                                    "attributes": {"key": "grandchild"},
                                                    "shape": (4, 4),
                                                    "chunk_grid": {
                                                        "configuration": {"chunk_shape": (4, 4)},
                                                        "name": "regular",
                                                    },
                                                }
                                            ),
                                        }
                                    ),
                                ),
                            },
                        ),
                    ),
                },
            ),
        )

        assert group2.metadata == expected
        group3 = await open(store=memory_store_with_hierarchy)
        assert group3.metadata == expected

        group4 = await open_consolidated(store=memory_store_with_hierarchy)
        assert group4.metadata == expected

        result_raw = json.loads(
            (
                await memory_store_with_hierarchy.get(
                    "zarr.json", prototype=default_buffer_prototype()
                )
            ).to_bytes()
        )["consolidated_metadata"]
        assert result_raw["kind"] == "inline"
        assert sorted(result_raw["metadata"]) == [
            "air",
            "child",
            "child/array",
            "child/grandchild",
            "child/grandchild/array",
            "child/grandchild/empty_group",
            "lat",
            "lon",
            "time",
        ]

    def test_consolidated_sync(self, memory_store):
        g = zarr.api.synchronous.group(store=memory_store, attributes={"foo": "bar"})
        dtype = "uint8"
        g.create_array(name="air", shape=(1, 2, 3), dtype=dtype)
        g.create_array(name="lat", shape=(1,), dtype=dtype)
        g.create_array(name="lon", shape=(2,), dtype=dtype)
        g.create_array(name="time", shape=(3,), dtype=dtype)

        zarr.api.synchronous.consolidate_metadata(memory_store)
        group2 = zarr.api.synchronous.Group.open(memory_store)

        array_metadata = {
            "attributes": {},
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default",
            },
            "codecs": (
                {"configuration": {"endian": "little"}, "name": "bytes"},
                {"configuration": {"level": 0, "checksum": False}, "name": "zstd"},
            ),
            "data_type": dtype,
            "fill_value": 0,
            "node_type": "array",
            # "shape": (1, 2, 3),
            "zarr_format": 3,
        }

        expected = GroupMetadata(
            attributes={"foo": "bar"},
            consolidated_metadata=ConsolidatedMetadata(
                kind="inline",
                must_understand=False,
                metadata={
                    "air": ArrayV3Metadata.from_dict(
                        {
                            "shape": (1, 2, 3),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1, 2, 3)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "lat": ArrayV3Metadata.from_dict(
                        {
                            "shape": (1,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "lon": ArrayV3Metadata.from_dict(
                        {
                            "shape": (2,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (2,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "time": ArrayV3Metadata.from_dict(
                        {
                            "shape": (3,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (3,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                },
            ),
        )
        assert group2.metadata == expected
        group3 = zarr.api.synchronous.open(store=memory_store)
        assert group3.metadata == expected

        group4 = zarr.api.synchronous.open_consolidated(store=memory_store)
        assert group4.metadata == expected

    async def test_not_writable_raises(self, memory_store: zarr.storage.MemoryStore) -> None:
        await group(store=memory_store, attributes={"foo": "bar"})
        read_store = zarr.storage.MemoryStore(store_dict=memory_store._store_dict, read_only=True)
        with pytest.raises(ValueError, match="does not support writing"):
            await consolidate_metadata(read_store)

    async def test_non_root_node(self, memory_store_with_hierarchy: Store) -> None:
        await consolidate_metadata(memory_store_with_hierarchy, path="child")
        root = await AsyncGroup.open(memory_store_with_hierarchy)
        child = await AsyncGroup.open(StorePath(memory_store_with_hierarchy) / "child")

        assert root.metadata.consolidated_metadata is None
        assert child.metadata.consolidated_metadata is not None
        assert "air" not in child.metadata.consolidated_metadata.metadata
        assert "grandchild" in child.metadata.consolidated_metadata.metadata

    def test_consolidated_metadata_from_dict(self):
        data = {"must_understand": False}

        # missing kind
        with pytest.raises(ValueError, match="kind='None'"):
            ConsolidatedMetadata.from_dict(data)

        # invalid kind
        data["kind"] = "invalid"
        with pytest.raises(ValueError, match="kind='invalid'"):
            ConsolidatedMetadata.from_dict(data)

        # missing metadata
        data["kind"] = "inline"

        with pytest.raises(TypeError, match="Unexpected type for 'metadata'"):
            ConsolidatedMetadata.from_dict(data)

        data["kind"] = "inline"
        # empty is fine
        data["metadata"] = {}
        ConsolidatedMetadata.from_dict(data)

    def test_flatten(self):
        array_metadata = {
            "attributes": {},
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default",
            },
            "codecs": ({"configuration": {"endian": "little"}, "name": "bytes"},),
            "data_type": "float64",
            "fill_value": np.float64(0.0),
            "node_type": "array",
            # "shape": (1, 2, 3),
            "zarr_format": 3,
        }

        metadata = ConsolidatedMetadata(
            kind="inline",
            must_understand=False,
            metadata={
                "air": ArrayV3Metadata.from_dict(
                    {
                        "shape": (1, 2, 3),
                        "chunk_grid": {
                            "configuration": {"chunk_shape": (1, 2, 3)},
                            "name": "regular",
                        },
                        **array_metadata,
                    }
                ),
                "lat": ArrayV3Metadata.from_dict(
                    {
                        "shape": (1,),
                        "chunk_grid": {
                            "configuration": {"chunk_shape": (1,)},
                            "name": "regular",
                        },
                        **array_metadata,
                    }
                ),
                "child": GroupMetadata(
                    attributes={"key": "child"},
                    consolidated_metadata=ConsolidatedMetadata(
                        metadata={
                            "array": ArrayV3Metadata.from_dict(
                                {
                                    **array_metadata,
                                    "attributes": {"key": "child"},
                                    "shape": (4, 4),
                                    "chunk_grid": {
                                        "configuration": {"chunk_shape": (4, 4)},
                                        "name": "regular",
                                    },
                                }
                            ),
                            "grandchild": GroupMetadata(
                                attributes={"key": "grandchild"},
                                consolidated_metadata=ConsolidatedMetadata(
                                    metadata={
                                        "array": ArrayV3Metadata.from_dict(
                                            {
                                                **array_metadata,
                                                "attributes": {"key": "grandchild"},
                                                "shape": (4, 4),
                                                "chunk_grid": {
                                                    "configuration": {"chunk_shape": (4, 4)},
                                                    "name": "regular",
                                                },
                                            }
                                        )
                                    }
                                ),
                            ),
                        },
                    ),
                ),
            },
        )
        result = metadata.flattened_metadata
        expected = {
            "air": metadata.metadata["air"],
            "lat": metadata.metadata["lat"],
            "child": GroupMetadata(
                attributes={"key": "child"}, consolidated_metadata=ConsolidatedMetadata(metadata={})
            ),
            "child/array": metadata.metadata["child"].consolidated_metadata.metadata["array"],
            "child/grandchild": GroupMetadata(
                attributes={"key": "grandchild"},
                consolidated_metadata=ConsolidatedMetadata(metadata={}),
            ),
            "child/grandchild/array": (
                metadata.metadata["child"]
                .consolidated_metadata.metadata["grandchild"]
                .consolidated_metadata.metadata["array"]
            ),
        }
        assert result == expected

    def test_invalid_metadata_raises(self):
        payload = {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "foo": [1, 2, 3]  # invalid
            },
        }

        with pytest.raises(TypeError, match="key='foo', type='list'"):
            ConsolidatedMetadata.from_dict(payload)

    def test_to_dict_empty(self):
        meta = ConsolidatedMetadata(
            metadata={
                "empty": GroupMetadata(
                    attributes={"key": "empty"},
                    consolidated_metadata=ConsolidatedMetadata(metadata={}),
                )
            }
        )
        result = meta.to_dict()
        expected = {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "empty": {
                    "attributes": {"key": "empty"},
                    "consolidated_metadata": {
                        "kind": "inline",
                        "must_understand": False,
                        "metadata": {},
                    },
                    "node_type": "group",
                    "zarr_format": 3,
                }
            },
        }
        assert result == expected

    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_open_consolidated_raises_async(self, zarr_format: ZarrFormat):
        store = zarr.storage.MemoryStore()
        await AsyncGroup.from_store(store, zarr_format=zarr_format)
        with pytest.raises(ValueError):
            await zarr.api.asynchronous.open_consolidated(store, zarr_format=zarr_format)

        with pytest.raises(ValueError):
            await zarr.api.asynchronous.open_consolidated(store, zarr_format=None)

    @pytest.fixture
    async def v2_consolidated_metadata_empty_dataset(
        self, memory_store: zarr.storage.MemoryStore
    ) -> AsyncGroup:
        zgroup_bytes = cpu.Buffer.from_bytes(json.dumps({"zarr_format": 2}).encode())
        zmetadata_bytes = cpu.Buffer.from_bytes(
            b'{"metadata":{".zgroup":{"zarr_format":2}},"zarr_consolidated_format":1}'
        )
        return AsyncGroup._from_bytes_v2(
            None, zgroup_bytes, zattrs_bytes=None, consolidated_metadata_bytes=zmetadata_bytes
        )

    async def test_consolidated_metadata_backwards_compatibility(
        self, v2_consolidated_metadata_empty_dataset
    ):
        """
        Test that consolidated metadata handles a missing .zattrs key. This is necessary for backwards compatibility  with zarr-python 2.x. See https://github.com/zarr-developers/zarr-python/issues/2694
        """
        store = zarr.storage.MemoryStore()
        await zarr.api.asynchronous.open(store=store, zarr_format=2)
        await zarr.api.asynchronous.consolidate_metadata(store)
        result = await zarr.api.asynchronous.open_consolidated(store, zarr_format=2)
        assert result.metadata == v2_consolidated_metadata_empty_dataset.metadata

    async def test_consolidated_metadata_v2(self):
        store = zarr.storage.MemoryStore()
        g = await AsyncGroup.from_store(store, attributes={"key": "root"}, zarr_format=2)
        dtype = parse_dtype("uint8", zarr_format=2)
        await g.create_array(name="a", shape=(1,), attributes={"key": "a"}, dtype=dtype)
        g1 = await g.create_group(name="g1", attributes={"key": "g1"})
        await g1.create_group(name="g2", attributes={"key": "g2"})

        await zarr.api.asynchronous.consolidate_metadata(store)
        result = await zarr.api.asynchronous.open_consolidated(store, zarr_format=2)

        expected = GroupMetadata(
            attributes={"key": "root"},
            zarr_format=2,
            consolidated_metadata=ConsolidatedMetadata(
                metadata={
                    "a": ArrayV2Metadata(
                        shape=(1,),
                        dtype=dtype,
                        attributes={"key": "a"},
                        chunks=(1,),
                        fill_value=0,
                        compressor=Blosc(),
                        order="C",
                    ),
                    "g1": GroupMetadata(
                        attributes={"key": "g1"},
                        zarr_format=2,
                        consolidated_metadata=ConsolidatedMetadata(
                            metadata={
                                "g2": GroupMetadata(
                                    attributes={"key": "g2"},
                                    zarr_format=2,
                                    consolidated_metadata=ConsolidatedMetadata(metadata={}),
                                )
                            }
                        ),
                    ),
                }
            ),
        )
        assert result.metadata == expected

    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_use_consolidated_false(
        self, memory_store: zarr.storage.MemoryStore, zarr_format: ZarrFormat
    ) -> None:
        with zarr.config.set(default_zarr_format=zarr_format):
            g = await group(store=memory_store, attributes={"foo": "bar"})
            await g.create_group(name="a")

            # test a stale read
            await zarr.api.asynchronous.consolidate_metadata(memory_store)
            await g.create_group(name="b")

            stale = await zarr.api.asynchronous.open_group(store=memory_store)
            assert len([x async for x in stale.members()]) == 1
            assert stale.metadata.consolidated_metadata
            assert list(stale.metadata.consolidated_metadata.metadata) == ["a"]

            # bypass stale data
            good = await zarr.api.asynchronous.open_group(
                store=memory_store, use_consolidated=False
            )
            assert len([x async for x in good.members()]) == 2

            # reconsolidate
            await zarr.api.asynchronous.consolidate_metadata(memory_store)

            good = await zarr.api.asynchronous.open_group(store=memory_store)
            assert len([x async for x in good.members()]) == 2
            assert good.metadata.consolidated_metadata
            assert sorted(good.metadata.consolidated_metadata.metadata) == ["a", "b"]

    async def test_stale_child_metadata_ignored(self, memory_store: zarr.storage.MemoryStore):
        # https://github.com/zarr-developers/zarr-python/issues/2921
        # When consolidating metadata, we should ignore any (possibly stale) metadata
        # from previous consolidations, *including at child nodes*.
        root = await zarr.api.asynchronous.group(store=memory_store, zarr_format=3)
        await root.create_group("foo")
        await zarr.api.asynchronous.consolidate_metadata(memory_store, path="foo")
        await root.create_group("foo/bar/spam")

        await zarr.api.asynchronous.consolidate_metadata(memory_store)

        reopened = await zarr.api.asynchronous.open_consolidated(store=memory_store, zarr_format=3)
        result = [x[0] async for x in reopened.members(max_depth=None)]
        expected = ["foo", "foo/bar", "foo/bar/spam"]
        assert result == expected

    async def test_use_consolidated_for_children_members(
        self, memory_store: zarr.storage.MemoryStore
    ):
        # A test that has *unconsolidated* metadata at the root group, but discovers
        # a child group with consolidated metadata.

        root = await zarr.api.asynchronous.create_group(store=memory_store)
        await root.create_group("a/b")
        # Consolidate metadata at "a/b"
        await zarr.api.asynchronous.consolidate_metadata(memory_store, path="a/b")

        # Add a new group a/b/c, that's not present in the CM at "a/b"
        await root.create_group("a/b/c")

        # Now according to the consolidated metadata, "a" has children ["b"]
        # but according to the unconsolidated metadata, "a" has children ["b", "c"]
        group = await zarr.api.asynchronous.open_group(store=memory_store, path="a")
        with pytest.warns(UserWarning, match="Object at 'c' not found"):
            result = sorted([x[0] async for x in group.members(max_depth=None)])
        expected = ["b"]
        assert result == expected

        result = sorted(
            [x[0] async for x in group.members(max_depth=None, use_consolidated_for_children=False)]
        )
        expected = ["b", "b/c"]
        assert result == expected


@pytest.mark.parametrize("fill_value", [np.nan, np.inf, -np.inf])
async def test_consolidated_metadata_encodes_special_chars(
    memory_store: Store, zarr_format: ZarrFormat, fill_value: float
):
    root = await group(store=memory_store, zarr_format=zarr_format)
    _time = await root.create_array("time", shape=(12,), dtype=np.float64, fill_value=fill_value)
    await zarr.api.asynchronous.consolidate_metadata(memory_store)

    root = await group(store=memory_store, zarr_format=zarr_format)
    root_buffer = root.metadata.to_buffer_dict(default_buffer_prototype())

    if zarr_format == 2:
        root_metadata = json.loads(root_buffer[".zmetadata"].to_bytes().decode("utf-8"))["metadata"]
    elif zarr_format == 3:
        root_metadata = json.loads(root_buffer["zarr.json"].to_bytes().decode("utf-8"))[
            "consolidated_metadata"
        ]["metadata"]

    expected_fill_value = _time._zdtype.to_json_scalar(fill_value, zarr_format=2)

    if zarr_format == 2:
        assert root_metadata["time/.zarray"]["fill_value"] == expected_fill_value
    elif zarr_format == 3:
        assert root_metadata["time"]["fill_value"] == expected_fill_value


class NonConsolidatedStore(zarr.storage.MemoryStore):
    """A store that doesn't support consolidated metadata"""

    @property
    def supports_consolidated_metadata(self) -> bool:
        return False


async def test_consolidate_metadata_raises_for_self_consolidating_stores():
    """Verify calling consolidate_metadata on a non supporting stores raises an error."""

    memory_store = NonConsolidatedStore()
    root = await zarr.api.asynchronous.create_group(store=memory_store)
    await root.create_group("a/b")

    with pytest.raises(TypeError, match="doesn't support consolidated metadata"):
        await zarr.api.asynchronous.consolidate_metadata(memory_store)


async def test_open_group_in_non_consolidating_stores():
    memory_store = NonConsolidatedStore()
    root = await zarr.api.asynchronous.create_group(store=memory_store)
    await root.create_group("a/b")

    # Opening a group without consolidatedion works as expected
    await AsyncGroup.open(memory_store, use_consolidated=False)

    # let the Store opt out of consolidation
    await AsyncGroup.open(memory_store, use_consolidated=None)

    # Opening a group with use_consolidated=True should fail
    with pytest.raises(ValueError, match="doesn't support consolidated metadata"):
        await AsyncGroup.open(memory_store, use_consolidated=True)
