from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

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
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.storage.common import StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import ZarrFormat


@pytest.fixture
async def memory_store_with_hierarchy(memory_store: Store) -> None:
    g = await group(store=memory_store, attributes={"foo": "bar"})
    await g.create_array(name="air", shape=(1, 2, 3))
    await g.create_array(name="lat", shape=(1,))
    await g.create_array(name="lon", shape=(2,))
    await g.create_array(name="time", shape=(3,))

    child = await g.create_group("child", attributes={"key": "child"})
    await child.create_array("array", shape=(4, 4), attributes={"key": "child"})

    grandchild = await child.create_group("grandchild", attributes={"key": "grandchild"})
    await grandchild.create_array("array", shape=(4, 4), attributes={"key": "grandchild"})
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
            "codecs": ({"configuration": {"endian": "little"}, "name": "bytes"},),
            "data_type": "float64",
            "fill_value": np.float64(0.0),
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
                            **{
                                "shape": (1, 2, 3),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (1, 2, 3)},
                                    "name": "regular",
                                },
                            },
                            **array_metadata,
                        }
                    ),
                    "lat": ArrayV3Metadata.from_dict(
                        {
                            **{
                                "shape": (1,),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (1,)},
                                    "name": "regular",
                                },
                            },
                            **array_metadata,
                        }
                    ),
                    "lon": ArrayV3Metadata.from_dict(
                        {
                            **{"shape": (2,)},
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (2,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "time": ArrayV3Metadata.from_dict(
                        {
                            **{
                                "shape": (3,),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (3,)},
                                    "name": "regular",
                                },
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
                                        **{
                                            "attributes": {"key": "child"},
                                            "shape": (4, 4),
                                            "chunk_grid": {
                                                "configuration": {"chunk_shape": (4, 4)},
                                                "name": "regular",
                                            },
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
                                                    **{
                                                        "attributes": {"key": "grandchild"},
                                                        "shape": (4, 4),
                                                        "chunk_grid": {
                                                            "configuration": {
                                                                "chunk_shape": (4, 4)
                                                            },
                                                            "name": "regular",
                                                        },
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
        g.create_array(name="air", shape=(1, 2, 3))
        g.create_array(name="lat", shape=(1,))
        g.create_array(name="lon", shape=(2,))
        g.create_array(name="time", shape=(3,))

        zarr.api.synchronous.consolidate_metadata(memory_store)
        group2 = zarr.api.synchronous.Group.open(memory_store)

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

        expected = GroupMetadata(
            attributes={"foo": "bar"},
            consolidated_metadata=ConsolidatedMetadata(
                kind="inline",
                must_understand=False,
                metadata={
                    "air": ArrayV3Metadata.from_dict(
                        {
                            **{
                                "shape": (1, 2, 3),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (1, 2, 3)},
                                    "name": "regular",
                                },
                            },
                            **array_metadata,
                        }
                    ),
                    "lat": ArrayV3Metadata.from_dict(
                        {
                            **{
                                "shape": (1,),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (1,)},
                                    "name": "regular",
                                },
                            },
                            **array_metadata,
                        }
                    ),
                    "lon": ArrayV3Metadata.from_dict(
                        {
                            **{"shape": (2,)},
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (2,)},
                                "name": "regular",
                            },
                            **array_metadata,
                        }
                    ),
                    "time": ArrayV3Metadata.from_dict(
                        {
                            **{
                                "shape": (3,),
                                "chunk_grid": {
                                    "configuration": {"chunk_shape": (3,)},
                                    "name": "regular",
                                },
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
        read_store = zarr.storage.MemoryStore(store_dict=memory_store._store_dict)
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
                        **{
                            "shape": (1, 2, 3),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1, 2, 3)},
                                "name": "regular",
                            },
                        },
                        **array_metadata,
                    }
                ),
                "lat": ArrayV3Metadata.from_dict(
                    {
                        **{
                            "shape": (1,),
                            "chunk_grid": {
                                "configuration": {"chunk_shape": (1,)},
                                "name": "regular",
                            },
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
                                    **{
                                        "attributes": {"key": "child"},
                                        "shape": (4, 4),
                                        "chunk_grid": {
                                            "configuration": {"chunk_shape": (4, 4)},
                                            "name": "regular",
                                        },
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
                                                **{
                                                    "attributes": {"key": "grandchild"},
                                                    "shape": (4, 4),
                                                    "chunk_grid": {
                                                        "configuration": {"chunk_shape": (4, 4)},
                                                        "name": "regular",
                                                    },
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
        store = zarr.storage.MemoryStore(mode="w")
        await AsyncGroup.from_store(store, zarr_format=zarr_format)
        with pytest.raises(ValueError):
            await zarr.api.asynchronous.open_consolidated(store, zarr_format=zarr_format)

        with pytest.raises(ValueError):
            await zarr.api.asynchronous.open_consolidated(store, zarr_format=None)

    async def test_consolidated_metadata_v2(self):
        store = zarr.storage.MemoryStore(mode="w")
        g = await AsyncGroup.from_store(store, attributes={"key": "root"}, zarr_format=2)
        await g.create_array(name="a", shape=(1,), attributes={"key": "a"})
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
                        dtype="float64",
                        attributes={"key": "a"},
                        chunks=(1,),
                        fill_value=None,
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
        with zarr.config.set(default_zarr_version=zarr_format):
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
