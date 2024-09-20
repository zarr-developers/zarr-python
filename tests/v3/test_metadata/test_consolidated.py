from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr.api.synchronous
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
from zarr.store.common import StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store


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
    return memory_store


class TestConsolidated:
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
            "data_type": np.dtype("float64"),
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
                                            )
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
            "data_type": np.dtype("float64"),
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

    async def test_not_writable_raises(self, memory_store: zarr.store.MemoryStore) -> None:
        await group(store=memory_store, attributes={"foo": "bar"})
        read_store = zarr.store.MemoryStore(store_dict=memory_store._store_dict)
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

        # invalid metadata
        data["metadata"]["a"] = {"node_type": "array", "zarr_format": 3}

        with pytest.raises(TypeError):
            ConsolidatedMetadata.from_dict(data)

    def test_flatten(self):
        array_metadata = {
            "attributes": {},
            "chunk_key_encoding": {
                "configuration": {"separator": "/"},
                "name": "default",
            },
            "codecs": ({"configuration": {"endian": "little"}, "name": "bytes"},),
            "data_type": np.dtype("float64"),
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
            "child": GroupMetadata(attributes={"key": "child"}),
            "child/array": metadata.metadata["child"].consolidated_metadata.metadata["array"],
            "child/grandchild": GroupMetadata(attributes={"key": "grandchild"}),
            "child/grandchild/array": metadata.metadata["child"]
            .consolidated_metadata.metadata["grandchild"]
            .consolidated_metadata.metadata["array"],
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
