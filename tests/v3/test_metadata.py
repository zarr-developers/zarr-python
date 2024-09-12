import numpy as np
import pytest

import zarr.api.synchronous
import zarr.store
from zarr.abc.store import Store
from zarr.api.asynchronous import (
    AsyncGroup,
    consolidate_metadata,
    group,
    open,
    open_consolidated,
)
from zarr.core.array import ArrayV3Metadata
from zarr.core.group import ConsolidatedMetadata, GroupMetadata


async def test_consolidated(memory_store_with_hierarchy: Store) -> None:
    # TODO: Figure out desired keys in
    # TODO: variety in the hierarchies
    # More nesting
    # arrays under arrays
    # single array
    # etc.
    g = await group(store=memory_store_with_hierarchy, attributes={"foo": "bar"})
    await g.create_array(name="air", shape=(1, 2, 3))
    await g.create_array(name="lat", shape=(1,))
    await g.create_array(name="lon", shape=(2,))
    await g.create_array(name="time", shape=(3,))

    child = await g.create_group("child", attributes={"key": "child"})
    await child.create_array("array", shape=(4, 4), attributes={"key": "child"})

    grandchild = await child.create_group("grandchild", attributes={"key": "grandchild"})
    await grandchild.create_array("array", shape=(4, 4), attributes={"key": "grandchild"})

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
                "child": GroupMetadata(attributes={"key": "child"}),
                "child/array": ArrayV3Metadata.from_dict(
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
                "child/grandchild": GroupMetadata(attributes={"key": "grandchild"}),
                "child/grandchild/array": ArrayV3Metadata.from_dict(
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
                ),
            },
        ),
    )
    assert group2.metadata == expected
    group3 = await open(store=memory_store_with_hierarchy)
    assert group3.metadata == expected

    group4 = await open_consolidated(store=memory_store_with_hierarchy)
    assert group4.metadata == expected


def test_consolidated_sync(memory_store_with_hierarchy):
    g = zarr.api.synchronous.group(store=memory_store_with_hierarchy, attributes={"foo": "bar"})
    g.create_array(name="air", shape=(1, 2, 3))
    g.create_array(name="lat", shape=(1,))
    g.create_array(name="lon", shape=(2,))
    g.create_array(name="time", shape=(3,))

    zarr.api.synchronous.consolidate_metadata(memory_store_with_hierarchy)
    group2 = zarr.api.synchronous.Group.open(memory_store_with_hierarchy)

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
    group3 = zarr.api.synchronous.open(store=memory_store_with_hierarchy)
    assert group3.metadata == expected

    group4 = zarr.api.synchronous.open_consolidated(store=memory_store_with_hierarchy)
    assert group4.metadata == expected


async def test_not_writable_raises(memory_store_with_hierarchy: zarr.store.MemoryStore) -> None:
    await group(store=memory_store_with_hierarchy, attributes={"foo": "bar"})
    read_store = zarr.store.MemoryStore(store_dict=memory_store_with_hierarchy._store_dict)
    with pytest.raises(ValueError, match="writable"):
        await consolidate_metadata(read_store)
