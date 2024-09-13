from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

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
from zarr.codecs.bytes import BytesCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.metadata.v3 import parse_dimension_names, parse_fill_value, parse_zarr_format
from zarr.store.common import StorePath

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from zarr.abc.codec import Codec
    from zarr.abc.store import Store

bool_dtypes = ("bool",)

int_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

float_dtypes = (
    "float16",
    "float32",
    "float64",
)

complex_dtypes = ("complex64", "complex128")

dtypes = (*bool_dtypes, *int_dtypes, *float_dtypes, *complex_dtypes)


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 3. Got {data}"):
        parse_zarr_format(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_auto_fill_value(dtype_str: str) -> None:
    """
    Test that parse_fill_value(None, dtype) results in the 0 value for the given dtype.
    """
    dtype = np.dtype(dtype_str)
    fill_value = None
    assert parse_fill_value(fill_value, dtype) == dtype.type(0)


@pytest.mark.parametrize("fill_value", [0, 1.11, False, True])
@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_fill_value_valid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) casts fill_value to the given dtype.
    """
    dtype = np.dtype(dtype_str)
    assert parse_fill_value(fill_value, dtype) == dtype.type(fill_value)


@pytest.mark.parametrize("fill_value", ["not a valid value"])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_value(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises ValueError for invalid values.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    with pytest.raises(ValueError):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0], [0, 1], complex(1, 1), np.complex64(0)])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly handles complex values represented
    as length-2 sequences
    """
    dtype = np.dtype(dtype_str)
    if isinstance(fill_value, list):
        expected = dtype.type(complex(*fill_value))
    else:
        expected = dtype.type(fill_value)
    assert expected == parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0, 3.0], [0, 1, 3], [1]])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex_invalid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly rejects sequences with length not
    equal to 2
    """
    dtype = np.dtype(dtype_str)
    match = (
        f"Got an invalid fill value for complex data type {dtype}."
        f"Expected a sequence with 2 elements, but {fill_value} has "
        f"length {len(fill_value)}."
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        parse_fill_value(fill_value=fill_value, dtype=dtype)


@pytest.mark.parametrize("fill_value", [{"foo": 10}])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_type(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid non-sequential types.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    match = "must be"
    with pytest.raises(TypeError, match=match):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize(
    "fill_value",
    [
        [
            1,
        ],
        (1, 23, 4),
    ],
)
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes])
def test_parse_fill_value_invalid_type_sequence(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid sequential types.
    This test excludes bool because the bool constructor takes anything, and complex because
    complex values can be created from length-2 sequences.
    """
    dtype = np.dtype(dtype_str)
    match = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}"
    with pytest.raises(TypeError, match=re.escape(match)):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("chunk_grid", ["regular"])
@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("codecs", [[BytesCodec()]])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("chunk_key_encoding", ["v2", "default"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
@pytest.mark.parametrize("dimension_names", ["nones", "strings", "missing"])
def test_metadata_to_dict(
    chunk_grid: str,
    codecs: list[Codec],
    fill_value: Any,
    chunk_key_encoding: Literal["v2", "default"],
    dimension_separator: Literal[".", "/"] | None,
    dimension_names: Literal["nones", "strings", "missing"],
    attributes: None | dict[str, Any],
) -> None:
    shape = (1, 2, 3)
    data_type = "uint8"
    if chunk_grid == "regular":
        cgrid = {"name": "regular", "configuration": {"chunk_shape": (1, 1, 1)}}

    cke: dict[str, Any]
    cke_name_dict = {"name": chunk_key_encoding}
    if dimension_separator is not None:
        cke = cke_name_dict | {"configuration": {"separator": dimension_separator}}
    else:
        cke = cke_name_dict
    dnames: tuple[str | None, ...] | None

    if dimension_names == "strings":
        dnames = tuple(map(str, range(len(shape))))
    elif dimension_names == "missing":
        dnames = None
    elif dimension_names == "nones":
        dnames = (None,) * len(shape)

    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "chunk_grid": cgrid,
        "data_type": data_type,
        "chunk_key_encoding": cke,
        "codecs": tuple(c.to_dict() for c in codecs),
        "fill_value": fill_value,
    }

    if attributes is not None:
        metadata_dict["attributes"] = attributes
    if dnames is not None:
        metadata_dict["dimension_names"] = dnames

    metadata = ArrayV3Metadata.from_dict(metadata_dict)
    observed = metadata.to_dict()
    expected = metadata_dict.copy()
    if attributes is None:
        assert observed["attributes"] == {}
        observed.pop("attributes")
    if dimension_separator is None:
        if chunk_key_encoding == "default":
            expected_cke_dict = DefaultChunkKeyEncoding(separator="/").to_dict()
        else:
            expected_cke_dict = V2ChunkKeyEncoding(separator=".").to_dict()
        assert observed["chunk_key_encoding"] == expected_cke_dict
        observed.pop("chunk_key_encoding")
        expected.pop("chunk_key_encoding")
    assert observed == expected


@pytest.mark.parametrize("fill_value", [-1, 0, 1, 2932897])
@pytest.mark.parametrize("precision", ["ns", "D"])
async def test_datetime_metadata(fill_value: int, precision: str) -> None:
    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": f"<M8[{precision}]",
        "chunk_key_encoding": {"name": "default", "separator": "."},
        "codecs": (),
        "fill_value": np.datetime64(fill_value, precision),
    }
    metadata = ArrayV3Metadata.from_dict(metadata_dict)
    # ensure there isn't a TypeError here.
    d = metadata.to_buffer_dict(default_buffer_prototype())

    result = json.loads(d["zarr.json"].to_bytes())
    assert result["fill_value"] == fill_value


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
