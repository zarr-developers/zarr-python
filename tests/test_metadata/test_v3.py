from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from zarr.codecs.bytes import BytesCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.core.config import config
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.core.dtype.npy.time import DateTime64
from zarr.core.group import GroupMetadata, parse_node_type
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    parse_dimension_names,
    parse_zarr_format,
)
from zarr.errors import MetadataValidationError, NodeTypeValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from zarr.abc.codec import Codec
    from zarr.core.common import JSON


from zarr.core.metadata.v3 import (
    parse_node_type_array,
)

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
flexible_dtypes = ("str", "bytes", "void")
if _NUMPY_SUPPORTS_VLEN_STRING:
    vlen_string_dtypes = ("T",)
else:
    vlen_string_dtypes = ("O",)

dtypes = (
    *bool_dtypes,
    *int_dtypes,
    *float_dtypes,
    *complex_dtypes,
    *flexible_dtypes,
    *vlen_string_dtypes,
)


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(
        MetadataValidationError,
        match=f"Invalid value for 'zarr_format'. Expected '3'. Got '{data}'.",
    ):
        parse_zarr_format(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


def test_parse_node_type_valid() -> None:
    assert parse_node_type("array") == "array"
    assert parse_node_type("group") == "group"


@pytest.mark.parametrize("node_type", [None, 2, "other"])
def test_parse_node_type_invalid(node_type: Any) -> None:
    with pytest.raises(
        MetadataValidationError,
        match=f"Invalid value for 'node_type'. Expected 'array or group'. Got '{node_type}'.",
    ):
        parse_node_type(node_type)


@pytest.mark.parametrize("data", [None, "group"])
def test_parse_node_type_array_invalid(data: Any) -> None:
    with pytest.raises(
        NodeTypeValidationError,
        match=f"Invalid value for 'node_type'. Expected 'array'. Got '{data}'.",
    ):
        parse_node_type_array(data)


def test_parse_node_typev_array_alid() -> None:
    assert parse_node_type_array("array") == "array"


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("fill_value", [[1.0, 0.0], [0, 1]])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_jsonify_fill_value_complex(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly handles complex values represented
    as length-2 sequences
    """
    zarr_format = 3
    dtype = get_data_type_from_native_dtype(dtype_str)
    expected = dtype.to_native_dtype().type(complex(*fill_value))
    observed = dtype.from_json_scalar(fill_value, zarr_format=zarr_format)
    assert observed == expected
    assert dtype.to_json_scalar(observed, zarr_format=zarr_format) == tuple(fill_value)


@pytest.mark.parametrize("fill_value", [{"foo": 10}])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_type(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid non-sequential types.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype_instance = get_data_type_from_native_dtype(dtype_str)
    with pytest.raises(TypeError, match=f"Invalid type: {fill_value}"):
        dtype_instance.from_json_scalar(fill_value, zarr_format=3)


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
    dtype_instance = get_data_type_from_native_dtype(dtype_str)
    with pytest.raises(TypeError, match=re.escape(f"Invalid type: {fill_value}")):
        dtype_instance.from_json_scalar(fill_value, zarr_format=3)


@pytest.mark.parametrize("chunk_grid", ["regular"])
@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("codecs", [[BytesCodec(endian=None)]])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("chunk_key_encoding", ["v2", "default"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
@pytest.mark.parametrize("dimension_names", ["nones", "strings", "missing"])
@pytest.mark.parametrize("storage_transformers", [None, ()])
def test_metadata_to_dict(
    chunk_grid: str,
    codecs: list[Codec],
    fill_value: Any,
    chunk_key_encoding: Literal["v2", "default"],
    dimension_separator: Literal[".", "/"] | None,
    dimension_names: Literal["nones", "strings", "missing"],
    attributes: dict[str, Any] | None,
    storage_transformers: tuple[dict[str, JSON]] | None,
) -> None:
    shape = (1, 2, 3)
    data_type_str = "uint8"
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
        "data_type": data_type_str,
        "chunk_key_encoding": cke,
        "codecs": tuple(c.to_dict() for c in codecs),
        "fill_value": fill_value,
        "storage_transformers": storage_transformers,
    }

    if attributes is not None:
        metadata_dict["attributes"] = attributes
    if dnames is not None:
        metadata_dict["dimension_names"] = dnames

    metadata = ArrayV3Metadata.from_dict(metadata_dict)
    observed = metadata.to_dict()
    expected = metadata_dict.copy()

    # if unset or None or (), storage_transformers gets normalized to ()
    assert observed["storage_transformers"] == ()
    observed.pop("storage_transformers")
    expected.pop("storage_transformers")

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


@pytest.mark.parametrize("indent", [2, 4, None])
def test_json_indent(indent: int):
    with config.set({"json_indent": indent}):
        m = GroupMetadata()
        d = m.to_buffer_dict(default_buffer_prototype())["zarr.json"].to_bytes()
        assert d == json.dumps(json.loads(d), indent=indent).encode()


@pytest.mark.parametrize("fill_value", [-1, 0, 1, 2932897])
@pytest.mark.parametrize("precision", ["ns", "D"])
async def test_datetime_metadata(fill_value: int, precision: str) -> None:
    dtype = DateTime64(unit=precision)
    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": dtype.to_json(zarr_format=3),
        "chunk_key_encoding": {"name": "default", "separator": "."},
        "codecs": (BytesCodec(),),
        "fill_value": dtype.to_json_scalar(
            dtype.to_native_dtype().type(fill_value, dtype.unit), zarr_format=3
        ),
    }
    metadata = ArrayV3Metadata.from_dict(metadata_dict)
    # ensure there isn't a TypeError here.
    d = metadata.to_buffer_dict(default_buffer_prototype())

    result = json.loads(d["zarr.json"].to_bytes())
    assert result["fill_value"] == fill_value


@pytest.mark.parametrize(
    ("data_type", "fill_value"), [("uint8", {}), ("int32", [0, 1]), ("float32", "foo")]
)
async def test_invalid_fill_value_raises(data_type: str, fill_value: float) -> None:
    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": data_type,
        "chunk_key_encoding": {"name": "default", "separator": "."},
        "codecs": ({"name": "bytes"},),
        "fill_value": fill_value,  # this is not a valid fill value for uint8
    }
    # multiple things can go wrong here, so we don't match on the error message.
    with pytest.raises(TypeError):
        ArrayV3Metadata.from_dict(metadata_dict)


@pytest.mark.parametrize("fill_value", [("NaN"), "Infinity", "-Infinity"])
async def test_special_float_fill_values(fill_value: str) -> None:
    metadata_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": "float64",
        "chunk_key_encoding": {"name": "default", "separator": "."},
        "codecs": [{"name": "bytes"}],
        "fill_value": fill_value,  # this is not a valid fill value for uint8
    }
    m = ArrayV3Metadata.from_dict(metadata_dict)
    d = json.loads(m.to_buffer_dict(default_buffer_prototype())["zarr.json"].to_bytes())
    assert m.fill_value is not None
    if fill_value == "NaN":
        assert np.isnan(m.fill_value)
        assert d["fill_value"] == "NaN"
    elif fill_value == "Infinity":
        assert np.isposinf(m.fill_value)
        assert d["fill_value"] == "Infinity"
    elif fill_value == "-Infinity":
        assert np.isneginf(m.fill_value)
        assert d["fill_value"] == "-Infinity"
