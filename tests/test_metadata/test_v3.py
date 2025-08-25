from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import config
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.core.dtype.npy.time import DateTime64
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    parse_codecs,
)
from zarr.errors import UnknownCodecError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from zarr.abc.codec import Codec
    from zarr.core.common import JSON, ArrayMetadataJSON_V3, NamedConfig


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


@pytest.mark.parametrize("fill_value", [[1.0, 0.0], [0, 1]])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_jsonify_fill_value_complex(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly handles complex values represented
    as length-2 sequences
    """
    zarr_format: Literal[3] = 3
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


@pytest.mark.parametrize(
    "chunk_grid", [{"name": "regular", "configuration": {"chunk_shape": (1, 1, 1)}}]
)
@pytest.mark.parametrize("codecs", [({"name": "bytes"},)])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("data_type", ["int8", "uint8"])
@pytest.mark.parametrize(
    "chunk_key_encoding",
    [
        {"name": "v2", "configuration": {"separator": "."}},
        {"name": "v2", "configuration": {"separator": "/"}},
        {"name": "v2"},
        {"name": "default", "configuration": {"separator": "."}},
        {"name": "default", "configuration": {"separator": "/"}},
        {"name": "default"},
    ],
)
@pytest.mark.parametrize("attributes", ["unset", {"foo": "bar"}])
@pytest.mark.parametrize("dimension_names", [(None, None, None), ("a", "b", None), "unset"])
@pytest.mark.parametrize("storage_transformers", [(), "unset"])
def test_metadata_to_dict(
    chunk_grid: NamedConfig[str, Mapping[str, object]],
    codecs: list[Codec],
    data_type: str,
    fill_value: Any,
    chunk_key_encoding: NamedConfig[str, Mapping[str, object]],
    dimension_names: tuple[str | None, ...] | Literal["unset"],
    attributes: dict[str, Any] | Literal["unset"],
    storage_transformers: tuple[dict[str, JSON]] | Literal["unset"],
) -> None:
    shape = (1, 2, 3)

    # These are the fields in the array metadata document that are optional
    not_required = {}

    if dimension_names != "unset":
        not_required["dimension_names"] = dimension_names

    if storage_transformers != "unset":
        not_required["storage_transformers"] = storage_transformers

    if attributes != "unset":
        not_required["attributes"] = attributes

    source_dict = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "chunk_grid": chunk_grid,
        "data_type": data_type,
        "chunk_key_encoding": chunk_key_encoding,
        "codecs": codecs,
        "fill_value": fill_value,
    } | not_required

    metadata = ArrayV3Metadata.from_dict(source_dict)
    parsed_dict = metadata.to_dict()

    for k, v in parsed_dict.items():
        if k in source_dict:
            if k == "chunk_key_encoding":
                assert v["name"] == chunk_key_encoding["name"]
                if chunk_key_encoding["name"] == "v2":
                    if "configuration" in chunk_key_encoding:
                        if "separator" in chunk_key_encoding["configuration"]:
                            assert (
                                v["configuration"]["separator"]
                                == chunk_key_encoding["configuration"]["separator"]
                            )
                    else:
                        assert v["configuration"]["separator"] == "."
                elif chunk_key_encoding["name"] == "default":
                    if "configuration" in chunk_key_encoding:
                        if "separator" in chunk_key_encoding["configuration"]:
                            assert (
                                v["configuration"]["separator"]
                                == chunk_key_encoding["configuration"]["separator"]
                            )
                    else:
                        assert v["configuration"]["separator"] == "/"
            else:
                assert source_dict[k] == v
        else:
            if k == "attributes":
                assert v == {}
            elif k == "storage_transformers":
                assert v == ()
            else:
                assert v is None


@pytest.mark.parametrize("indent", [2, 4, None])
def test_json_indent(indent: int) -> None:
    with config.set({"json_indent": indent}):
        m = GroupMetadata()
        d = m.to_buffer_dict(default_buffer_prototype())["zarr.json"].to_bytes()
        assert d == json.dumps(json.loads(d), indent=indent).encode()


@pytest.mark.parametrize("fill_value", [-1, 0, 1, 2932897])
@pytest.mark.parametrize("precision", ["ns", "D"])
async def test_datetime_metadata(fill_value: int, precision: Literal["ns", "D"]) -> None:
    dtype = DateTime64(unit=precision)
    metadata_dict: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": dtype.to_json(zarr_format=3),
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
        "codecs": ({"name": "bytes"},),
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
    metadata_dict: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": data_type,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
        "codecs": ({"name": "bytes"},),
        "fill_value": fill_value,  # this is not a valid fill value for uint8
    }
    # multiple things can go wrong here, so we don't match on the error message.
    with pytest.raises(TypeError):
        ArrayV3Metadata.from_dict(metadata_dict)


@pytest.mark.parametrize("fill_value", [("NaN"), "Infinity", "-Infinity"])
async def test_special_float_fill_values(fill_value: str) -> None:
    metadata_dict: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": "float64",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
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


def test_parse_codecs_unknown_codec_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import defaultdict

    import zarr.registry
    from zarr.registry import Registry

    # to make sure the codec is always unknown (not sure if that's necessary)
    monkeypatch.setattr(zarr.registry, "__codec_registries", defaultdict(Registry))

    codecs = [{"name": "unknown"}]
    with pytest.raises(UnknownCodecError):
        parse_codecs(codecs)
