"""Tests for zarr v3 metadata classes and parsing helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.conftest import Expect, ExpectFail
from tests.test_metadata.conftest import minimal_metadata_dict_v3
from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import config
from zarr.core.dtype import UInt8
from zarr.core.group import GroupMetadata, parse_node_type
from zarr.core.metadata.v3 import (
    ARRAY_METADATA_KEYS,
    ArrayMetadataJSON_V3,
    ArrayV3Metadata,
    parse_codecs,
    parse_dimension_names,
    parse_node_type_array,
    parse_zarr_format,
)
from zarr.errors import (
    MetadataValidationError,
    NodeTypeValidationError,
    UnknownCodecError,
)

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_parse_zarr_format_valid() -> None:
    """The integer 3 is the only valid zarr_format for v3."""
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    """Non-3 values are rejected."""
    with pytest.raises(MetadataValidationError):
        parse_zarr_format(data)


def test_parse_node_type_valid() -> None:
    """'array' and 'group' are the only valid node types."""
    assert parse_node_type("array") == "array"
    assert parse_node_type("group") == "group"


@pytest.mark.parametrize("data", [None, 2, "other"])
def test_parse_node_type_invalid(data: Any) -> None:
    """Non-string and unrecognized values are rejected."""
    with pytest.raises(MetadataValidationError):
        parse_node_type(data)


def test_parse_node_type_array_valid() -> None:
    """parse_node_type_array accepts only 'array'."""
    assert parse_node_type_array("array") == "array"


@pytest.mark.parametrize("data", [None, "group"])
def test_parse_node_type_array_invalid(data: Any) -> None:
    """parse_node_type_array rejects 'group' and non-string values."""
    with pytest.raises(NodeTypeValidationError):
        parse_node_type_array(data)


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"], ()])
def test_parse_dimension_names_valid(data: Any) -> None:
    """None, tuples of strings, lists of strings, and empty tuples are accepted."""
    result = parse_dimension_names(data)
    if data is None:
        assert result is None
    else:
        assert result == tuple(data)


@pytest.mark.parametrize("data", [[1, 2, "a"], [None, 3]])
def test_parse_dimension_names_invalid(data: Any) -> None:
    """Iterables containing non-string elements are rejected."""
    with pytest.raises(TypeError, match="Expected either None or"):
        parse_dimension_names(data)


def test_parse_codecs_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unregistered codec name raises UnknownCodecError."""
    from collections import defaultdict

    import zarr.registry
    from zarr.registry import Registry

    monkeypatch.setattr(zarr.registry, "_codec_registries", defaultdict(Registry))
    with pytest.raises(UnknownCodecError):
        parse_codecs([{"name": "unknown"}])


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_array_metadata_keys_matches_typeddict() -> None:
    """
    Test that the variable modelling the set of keys for array v3 metadata matches
    the keys of the typeddict model for the metadata.
    """
    assert ARRAY_METADATA_KEYS == set(ArrayMetadataJSON_V3.__annotations__.keys())


# ---------------------------------------------------------------------------
# ArrayV3Metadata: round-trip
# ---------------------------------------------------------------------------

# Codecs after evolution for single-byte (uint8) and multi-byte (float64) types.
_UINT8_CODECS = ({"name": "bytes"},)
_FLOAT64_CODECS = ({"name": "bytes", "configuration": {"endian": "little"}},)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={},
            output=minimal_metadata_dict_v3(codecs=_UINT8_CODECS),
            id="minimal",
        ),
        Expect(
            input={"attributes": {"key": "value"}},
            output=minimal_metadata_dict_v3(attributes={"key": "value"}, codecs=_UINT8_CODECS),
            id="with_attributes",
        ),
        Expect(
            input={"dimension_names": ("x", "y")},
            output=minimal_metadata_dict_v3(dimension_names=("x", "y"), codecs=_UINT8_CODECS),
            id="with_dimension_names",
        ),
        Expect(
            input={"storage_transformers": ()},
            output=minimal_metadata_dict_v3(storage_transformers=(), codecs=_UINT8_CODECS),
            id="with_storage_transformers",
        ),
        Expect(
            input={"data_type": "float64", "fill_value": 0.0},
            output=minimal_metadata_dict_v3(
                data_type="float64", fill_value=0.0, codecs=_FLOAT64_CODECS
            ),
            id="float64",
        ),
        Expect(
            input={"chunk_key_encoding": {"name": "v2", "configuration": {"separator": "."}}},
            output=minimal_metadata_dict_v3(
                chunk_key_encoding={"name": "v2", "configuration": {"separator": "."}},
                codecs=_UINT8_CODECS,
            ),
            id="v2_chunk_key_encoding",
        ),
        Expect(
            input={"data_type": "float64", "fill_value": "NaN"},
            output=minimal_metadata_dict_v3(
                data_type="float64", fill_value="NaN", codecs=_FLOAT64_CODECS
            ),
            id="nan_fill_value",
        ),
        Expect(
            input={"data_type": "float64", "fill_value": "Infinity"},
            output=minimal_metadata_dict_v3(
                data_type="float64", fill_value="Infinity", codecs=_FLOAT64_CODECS
            ),
            id="inf_fill_value",
        ),
        Expect(
            input={"data_type": "float64", "fill_value": "-Infinity"},
            output=minimal_metadata_dict_v3(
                data_type="float64", fill_value="-Infinity", codecs=_FLOAT64_CODECS
            ),
            id="neg_inf_fill_value",
        ),
        Expect(
            input={
                "attributes": {},
                "storage_transformers": (),
                "extra_fields": {"my_ext": {"must_understand": False, "data": [1, 2, 3]}},
            },
            output=minimal_metadata_dict_v3(
                attributes={},
                storage_transformers=(),
                codecs=_UINT8_CODECS,
                extra_fields={"my_ext": {"must_understand": False, "data": [1, 2, 3]}},
            ),
            id="extra_fields",
        ),
    ],
    ids=lambda case: case.id,
)
def test_array_metadata_roundtrip(case: Expect[dict[str, Any], dict[str, Any]]) -> None:
    """from_dict(d).to_dict() produces the expected output, including codec evolution."""
    d = minimal_metadata_dict_v3(**case.input)
    m = ArrayV3Metadata.from_dict(d)  # type: ignore[arg-type]
    assert m.to_dict() == case.output


# ---------------------------------------------------------------------------
# ArrayV3Metadata: failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={"dimension_names": ("x", "y", "z")},
            exception=ValueError,
            msg="dimension_names.*shape",
            id="dimension_names_length_mismatch",
        ),
        ExpectFail(
            input={"data_type": "uint8", "fill_value": {}},
            exception=TypeError,
            msg=".*",
            id="invalid_fill_value_type",
        ),
    ],
    ids=lambda case: case.id,
)
def test_array_metadata_from_dict_fails(case: ExpectFail[dict[str, Any]]) -> None:
    """from_dict rejects invalid metadata documents."""
    d = minimal_metadata_dict_v3(**case.input)
    with pytest.raises(case.exception, match=case.msg):
        ArrayV3Metadata.from_dict(d)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input=minimal_metadata_dict_v3(extra_fields={"my_ext": {"must_understand": True}}),
            exception=MetadataValidationError,
            msg="disallowed extra fields",
            id="must_understand_true",
        ),
        ExpectFail(
            input=minimal_metadata_dict_v3(extra_fields={"my_ext": 42}),
            exception=MetadataValidationError,
            msg="disallowed extra fields",
            id="non_dict_extra_field",
        ),
    ],
    ids=lambda case: case.id,
)
def test_array_metadata_extra_fields_rejected(case: ExpectFail[dict[str, Any]]) -> None:
    """from_dict rejects extra fields that don't conform to the spec."""
    with pytest.raises(case.exception, match=case.msg):
        ArrayV3Metadata.from_dict(case.input)


def test_init_extra_fields_collision() -> None:
    """Extra field keys that collide with reserved metadata field names are rejected."""
    extra_fields: dict[str, object] = {"shape": (10,), "data_type": "uint8"}
    with pytest.raises(ValueError, match="collide with keys reserved"):
        ArrayV3Metadata(
            shape=(10,),
            data_type=UInt8(),
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10,)}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
            fill_value=0,
            codecs=({"name": "bytes", "configuration": {"endian": "little"}},),
            attributes={},
            dimension_names=None,
            extra_fields=extra_fields,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# JSON indent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indent", [2, 4, None])
def test_json_indent(indent: int | None) -> None:
    """The json_indent config setting controls indentation in to_buffer_dict output."""
    with config.set({"json_indent": indent}):
        m = GroupMetadata()
        d = m.to_buffer_dict(default_buffer_prototype())["zarr.json"].to_bytes()
        assert d == json.dumps(json.loads(d), indent=indent).encode()


# ---------------------------------------------------------------------------
# GroupMetadata.to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
def test_group_metadata_to_dict(attributes: dict[str, Any] | None) -> None:
    """GroupMetadata.to_dict produces the expected v3 JSON structure."""
    meta = GroupMetadata(attributes=attributes)
    assert meta.to_dict() == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attributes or {},
    }


@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
def test_group_metadata_to_dict_consolidated(attributes: dict[str, Any] | None) -> None:
    """GroupMetadata.to_dict includes consolidated_metadata when present."""
    from zarr import consolidate_metadata, create_group
    from zarr.errors import ZarrUserWarning

    store: dict[str, object] = {}
    group = create_group(store, attributes=attributes, zarr_format=3)
    group.create_group("foo")
    with pytest.warns(
        ZarrUserWarning,
        match="Consolidated metadata is currently not part in the Zarr format 3 specification.",
    ):
        group = consolidate_metadata(store)

    assert group.metadata.to_dict() == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attributes or {},
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "foo": {
                    "attributes": {},
                    "zarr_format": 3,
                    "node_type": "group",
                    "consolidated_metadata": {
                        "kind": "inline",
                        "metadata": {},
                        "must_understand": False,
                    },
                }
            },
        },
    }
