"""The two kinds of field in a v2 array document: a dtype in NumPy's spelling, and a numcodecs configuration."""

from __future__ import annotations

import math
from typing import Any

import pytest

from zarr_metadata.definition import AcceptedField, EmptyConfiguration, RefusedField, resolve
from zarr_metadata.definition._definition import filed_under
from zarr_metadata.v2._definition import parse_typestr
from zarr_metadata.v2.definition import CORE_V2, ZarrV2CodecDefinition, ZarrV2DataTypeDefinition

Loc = tuple[str | int, ...]


@pytest.mark.parametrize(
    ("name", "code", "carried"),
    [
        ("<f4", "f", {"byteorder": "<", "itemsize": 4}),
        (">i8", "i", {"byteorder": ">", "itemsize": 8}),
        ("|b1", "b", {"byteorder": "|", "itemsize": 1}),
        ("|S12", "S", {"byteorder": "|", "itemsize": 12}),
        ("<U0", "U", {"byteorder": "<", "itemsize": 0}),
        ("|O", "O", {"byteorder": "|"}),
        ("<M8[ns]", "M", {"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 1}),
        ("<m8[10s]", "m", {"byteorder": "<", "itemsize": 8, "unit": "s", "scale_factor": 10}),
        ("<M8", "M", {"byteorder": "<", "itemsize": 8}),
    ],
)
def test_a_typestr_parses_into_its_code_and_what_it_carries(
    name: str, code: str, carried: dict[str, Any]
) -> None:
    """A NumPy typestr splits into its type code and the values the name carries: the byte order, the item size if one is written, and a time unit with its multiplier if one is given in brackets."""
    assert parse_typestr(name) == (code, carried)


@pytest.mark.parametrize(
    "name",
    [
        "float32",
        "f4",
        "<f",
        "<f4x",
        "",
        "<",
        "<f4[ns",
        "[('a','<f4')]",
        "<M8[ns][s]",
        "<f\u0664",
        "<M8[\u0663ns]",
    ],
)
def test_error_a_string_that_is_no_typestr_parses_to_none(name: str) -> None:
    """A string without a byte order, a type code and a size in that order is not a typestr."""
    assert parse_typestr(name) is None


@pytest.mark.parametrize(
    ("name", "filed", "carried"),
    [
        ("<f4", "float", {"byteorder": "<", "itemsize": 4}),
        ("|b1", "bool", {"byteorder": "|", "itemsize": 1}),
        (
            "<M8[ns]",
            "datetime64",
            {"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 1},
        ),
        ("struct", "struct", None),
        ("<e2", "<e2", None),
        ("float", None, None),
    ],
    ids=["float", "bool", "datetime", "struct", "unknown-code", "family-name"],
)
def test_a_v2_dtype_name_is_filed_under_its_family(
    name: str, filed: str | None, carried: dict[str, Any] | None
) -> None:
    """A typestr is filed under its family and carries its byte order, size and unit. `struct` is filed under itself. A typestr whose type code the spec does not list is filed under itself, so a scope leaves it unclaimed. A family name is not filed under anything, because no document writes it."""
    assert filed_under(ZarrV2DataTypeDefinition, name) == (filed, carried)


@pytest.mark.parametrize(
    ("value", "name", "configuration"),
    [
        ("<f4", "<f4", {"byteorder": "<", "itemsize": 4}),
        (
            [["x", "<f4"], ["y", "<i4", [2]]],
            "struct",
            {"fields": (("x", "<f4"), ("y", "<i4", (2,)))},
        ),
        ((("x", "<f4"),), "struct", {"fields": (("x", "<f4"),)}),
        (3, None, None),
        ({"name": "float32"}, None, None),
    ],
    ids=["string", "records", "refined-records", "number", "v3-object"],
)
def test_a_v2_dtype_is_a_string_or_records(
    value: object, name: str | None, configuration: object
) -> None:
    """A dtype field is either a string, which is its name and carries its configuration, or an array of field records, which is a `struct` whose configuration is the records. Any other value has no name, and is refused."""
    field, _ = resolve(value, ZarrV2DataTypeDefinition, CORE_V2)
    assert field.name == name
    if configuration is None:
        assert isinstance(field, RefusedField)
    else:
        assert isinstance(field, AcceptedField)
        assert field.configuration == configuration


@pytest.mark.parametrize(
    ("value", "problems"),
    [
        ("<f4", []),
        ("struct", [((), "missing_key")]),
        ([["x", "<f4"]], []),
        ("float32", [((), "invalid_value")]),
        (3, [((), "invalid_type")]),
        ({"name": "float32"}, [((), "invalid_type")]),
    ],
)
def test_a_v2_dtype_envelope_is_a_typestr_or_records(
    value: object, problems: list[tuple[Loc, str]]
) -> None:
    """The envelope of a dtype field is valid when it is a typestr or an array of records. `struct` is a valid name too, and its definition then requires the records. A string that is neither is reported as an invalid value, and the message names the typestr form. Any other value is reported as an invalid type."""
    _, found = resolve(value, ZarrV2DataTypeDefinition, CORE_V2)
    assert [(p.loc, p.kind) for p in found] == problems
    if value == "float32":
        assert "typestr" in found[0].message


def test_a_v2_dtype_writes_back_as_a_string_or_records() -> None:
    """A dtype is written back as its string, and a struct as its records. Both the name and the configuration are located at the field: a string that is not a typestr is reported at the field, and the records are a member of a configuration located at the field."""
    typestr, _ = resolve("<f4", ZarrV2DataTypeDefinition, CORE_V2)
    records = (("x", "<f4"),)
    struct, _ = resolve(records, ZarrV2DataTypeDefinition, CORE_V2)
    assert isinstance(typestr, AcceptedField)
    assert isinstance(struct, AcceptedField)
    assert typestr.to_json() == "<f4"
    assert struct.to_json() == records
    _, found = resolve("float32", ZarrV2DataTypeDefinition, CORE_V2, ("dtype",))
    assert [p.loc for p in found] == [("dtype",)]
    _, found = resolve((("x", "<f9"),), ZarrV2DataTypeDefinition, CORE_V2, ("dtype",))
    assert [p.loc for p in found] == [("dtype", "fields", 0, 1)]


@pytest.mark.parametrize(
    ("value", "name", "configuration", "problems"),
    [
        ({"id": "zlib", "level": 1}, "zlib", {"level": 1}, []),
        ({"id": "packbits"}, "packbits", {}, []),
        ({"level": 1}, None, None, [(("id",), "missing_key")]),
        ({"id": 3}, None, None, [(("id",), "invalid_type")]),
        ({"id": ""}, "", None, [(("id",), "invalid_value")]),
        ("zlib", None, None, [((), "invalid_type")]),
    ],
    ids=["parameters", "bare", "no-id", "id-not-a-string", "empty-id", "not-an-object"],
)
def test_a_v2_codec_is_an_object_with_a_string_id(
    value: object,
    name: str | None,
    configuration: dict[str, Any] | None,
    problems: list[tuple[Loc, str]],
) -> None:
    """A codec field is an object. Its `id` is its name and its other members are its parameters. A value that is not an object, or has no string `id`, has no name, and the problems say why. An empty `id` is a name no document may use."""
    field, found = resolve(value, ZarrV2CodecDefinition, CORE_V2)
    assert field.name == name
    if configuration is None:
        assert isinstance(field, RefusedField)
    else:
        assert isinstance(field, AcceptedField)
        assert field.configuration == configuration
    assert [(p.loc, p.kind) for p in found] == problems


def test_a_v2_codec_writes_back_with_its_id_beside_its_parameters() -> None:
    """A codec is written back as `{"id": name, **parameters}`. The parameters are located at the field, and the name is located under `id`. A codec id is filed under itself."""
    field, _ = resolve({"id": "zlib", "level": 1}, ZarrV2CodecDefinition, CORE_V2)
    assert isinstance(field, AcceptedField)
    assert field.to_json() == {"id": "zlib", "level": 1}
    at = ("compressor",)
    _, found = resolve({"id": "zlib", "level": 99}, ZarrV2CodecDefinition, CORE_V2, at)
    assert [p.loc for p in found] == [("compressor", "level")]
    _, found = resolve({"id": "", "level": math.nan}, ZarrV2CodecDefinition, CORE_V2, at)
    assert [p.loc for p in found] == [("compressor", "id"), ("compressor", "level")]
    assert filed_under(ZarrV2CodecDefinition, "zlib") == ("zlib", None)


def test_error_a_v2_definition_is_named_as_its_format_names_one() -> None:
    """Building a v2 codec definition with an empty id raises an error. So does building a v2 data type definition named by a typestr that a document would write, instead of by a family name."""
    with pytest.raises(TypeError, match="so no document names it"):
        ZarrV2CodecDefinition(name="", configuration=EmptyConfiguration)
    with pytest.raises(TypeError, match="is how a document writes"):
        ZarrV2DataTypeDefinition(name="<f4", configuration=EmptyConfiguration)
