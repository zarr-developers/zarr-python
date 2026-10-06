"""The two kinds of a v2 array document's fields: a dtype as NumPy spells one, and a numcodecs configuration."""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.v2._definition import (
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    parse_typestr,
)
from zarr_metadata.v3.definition import EmptyConfiguration

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
    """A NumPy typestr splits into its type code and what the name carries: the byte order, the item size when written, and a time unit with its multiplier when bracketed."""
    assert parse_typestr(name) == (code, carried)


@pytest.mark.parametrize(
    "name", ["float32", "f4", "<f", "<f4x", "", "<", "<f4[ns", "[('a','<f4')]", "<M8[ns][s]"]
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
    """A typestr is filed under its family and carries its byte order, size and unit; `struct` is filed as itself; a typestr of a type code the spec does not list is filed under itself, so a scope leaves it unclaimed; a family name is filed by nothing, since no document writes it."""
    assert ZarrV2DataTypeDefinition.spelled(name) == (filed, carried)


@pytest.mark.parametrize(
    ("value", "name", "configuration"),
    [
        ("<f4", "<f4", None),
        (
            [["x", "<f4"], ["y", "<i4", [2]]],
            "struct",
            {"fields": [["x", "<f4"], ["y", "<i4", [2]]]},
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
    """A dtype field is a string, which is its name, or an array of field records, which is a `struct` whose configuration is the records; anything else names nothing."""
    assert ZarrV2DataTypeDefinition.named_configuration(value) == (name, configuration, ())


@pytest.mark.parametrize(
    ("value", "problems"),
    [
        ("<f4", []),
        ("struct", []),
        ([["x", "<f4"]], []),
        ("float32", [((), "invalid_value")]),
        (3, [((), "invalid_type")]),
        ({"name": "float32"}, [((), "invalid_type")]),
    ],
)
def test_a_v2_dtype_envelope_is_a_typestr_or_records(
    value: object, problems: list[tuple[Loc, str]]
) -> None:
    """The envelope of a dtype field holds when it is a typestr or an array of records; a string that is neither is reported as an invalid value, naming the typestr form, and anything else as an invalid type."""
    found = ZarrV2DataTypeDefinition.envelope_problems(value)
    assert [(p.loc, p.kind) for p in found] == problems
    if value == "float32":
        assert "typestr" in found[0].message


def test_a_v2_dtype_writes_back_as_a_string_or_records() -> None:
    """`envelope_json` writes a dtype as the string that carries it, and a struct as its records; its name and configuration both sit at the field."""
    assert ZarrV2DataTypeDefinition.envelope_json("<f4", {}) == "<f4"
    records = (("x", "<f4"),)
    assert ZarrV2DataTypeDefinition.envelope_json("struct", {"fields": records}) == records
    assert ZarrV2DataTypeDefinition.configuration_loc(("dtype",)) == ("dtype",)
    assert ZarrV2DataTypeDefinition.name_loc(("dtype",)) == ("dtype",)


@pytest.mark.parametrize(
    ("value", "name", "configuration", "problems"),
    [
        ({"id": "zlib", "level": 1}, "zlib", {"level": 1}, []),
        ({"id": "packbits"}, "packbits", {}, []),
        ({"level": 1}, None, None, [(("id",), "missing_key")]),
        ({"id": 3}, None, None, [(("id",), "invalid_type")]),
        ({"id": ""}, "", {}, [(("id",), "invalid_value")]),
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
    """A codec field is an object whose `id` names it and whose other members are its parameters; one without a string `id`, or not an object, names nothing, and the envelope says why."""
    assert ZarrV2CodecDefinition.named_configuration(value) == (name, configuration, ())
    assert [(p.loc, p.kind) for p in ZarrV2CodecDefinition.envelope_problems(value)] == problems


def test_a_v2_codec_writes_back_with_its_id_beside_its_parameters() -> None:
    """`envelope_json` writes a codec as `{"id": name, **parameters}`: its parameters at the field and its name under `id`."""
    assert ZarrV2CodecDefinition.envelope_json("zlib", {"level": 1}) == {"id": "zlib", "level": 1}
    assert ZarrV2CodecDefinition.configuration_loc(("compressor",)) == ("compressor",)
    assert ZarrV2CodecDefinition.name_loc(("compressor",)) == ("compressor", "id")
    assert ZarrV2CodecDefinition.spelled("zlib") == ("zlib", None)


def test_error_a_v2_definition_is_named_as_its_format_names_one() -> None:
    """A v2 codec definition with an empty id, and a v2 data type definition named by a typestr a document would write rather than a family, are refused when built."""
    with pytest.raises(TypeError, match="so no document names it"):
        ZarrV2CodecDefinition(name="", configuration=EmptyConfiguration)
    with pytest.raises(TypeError, match="is how a document writes"):
        ZarrV2DataTypeDefinition(name="<f4", configuration=EmptyConfiguration)
