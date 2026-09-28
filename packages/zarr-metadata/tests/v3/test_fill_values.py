"""Fill values, judged by the data type they fill.

Each data type's definition declares the JSON shape of its fill value and
the rules for one of that shape; `fill_value_problems` judges a fill value
against a data type field a scope read, and the v3 array validators judge
a document's `fill_value` against its `data_type`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypedDict

from zarr_metadata._json import value_at
from zarr_metadata.model import validate_array_metadata_v3, validate_group_metadata_v3
from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    DataTypeDefinition,
    EmptyConfiguration,
    JSONValue,
    Nested,
    Read,
    ValidationProblem,
    fill_value_problems,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

DATETIME: JSONValue = {
    "name": "numpy.datetime64",
    "configuration": {"unit": "s", "scale_factor": 1},
}
TIMEDELTA: JSONValue = {
    "name": "numpy.timedelta64",
    "configuration": {"unit": "ms", "scale_factor": 1},
}
STRUCT: JSONValue = {
    "name": "struct",
    "configuration": {
        "fields": [
            {"name": "a", "data_type": "int8"},
            {"name": "b", "data_type": "float32"},
        ]
    },
}


class AcmePointFillValue(TypedDict, closed=True):
    x: int


def acme_point_fill_value_rules(
    configuration: EmptyConfiguration, nested: Nested, value: AcmePointFillValue
) -> Iterator[ValidationProblem]:
    if value["x"] > 10:
        yield ValidationProblem(("x",), "expected x <= 10", "invalid_value")


ACME_POINT = DataTypeDefinition(
    name="acme.point",
    configuration=EmptyConfiguration,
    fill_value=AcmePointFillValue,
    fill_value_rules=acme_point_fill_value_rules,
)


def _problems(data_type: JSONValue, value: object) -> list[tuple[tuple[str | int, ...], str]]:
    resolved, found = resolve(data_type, DataTypeDefinition, CORE_AND_EXTENSIONS)
    assert found == ()
    return [(problem.loc, problem.kind) for problem in fill_value_problems(resolved, value)]


@pytest.mark.parametrize(
    ("data_type", "fill_value"),
    [
        ("bool", False),
        ("int8", -128),
        ("int8", 127),
        ("uint8", 255),
        ("int64", -(2**63)),
        ("uint64", 2**64 - 1),
        ("float16", 1.5),
        # A number takes any value: a reader rounds it to the type.
        ("float16", 1e10),
        ("float32", 0),
        ("float32", "NaN"),
        ("float32", "-Infinity"),
        ("float16", "0x7e00"),
        ("float32", "0x7fc00000"),
        ("float64", "0x7ff8000000000000"),
        ("complex64", [1.5, "NaN"]),
        ("complex128", ["0x7ff8000000000000", 0]),
        ("r8", [255]),
        ("r16", [0, 1]),
        ("r008", [7]),
        ("bytes", [0, 255]),
        ("bytes", "AQI="),
        ("bytes", ""),
        ("string", "a"),
        (DATETIME, "NaT"),
        (DATETIME, -(2**63)),
        (TIMEDELTA, 2**63 - 1),
        (STRUCT, {"a": 1, "b": "NaN"}),
        # A field type nothing in scope claims leaves its fill value unjudged.
        (
            {
                "name": "struct",
                "configuration": {"fields": [{"name": "a", "data_type": "acme.decimal"}]},
            },
            {"a": "anything"},
        ),
        # So does a data type nothing in scope claims, or one that is not read.
        ("acme.decimal", "anything"),
        (
            {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 0}},
            "anything",
        ),
    ],
)
def test_every_data_type_takes_its_fill_values(data_type: JSONValue, fill_value: object) -> None:
    resolved, _ = resolve(data_type, DataTypeDefinition, CORE_AND_EXTENSIONS)
    assert fill_value_problems(resolved, fill_value) == ()


@pytest.mark.parametrize(
    ("data_type", "fill_value"),
    [
        ("bool", 0),
        ("int8", True),
        ("int8", 1.0),
        ("int8", "5"),
        ("float32", None),
        ("string", 1),
        ("complex64", [1]),
        ("r16", "AQI="),
        (DATETIME, 1.5),
        (STRUCT, [1, 2]),
    ],
)
def test_error_a_fill_value_of_the_wrong_json_type(
    data_type: JSONValue, fill_value: object
) -> None:
    assert _problems(data_type, fill_value) == [((), "invalid_type")]


def test_error_a_fill_value_that_is_not_json() -> None:
    # A float's NaN is the string "NaN"; the number is not JSON.
    assert _problems("float32", math.nan) == [((), "invalid_value")]


@pytest.mark.parametrize(
    ("data_type", "fill_value"),
    [
        ("int8", 128),
        ("int8", -129),
        ("uint8", -1),
        ("uint64", 2**64),
        ("int64", -(2**63) - 1),
    ],
)
def test_error_an_integer_fill_value_out_of_its_type_s_range(
    data_type: str, fill_value: int
) -> None:
    assert _problems(data_type, fill_value) == [((), "invalid_value")]


@pytest.mark.parametrize(
    ("data_type", "fill_value", "loc"),
    [
        # Not one of the named values: the spec spells them this way only.
        ("float32", "nan", ()),
        # A hex string of another type's width.
        ("float32", "0x7e00", ()),
        ("float16", "0x7fc00000", ()),
        ("complex64", [1, "0x7ff8000000000000"], (1,)),
    ],
)
def test_error_a_float_string_that_is_not_a_named_value_or_a_hex_string_of_its_width(
    data_type: str, fill_value: object, loc: tuple[int, ...]
) -> None:
    assert _problems(data_type, fill_value) == [(loc, "invalid_value")]


@pytest.mark.parametrize(("data_type", "fill_value"), [("r16", [1]), ("r8", [1, 2])])
def test_error_a_raw_bits_fill_value_of_another_size(data_type: str, fill_value: object) -> None:
    # One byte value for each 8 bits of the size.
    assert _problems(data_type, fill_value) == [((), "invalid_value")]


@pytest.mark.parametrize(
    ("data_type", "fill_value", "loc"), [("r16", [1, 256], (1,)), ("bytes", [-1], (0,))]
)
def test_error_a_byte_value_out_of_range(
    data_type: str, fill_value: object, loc: tuple[int, ...]
) -> None:
    assert _problems(data_type, fill_value) == [(loc, "invalid_value")]


@pytest.mark.parametrize(
    ("data_type", "fill_value", "loc", "ctx"),
    [
        ("int8", 128, (), {"ge": -128, "le": 127}),
        ("uint16", -1, (), {"ge": 0, "le": 2**16 - 1}),
        ("int64", 2**63, (), {"ge": -(2**63), "le": 2**63 - 1}),
        ("uint64", 2**64, (), {"ge": 0, "le": 2**64 - 1}),
        ("r16", [1, 256], (1,), {"ge": 0, "le": 255}),
        ("bytes", [-1], (0,), {"ge": 0, "le": 255}),
        (DATETIME, 2**63, (), {"ge": -(2**63), "le": 2**63 - 1}),
    ],
    ids=["int8", "uint16", "int64", "uint64", "raw-bits-byte", "bytes-byte", "numpy-time"],
)
def test_a_fill_value_s_range_is_its_type_s(
    data_type: JSONValue, fill_value: object, loc: tuple[int, ...], ctx: dict[str, int]
) -> None:
    # `Int8FillValue` is `Annotated[int, Interval(ge=-128, le=127)]`: the
    # problem holds the range, and what was found.
    resolved, _ = resolve(data_type, DataTypeDefinition, CORE_AND_EXTENSIONS)
    problems = fill_value_problems(resolved, fill_value)
    assert [(p.loc, p.input, dict(p.ctx)) for p in problems] == [
        (loc, value_at(fill_value, loc), ctx)
    ]


@pytest.mark.parametrize("fill_value", ["!!", "AQI"])
def test_error_a_bytes_fill_value_that_is_not_base64(fill_value: str) -> None:
    assert _problems("bytes", fill_value) == [((), "invalid_value")]


@pytest.mark.parametrize(
    ("data_type", "fill_value"), [(DATETIME, 2**63), (TIMEDELTA, -(2**63) - 1)]
)
def test_error_a_time_fill_value_outside_a_signed_64_bit_integer(
    data_type: JSONValue, fill_value: int
) -> None:
    assert _problems(data_type, fill_value) == [((), "invalid_value")]


def test_error_a_struct_fill_value_missing_a_field() -> None:
    assert _problems(STRUCT, {"a": 1}) == [(("b",), "missing_key")]


def test_error_a_struct_fill_value_for_no_field() -> None:
    assert _problems(STRUCT, {"a": 1, "b": 0, "c": 2}) == [(("c",), "unknown_key")]


def test_error_a_struct_field_s_fill_value_its_own_type_refuses() -> None:
    # Judged by the field's type, as the scope read it, at the field's name.
    assert _problems(STRUCT, {"a": 300, "b": "x"}) == [
        (("a",), "invalid_value"),
        (("b",), "invalid_value"),
    ]


def test_error_an_array_document_s_fill_value_its_data_type_refuses() -> None:
    # Judged against the data type the document names, and in a
    # consolidated group, where the array sits.
    document = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"fill_value": 300}
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("fill_value",), "invalid_value")
    ]
    group = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": document},
        },
    }
    assert [(p.loc, p.kind) for p in validate_group_metadata_v3(group)] == [
        (("consolidated_metadata", "metadata", "a", "fill_value"), "invalid_value")
    ]


def test_error_a_key_the_fill_value_shape_does_not_declare_hides_no_rule() -> None:
    # Reported, and left out of what the rules see, which still judge the rest.
    scope = CORE_AND_EXTENSIONS.extended_with(ACME_POINT)
    resolved, _ = resolve("acme.point", DataTypeDefinition, scope)
    found = fill_value_problems(resolved, {"x": 99, "extra": 1})
    assert [(problem.loc, problem.kind) for problem in found] == [
        (("extra",), "unknown_key"),
        (("x",), "invalid_value"),
    ]


def test_a_fill_value_nested_hundreds_deep_is_read() -> None:
    def deep(levels: int) -> dict[str, object]:
        value: dict[str, object] = {}
        for _ in range(levels):
            value = {"x": value}
        return value

    document = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {
        "data_type": STRUCT,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": {"a": 1, "b": 0.5, "c": deep(600)},
    }
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("fill_value", "c"), "unknown_key")
    ]


def test_a_struct_read_without_its_field_types_leaves_its_fields_unjudged() -> None:
    # A reading built by hand, holding no field type's reading.
    configuration = {"fields": ({"name": "a", "data_type": "int8"},)}
    struct = Read(
        json=STRUCT, name="struct", definition=STRUCT_DATA_TYPE, configuration=configuration
    )
    assert fill_value_problems(struct, {"a": 300}) == ()
