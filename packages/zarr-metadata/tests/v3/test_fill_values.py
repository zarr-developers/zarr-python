"""Fill values, judged by the data type they fill.

Each data type's definition declares the JSON shape of its fill value and
the rules for one of that shape; `fill_value_problems` judges a fill value
against a data type field a scope read, and the v3 array validators judge
a document's `fill_value` against its `data_type`.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing_extensions import TypedDict

from zarr_metadata._json import JSON_DEPTH, value_at
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model import validate_array_metadata_v3, validate_group_metadata_v3
from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.v3.data_type._float import FloatWidth, float_bits
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    DataTypeDefinition,
    EmptyConfiguration,
    JSONValue,
    Nested,
    Read,
    ValidationProblem,
    canonical_fill_value,
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


def test_a_fill_value_nested_as_deep_as_a_reader_walks_is_read() -> None:
    def deep(levels: int) -> dict[str, object]:
        value: dict[str, object] = {}
        for _ in range(levels):
            value = {"x": value}
        return value

    document = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {
        "data_type": STRUCT,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        # `c` sits two levels down, and holds the deepest value the cap admits.
        "fill_value": {"a": 1, "b": 0.5, "c": deep(JSON_DEPTH - 3)},
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


# --- canonical spellings ---------------------------------------------------


def _alike(left: object, right: object) -> bool:
    """Whether two JSON values are written alike: `==` takes `-0.0` for `0.0`."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _read(data_type: JSONValue) -> Read[DataTypeDefinition[Any]]:
    resolved, found = resolve(data_type, DataTypeDefinition, CORE_AND_EXTENSIONS)
    assert found == ()
    assert isinstance(resolved, Read)
    return resolved


@pytest.mark.parametrize(
    ("data_type", "spellings", "canonical"),
    [
        ("float32", ["NaN", "0x7fc00000", "0x7FC00000"], "NaN"),
        # Any other NaN is its bits: the spec names the one.
        ("float32", ["0xffc00000"], "0xffc00000"),
        ("float32", ["0x7fc00001", "0x7FC00001"], "0x7fc00001"),
        ("float32", ["Infinity", "0x7f800000", 3.5e38, 10**39], "Infinity"),
        ("float32", ["-Infinity", "0xff800000", -(10**39)], "-Infinity"),
        ("float32", [0, 0.0, "0x00000000", 1e-46], 0.0),
        # Zero's sign is a value of its own.
        ("float32", [-0.0, "0x80000000"], -0.0),
        ("float32", [1, 1.0, "0x3f800000"], 1.0),
        # The number of the fewest digits that rounds to the value.
        ("float32", [0.1, 0.10000000149011612, "0x3dcccccd"], 0.1),
        # A number is read as a float64, as a JSON parser reads one, and
        # then rounded to the type, as zarrs, tensorstore and numpy round
        # it: an integer and a number with a fraction that read as one
        # float64 are one value, and an integer whose float64 is halfway
        # between two float32 values is the even one, as they store it.
        ("float32", [2**53 + 2**29 + 1, 9007199791611905.0, 2**53], 9007199000000000.0),
        ("float32", [2**60 + 2**36 + 1, 2**60, 1.1529215e18], 1.1529215e18),
        # Of the numbers of the fewest digits, the one past the nearest: at
        # a power of two, those that round to it reach further above it.
        ("float16", [0.015625, "0x2400"], 0.01563),
        ("float32", ["0x6b000000"], 1.5474251e26),
        ("float16", [65504, 65519, "0x7bff"], 65500.0),
        ("float16", [65520, "Infinity", "0x7c00"], "Infinity"),
        # Halfway rounds to the even value.
        ("float16", [2049, 2048], 2048.0),
        ("float16", [2051, 2052], 2052.0),
        ("float16", ["NaN", "0x7e00", "0x7E00"], "NaN"),
        ("float16", [-0.0, "0x8000"], -0.0),
        ("float64", [2**53 + 1, 2**53], float(2**53)),
        ("float64", [10**400, "Infinity", "0x7ff0000000000000"], "Infinity"),
        ("float64", [-(10**400), "-Infinity", "0xfff0000000000000"], "-Infinity"),
        ("float64", ["NaN", "0x7ff8000000000000"], "NaN"),
        ("complex64", [["NaN", -0.0], ["0x7fc00000", "0x80000000"]], ("NaN", -0.0)),
        ("complex128", [[0.1, 1], ["0x3fb999999999999a", 1.0]], (0.1, 1.0)),
        ("bytes", [[65], "QQ==", "QR=="], "QQ=="),
        ("bytes", [[], ""], ""),
        (DATETIME, ["NaT", -(2**63)], "NaT"),
        (TIMEDELTA, ["NaT", -(2**63)], "NaT"),
        (TIMEDELTA, [5], 5),
        (STRUCT, [{"a": 1, "b": "0x7fc00000"}, {"b": "NaN", "a": 1}], {"a": 1, "b": "NaN"}),
        # A field type nothing in scope claims spells its fill value as written.
        (
            {
                "name": "struct",
                "configuration": {"fields": [{"name": "a", "data_type": "acme.decimal"}]},
            },
            [{"a": -0.0}],
            {"a": -0.0},
        ),
        # A type that spells each value one way spells it as written.
        ("int8", [-128], -128),
        ("bool", [True], True),
        ("string", ["NaN"], "NaN"),
        ("r16", [[0, 1]], (0, 1)),
    ],
)
def test_a_fill_value_s_canonical_spelling_is_its_value_s(
    data_type: JSONValue, spellings: list[object], canonical: JSONValue
) -> None:
    read = _read(data_type)
    for spelling in spellings:
        spelled = canonical_fill_value(read, spelling)
        assert _alike(spelled, canonical), (spelling, spelled)
    # A fill value of the type, spelled as itself.
    assert fill_value_problems(read, canonical) == ()
    assert _alike(canonical_fill_value(read, canonical), canonical)


def test_a_data_type_nothing_in_scope_claims_spells_a_fill_value_as_written() -> None:
    unclaimed, _ = resolve("acme.decimal", DataTypeDefinition, CORE_AND_EXTENSIONS)
    values: list[JSONValue] = [-0.0, True, 1, [1.0], None]
    for value in values:
        assert _alike(canonical_fill_value(unclaimed, value), value)


WIDTHS: tuple[FloatWidth, ...] = get_args(FloatWidth)


@given(st.data())
def test_a_float_s_canonical_spelling_spells_its_bits(data: st.DataObject) -> None:
    # Every value of each float type, NaNs among them: its canonical
    # spelling is a fill value of the type, spelling the same bits, and
    # its own canonical spelling.
    width = data.draw(st.sampled_from(WIDTHS))
    # Drawn by part, so normal values, infinities and NaNs each turn up:
    # bits drawn whole are almost all subnormals.
    fraction = {16: 10, 32: 23, 64: 52}[width]
    sign = data.draw(st.integers(0, 1))
    exponent = data.draw(st.integers(0, 2 ** (width - 1 - fraction) - 1))
    mantissa = data.draw(st.integers(0, 2**fraction - 1))
    bits = (sign << (width - 1)) | (exponent << fraction) | mantissa
    read = _read(f"float{width}")
    spelled = canonical_fill_value(read, f"0x{bits:0{width // 4}x}")
    assert spelled is not UNSET
    assert fill_value_problems(read, spelled) == ()
    assert float_bits(cast("float | str", spelled), width) == bits
    assert _alike(canonical_fill_value(read, spelled), spelled)


def test_error_a_fill_value_with_a_problem_has_no_canonical_spelling() -> None:
    # `UNSET`, since `None` is the JSON null, a fill value of a data type
    # the scope did not read.
    assert canonical_fill_value(_read("float32"), "0x7fc0") is UNSET
    assert canonical_fill_value(_read("int8"), 1.0) is UNSET
    assert canonical_fill_value(_read("int8"), math.nan) is UNSET
    unclaimed, _ = resolve("acme.decimal", DataTypeDefinition, CORE_AND_EXTENSIONS)
    assert canonical_fill_value(unclaimed, None) is None


def test_error_a_canonical_spelling_that_raises_says_which_data_type_raised_it() -> None:
    def refuses(configuration: EmptyConfiguration, nested: Nested, value: object) -> JSONValue:
        raise ValueError("no")

    scope = CORE_AND_EXTENSIONS.extended_with(
        dataclasses.replace(ACME_POINT, fill_value_canonical=refuses)
    )
    resolved, _ = resolve("acme.point", DataTypeDefinition, scope)
    with pytest.raises(ValueError, match="no") as raised:
        canonical_fill_value(resolved, {"x": 1})
    assert raised.value.__notes__ == ["raised by the fill_value_canonical of 'acme.point'"]


def test_error_a_canonical_spelling_that_is_not_json_is_refused() -> None:
    def not_json(configuration: EmptyConfiguration, nested: Nested, value: object) -> JSONValue:
        return cast("JSONValue", {1, 2})

    scope = CORE_AND_EXTENSIONS.extended_with(
        dataclasses.replace(ACME_POINT, fill_value_canonical=not_json)
    )
    resolved, _ = resolve("acme.point", DataTypeDefinition, scope)
    with pytest.raises(TypeError, match="'acme.point': its fill_value_canonical gives JSON"):
        canonical_fill_value(resolved, {"x": 1})


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("int8", -(2**7), 2**7 - 1),
        ("int16", -(2**15), 2**15 - 1),
        ("int32", -(2**31), 2**31 - 1),
        ("int64", -(2**63), 2**63 - 1),
        ("uint8", 0, 2**8 - 1),
        ("uint16", 0, 2**16 - 1),
        ("uint32", 0, 2**32 - 1),
        ("uint64", 0, 2**64 - 1),
    ],
)
def test_an_integer_type_takes_exactly_its_range(name: str, low: int, high: int) -> None:
    read = _read(name)
    assert fill_value_problems(read, low) == ()
    assert fill_value_problems(read, high) == ()
    for outside in (low - 1, high + 1):
        (problem,) = fill_value_problems(read, outside)
        assert (problem.loc, problem.kind, dict(problem.ctx)) == (
            (),
            "invalid_value",
            {"ge": low, "le": high},
        )
