"""Reading each v2 data type by its family: which typestrs and which fill values each family accepts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    RefusedField,
    UnclaimedField,
    canonical_fill_value,
    canonical_of,
    fill_value_problems,
    resolve,
)
from zarr_metadata.definition._definition import fields_of
from zarr_metadata.v2.data_type import DATA_TYPES_V2
from zarr_metadata.v2.definition import ZarrV2Context, ZarrV2DataTypeDefinition

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

CONTEXT = ZarrV2Context(data_types=DATA_TYPES_V2)

Loc = tuple[str | int, ...]


def _read(value: object) -> tuple[type, list[tuple[Loc, str]]]:
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, CONTEXT, ("dtype",))
    return type(resolved), [(p.loc, p.kind) for p in problems]


@pytest.mark.parametrize(
    ("value", "family", "canonical"),
    [
        ("|b1", "bool", "|b1"),
        ("<b1", "bool", "|b1"),
        ("|i1", "int", "|i1"),
        ("<i2", "int", "<i2"),
        (">i4", "int", ">i4"),
        ("<i8", "int", "<i8"),
        ("|u1", "uint", "|u1"),
        (">u8", "uint", ">u8"),
        ("<f2", "float", "<f2"),
        ("<f4", "float", "<f4"),
        (">f8", "float", ">f8"),
        ("<c8", "complex", "<c8"),
        (">c16", "complex", ">c16"),
        ("|S0", "bytes", "|S0"),
        ("|S12", "bytes", "|S12"),
        ("<S3", "bytes", "|S3"),
        ("<U5", "str", "<U5"),
        (">U0", "str", ">U0"),
        ("|V8", "void", "|V8"),
        ("|O", "object", "|O"),
        ("<O", "object", "|O"),
        ("<M8[ns]", "datetime64", "<M8[ns]"),
        (">M8[10s]", "datetime64", ">M8[10s]"),
        ("<M8[μs]", "datetime64", "<M8[us]"),
        ("<m8[D]", "timedelta64", "<m8[D]"),
        ([["x", "<f4"], ["y", "<i4", [2]]], "struct", (("x", "<f4"), ("y", "<i4", (2,)))),
        ([["r", "|u1"], ["s", [["t", "<f4"]]]], "struct", (("r", "|u1"), ("s", (("t", "<f4"),)))),
        # NumPy names the padding fields of an aligned struct "", and zarr 2.x writes them.
        (
            [["a", "|i1"], ["", "|V3"], ["b", "<i4"], ["", "|V3"]],
            "struct",
            (("a", "|i1"), ("", "|V3"), ("b", "<i4"), ("", "|V3")),
        ),
    ],
)
def test_every_family_reads_its_typestrs(value: object, family: str, canonical: object) -> None:
    """Each typestr of a family is read by the family's definition. Its canonical spelling is the typestr NumPy writes: `|` for a type of one byte or with no byte order, and `us` for a microsecond unit. A records array is read as `struct`, and the type of each record is read too."""
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, CONTEXT)
    assert isinstance(resolved, AcceptedField)
    assert resolved.definition.name == family
    assert problems == ()
    assert canonical_of(resolved, problems) == canonical


@pytest.mark.parametrize("value", ["<e2", "<T16", "|a1"])
def test_a_type_code_the_spec_does_not_list_is_unclaimed(value: str) -> None:
    """A typestr whose type code is not listed in the v2 spec is filed under its own name, which no definition in scope claims. It is read as `UnclaimedField` with no problem, and written back unchanged."""
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, CONTEXT)
    assert isinstance(resolved, UnclaimedField)
    assert problems == ()
    assert resolved.to_json() == value


@pytest.mark.parametrize(
    ("value", "at"),
    [
        ("float32", ("dtype",)),
        ("<f", ("dtype",)),
        ("|S", ("dtype",)),
        ("<b2", ("dtype",)),
        ("<i3", ("dtype",)),
        ("|i4", ("dtype",)),
        ("<f16", ("dtype",)),
        ("|f4", ("dtype",)),
        ("<c4", ("dtype",)),
        ("<f4[ns]", ("dtype",)),
        ("<M8", ("dtype",)),
        ("<M4[ns]", ("dtype",)),
        ("<M8[ns][s]", ("dtype",)),
        ("<M8[x]", ("dtype",)),
        ("|O4", ("dtype",)),
        ([], ("dtype", "fields")),
        ([["x", "<f4"], ["x", "<i4"]], ("dtype", "fields", 1, 0)),
        ([["x", "float32"]], ("dtype", "fields", 0, 1)),
        ([["x", "<f4", [-1]]], ("dtype", "fields", 0, 2, 0)),
        ([["x"]], ("dtype", "fields", 0)),
        ([["x", "<f4", [2], 0]], ("dtype", "fields", 0)),
        ([["r", "|u1"], ["s", [["t", "float32"]]]], ("dtype", "fields", 1, 1, "fields", 0, 1)),
    ],
    ids=[
        "no-typestr",
        "no-size",
        "no-size-bytes",
        "bool-size",
        "int-size",
        "int-order",
        "float-size",
        "float-order",
        "complex-size",
        "float-unit",
        "time-no-unit",
        "time-size",
        "two-units",
        "bad-unit",
        "object-size",
        "no-records",
        "duplicate-name",
        "record-type",
        "record-shape",
        "short-record",
        "long-record",
        "nested-record-type",
    ],
)
def test_error_a_typestr_the_family_does_not_take_is_a_problem(value: object, at: Loc) -> None:
    """A typestr with a size, byte order or unit that its family does not accept is refused, and the problem is located at the field. A struct's problems are located under `fields`, at the record and the position inside it. If a record's type is refused, the problem is reported there and the struct is still read."""
    kind, problems = _read(value)
    assert kind is RefusedField or "fields" in at
    assert next(loc for loc, _ in problems) == at


@pytest.mark.parametrize(
    ("dtype", "value", "canonical"),
    [
        ("|b1", True, True),
        ("|b1", None, None),
        ("<i2", -32768, -32768),
        ("<u2", 65535, 65535),
        ("<i8", None, None),
        ("<f4", 0, 0.0),
        ("<f4", 1.5, 1.5),
        ("<f8", "NaN", "NaN"),
        ("<f2", "-Infinity", "-Infinity"),
        ("<c8", [1, "Infinity"], (1.0, "Infinity")),
        ("|S3", "YWJj", "YWJj"),
        ("|S3", None, None),
        ("<U3", "abc", "abc"),
        ("|V2", "AAA=", "AAA="),
        ("|O", {"any": [1, "json"]}, {"any": (1, "json")}),
        ("<M8[ns]", 0, 0),
        ("<M8[ns]", "NaT", "NaT"),
        ("<m8[s]", -(2**63), "NaT"),
        ([["x", "<f4"]], "AAAAAA==", "AAAAAA=="),
    ],
)
def test_every_family_takes_its_fill_values(
    dtype: object, value: object, canonical: object
) -> None:
    """Each family accepts the fill values the v2 spec allows for it, and always accepts `null`: booleans; integers in the type's range; numbers or the named non-finite strings for floats and for each complex component; base64 for bytes, void and struct; a string for str; any JSON for object; and ticks or `NaT` for times. The canonical spelling turns an integer written for a float into a float, and turns the ticks value of `NaT` into `"NaT"`."""
    resolved, _ = resolve(dtype, ZarrV2DataTypeDefinition, CONTEXT)
    assert fill_value_problems(resolved, value) == ()
    assert canonical_fill_value(resolved, value) == canonical


@pytest.mark.parametrize(
    ("dtype", "value", "at"),
    [
        ("|b1", 1, ()),
        ("<i2", 32768, ()),
        ("<u1", -1, ()),
        ("<i4", "NaN", ()),
        ("<f4", "nan", ()),
        ("<f4", "0x7fc00000", ()),
        ("<c8", [1, 2, 3], ()),
        ("<c8", ["x", 1], (0,)),
        ("|S3", "not base64!", ()),
        ("|S3", [1, 2, 3], ()),
        ("<U3", 3, ()),
        ("<M8[ns]", 1.5, ()),
        ("<M8[ns]", "nat", ()),
        ([["x", "<f4"]], {"x": 1.0}, ()),
    ],
)
def test_error_a_fill_value_the_family_does_not_take_is_a_problem(
    dtype: object, value: object, at: Loc
) -> None:
    """A fill value outside the shape or range its family accepts is reported, at the value or at the wrong component. A v2 float does not accept a hex string, and a struct's fill value is the base64 of the record, not an object of fields."""
    resolved, _ = resolve(dtype, ZarrV2DataTypeDefinition, CONTEXT)
    problems = fill_value_problems(resolved, value, ("fill_value",))
    assert len(problems) != 0
    assert problems[0].loc == ("fill_value", *at)


def test_every_family_has_an_example() -> None:
    """Every definition in `DATA_TYPES_V2` has an example above, so no definition goes untested."""
    names = {definition.name for definition in DATA_TYPES_V2}
    assert names == {
        "bool",
        "int",
        "uint",
        "float",
        "complex",
        "bytes",
        "str",
        "void",
        "datetime64",
        "timedelta64",
        "object",
        "struct",
    }


def test_a_struct_record_type_sits_where_the_document_writes_it() -> None:
    """`fields_of` places a record's type under the struct's own configuration location, `("dtype", "fields", 0, 1)`, not under a `configuration` key a v2 document does not have."""
    resolved, _ = resolve(
        [["a", "<i0"], ["b", "<f4"]], ZarrV2DataTypeDefinition, CONTEXT, ("dtype",)
    )
    assert [loc for loc, _ in fields_of(resolved, ("dtype",))] == [
        ("dtype",),
        ("dtype", "fields", 0, 1),
        ("dtype", "fields", 1, 1),
    ]


def test_a_float_fill_value_past_the_largest_of_its_width_reads_as_an_infinity() -> None:
    """A float fill value that is too large for a float of the dtype's width (`<f4` past float32, `<f2` past float16, written as an integer or a float) becomes the infinity of the same sign. This matches how the v3 float types read it and how NumPy stores it. It applies to a float and to each component of a complex. `read_array_metadata_v2` reads it without problems, and the model equals one written with `"Infinity"`."""
    from zarr_metadata.model import ZarrV2ArrayMetadata, read_array_metadata_v2

    cases: list[tuple[str, JSONValue, JSONValue]] = [
        ("<f8", 10**400, "Infinity"),
        ("<f8", -(10**400), "-Infinity"),
        ("<f4", 10**39, "Infinity"),
        ("<f4", 1e39, "Infinity"),
        ("<f4", -(2**128), "-Infinity"),
        ("<f2", 70000, "Infinity"),
        ("<f2", 10**309, "Infinity"),
        ("<c16", [10**400, 0], ["Infinity", 0.0]),
        ("<c8", [10**39, -1e39], ["Infinity", "-Infinity"]),
    ]
    for dtype, huge, named in cases:
        written = ZarrV2ArrayMetadata.create_default(dtype=dtype, fill_value=huge)
        assert written == ZarrV2ArrayMetadata.create_default(dtype=dtype, fill_value=named)
        assert read_array_metadata_v2(written.to_json()).problems == ()


@pytest.mark.parametrize(
    ("dtype", "fill_value", "accepted"),
    [
        ("|V4", "AQIDBA==", True),
        ("|V4", "AQ==", False),
        ("|S5", "AQID", True),
        ("|S5", "AQIDBAUG", False),
        ([["a", "<i4"], ["b", "<f8"]], "AAAAAAAAAAAAAAAA", True),
        ([["a", "<i4"], ["b", "<f8"]], "AA==", False),
        ([["a", "<i4", [2]], ["b", "<M8[ns]"]], "AAAAAAAAAAAAAAAAAAAAAA==", True),
        ([["a", "<i4", [2]], ["b", "<M8[ns]"]], "AAAAAAAAAAAAAAAA", False),
        ([["a", [["x", "<i2"], ["y", "<i2"]]], ["b", "|V1"]], "AAAAAAA=", True),
        ([["a", "|O"]], "AA==", True),
    ],
    ids=[
        "void",
        "void-short",
        "bytes",
        "bytes-long",
        "struct",
        "struct-short",
        "subarray",
        "subarray-short",
        "nested-struct",
        "object-field-unknown",
    ],
)
def test_a_base64_fill_value_is_as_long_as_its_item(
    dtype: JSONValue, fill_value: str, accepted: bool
) -> None:
    """A void or struct fill value must decode to exactly the item size. A record's size is the sum of its fields' sizes, each multiplied by the product of its subarray shape, including nested structs. A byte string's fill value must decode to at most the item size, because NumPy pads a shorter one. A record with a field of no fixed size (`|O`) is not checked. zarr-python 3 fails to open an array whose fill value has a different size."""
    from zarr_metadata.model import ZarrV2ArrayMetadata, validate_array_metadata_v2

    document = {
        **ZarrV2ArrayMetadata.create_default(shape=(4,), chunks=(2,)).to_json(),
        "dtype": dtype,
        "fill_value": fill_value,
    }
    found = validate_array_metadata_v2(document)
    assert (len(found) == 0) is accepted, [p.message for p in found]
    if not accepted:
        assert [(p.loc, p.kind) for p in found] == [(("fill_value",), "invalid_value")]
