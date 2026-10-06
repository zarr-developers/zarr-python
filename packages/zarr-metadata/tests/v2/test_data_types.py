"""Every v2 data type, read as its family: which typestrs it takes, and which fill values."""

from __future__ import annotations

import pytest

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type import V2_DATA_TYPES
from zarr_metadata.v3.definition import (
    Context,
    Read,
    Refused,
    Unclaimed,
    canonical_fill_value,
    canonical_of,
    fields_of,
    fill_value_problems,
    resolve,
)

SCOPE = Context.of(*V2_DATA_TYPES)

Loc = tuple[str | int, ...]


def _read(value: object) -> tuple[type, list[tuple[Loc, str]]]:
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, SCOPE, ("dtype",))
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
        # NumPy names the padding of an aligned struct "", and zarr 2.x writes it.
        (
            [["a", "|i1"], ["", "|V3"], ["b", "<i4"], ["", "|V3"]],
            "struct",
            (("a", "|i1"), ("", "|V3"), ("b", "<i4"), ("", "|V3")),
        ),
    ],
)
def test_every_family_reads_its_typestrs(value: object, family: str, canonical: object) -> None:
    """Each typestr of a family reads by the family's definition, and its simplest spelling is the typestr NumPy writes: `|` for a type of one byte or no byte order, `us` for a microsecond unit; a records array reads as `struct`, each record's type read too."""
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, SCOPE)
    assert isinstance(resolved, Read)
    assert resolved.definition.name == family
    assert problems == ()
    assert canonical_of(resolved, problems) == canonical


@pytest.mark.parametrize("value", ["<e2", "<T16", "|a1"])
def test_a_type_code_the_spec_does_not_list_is_unclaimed(value: str) -> None:
    """A typestr whose type code the v2 spec does not list is filed under itself, which nothing in scope claims: read as `Unclaimed`, with no problem, and written back as it was."""
    resolved, problems = resolve(value, ZarrV2DataTypeDefinition, SCOPE)
    assert isinstance(resolved, Unclaimed)
    assert problems == ()
    assert resolved.to_json() == value


@pytest.mark.parametrize(
    ("value", "at"),
    [
        ("float32", ("dtype",)),
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
    """A typestr of a size, byte order or unit its family does not take is refused, the problem at the field; a struct's problems sit under `fields`, at the record and its position, and a record's type that is refused is reported there while the struct still reads."""
    kind, problems = _read(value)
    assert kind is Refused or "fields" in at
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
    """Each family takes the fill values the v2 spec gives it, and `null` always: booleans, integers in the type's range, numbers or the named non-finite strings for floats and each complex component, base64 for bytes, void and struct, a string for str, any JSON for object, ticks or `NaT` for times. The canonical spelling makes an integer written for a float a float, and `NaT` of its ticks."""
    resolved, _ = resolve(dtype, ZarrV2DataTypeDefinition, SCOPE)
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
    """A fill value outside the shape or range its family takes is reported, at the value or at the component that is wrong: a v2 float takes no hex string, and a struct's fill value is base64 of the record, not an object of fields."""
    resolved, _ = resolve(dtype, ZarrV2DataTypeDefinition, SCOPE)
    problems = fill_value_problems(resolved, value, ("fill_value",))
    assert len(problems) != 0
    assert problems[0].loc == ("fill_value", *at)


def test_every_family_has_an_example() -> None:
    """Every definition in `V2_DATA_TYPES` is exercised above, so none is filed untested."""
    names = {definition.name for definition in V2_DATA_TYPES}
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
    resolved, _ = resolve([["a", "<i0"], ["b", "<f4"]], ZarrV2DataTypeDefinition, SCOPE, ("dtype",))
    assert [loc for loc, _ in fields_of(resolved, ("dtype",))] == [
        ("dtype",),
        ("dtype", "fields", 0, 1),
        ("dtype", "fields", 1, 1),
    ]
