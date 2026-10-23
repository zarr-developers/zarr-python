"""Tests for reading a v2 array document once. A reading holds each field as the scope read it, every problem, and the model."""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition import (
    AcceptedField,
    Context,
    EmptyConfiguration,
    RefusedField,
    UnclaimedField,
)
from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ArrayMetadata,
    is_array_metadata_v2,
    parse_array_metadata_v2,
    read_array_metadata_v2,
    validate_array_metadata_v2,
)
from zarr_metadata.v2.codec.compression import ZLIB_V2
from zarr_metadata.v2.data_type.scalar import FLOAT_V2
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2CodecDefinition,
)

Loc = tuple[str | int, ...]
BASE: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
PRIVATE = Context.of(FLOAT_V2, ZLIB_V2)


@pytest.mark.parametrize(
    ("changes", "context", "dtype", "compressor", "filters", "locs"),
    [
        ({}, None, AcceptedField, None, None, [("dtype",)]),
        (
            {
                "dtype": "<f4",
                "compressor": {"id": "zlib", "level": 1},
                "filters": [{"id": "delta", "dtype": "<f4"}],
            },
            None,
            AcceptedField,
            AcceptedField,
            (AcceptedField,),
            [("dtype",), ("compressor",), ("filters", 0)],
        ),
        (
            {"dtype": "<e2", "compressor": {"id": "x"}},
            None,
            UnclaimedField,
            UnclaimedField,
            None,
            [("dtype",), ("compressor",)],
        ),
        ({"dtype": "|u1"}, PRIVATE, UnclaimedField, None, None, [("dtype",)]),
        (
            {"dtype": [["a", "<f4"]], "filters": [], "fill_value": None},
            None,
            AcceptedField,
            None,
            (),
            [("dtype",), ("dtype", "fields", 0, 1)],
        ),
    ],
    ids=["default", "all-read", "unclaimed", "private-scope", "struct"],
)
def test_a_reading_holds_each_field_as_the_scope_read_it(
    changes: dict[str, Any],
    context: Context | None,
    dtype: type,
    compressor: object,
    filters: object,
    locs: list[Loc],
) -> None:
    """`read_array_metadata_v2` reads the document once in the given scope (`CORE_V2` by default). Each of `dtype`, `compressor` and `filters` is an `AcceptedField`, an `UnclaimedField`, or None, depending on what the document contains. `fields()` returns them in document order, with a struct's record types after the struct. A valid document produces no problems."""
    reading = read_array_metadata_v2({**BASE, **changes}, context=context)
    assert reading.problems == ()
    assert type(reading.dtype) is dtype
    assert (None if reading.compressor is None else type(reading.compressor)) == compressor
    assert reading.filters is not UNSET
    found = None if reading.filters is None else tuple(type(f) for f in reading.filters)
    assert found == filters
    assert [loc for loc, _ in reading.fields()] == locs


@pytest.mark.parametrize(
    ("value", "problems"),
    [
        ({**BASE, "dtype": "float32"}, [(("dtype",), "invalid_value")]),
        (
            {**BASE, "compressor": {"id": "zlib", "level": 10}},
            [(("compressor", "level"), "invalid_value")],
        ),
        ({**BASE, "fill_value": "NaN"}, [(("fill_value",), "invalid_type")]),
        (3, [((), "invalid_type")]),
    ],
    ids=["dtype", "compressor", "fill", "not-an-object"],
)
def test_error_a_reading_reports_what_the_validator_reports(
    value: object, problems: list[tuple[Loc, str]]
) -> None:
    """Reading an invalid document reports every problem that `validate_array_metadata_v2` finds. A refused field is held as a `RefusedField`, and the reading has no model."""
    reading = read_array_metadata_v2(value)
    assert [(p.loc, p.kind) for p in reading.problems] == problems
    assert reading.metadata is None
    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(value)] == problems
    if isinstance(value, dict) and "dtype" in problems[0][0]:
        assert isinstance(reading.dtype, RefusedField)


def test_the_v2_readers_read_in_the_scope_given() -> None:
    """`validate_array_metadata_v2`, `is_array_metadata_v2` and `parse_array_metadata_v2` take a scope. In a scope that does not claim `|u1`, the default document still validates, because an unclaimed dtype is not checked. In a scope where a private `zlib` definition takes no level, a `level` in the document is a problem."""
    assert validate_array_metadata_v2(BASE, context=PRIVATE) == ()
    assert is_array_metadata_v2(BASE, context=PRIVATE)
    assert parse_array_metadata_v2(BASE, context=PRIVATE)["dtype"] == "|u1"
    bare = ZarrV2CodecDefinition(name="zlib", configuration=EmptyConfiguration)
    doc = {**BASE, "compressor": {"id": "zlib", "level": 1}}
    assert validate_array_metadata_v2(doc) == ()
    found = validate_array_metadata_v2(doc, context=CORE_V2.extended_with(bare))
    assert [p.loc for p in found] == [("compressor", "level")]


class _EqualTo:
    """A value that compares equal to a string without being one."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return other == self.text

    __hash__ = None  # pyright: ignore[reportAssignmentType]


@pytest.mark.parametrize(
    ("key", "value"),
    [("order", _EqualTo("C")), ("dimension_separator", _EqualTo("."))],
    ids=["order", "dimension_separator"],
)
def test_error_a_v2_member_that_only_compares_equal_to_an_allowed_value_is_refused(
    key: str, value: object
) -> None:
    """`order` and `dimension_separator` are checked after refinement. A value that is not a string, but compares equal to an allowed string, is outside the allowed set. It is reported as the wrong type, and no model accepts it."""
    document = {**BASE, key: value}
    problems = validate_array_metadata_v2(document)
    assert [(found.loc, found.kind) for found in problems] == [((key,), "invalid_type")]
    assert read_array_metadata_v2(document).metadata is None
    with pytest.raises(MetadataValidationError):
        ZarrV2ArrayMetadata(document)
