"""A v2 array document read once: each field as the scope read it, every problem, and the model."""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata._sentinel import UNSET
from zarr_metadata.model import (
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
    Context,
    Read,
    Refused,
    Unclaimed,
    ZarrV2CodecDefinition,
)
from zarr_metadata.v3.definition import (
    EmptyConfiguration,
)

Loc = tuple[str | int, ...]
BASE: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
PRIVATE = Context.of(FLOAT_V2, ZLIB_V2)


@pytest.mark.parametrize(
    ("changes", "context", "dtype", "compressor", "filters", "locs"),
    [
        ({}, None, Read, None, None, [("dtype",)]),
        (
            {
                "dtype": "<f4",
                "compressor": {"id": "zlib", "level": 1},
                "filters": [{"id": "delta", "dtype": "<f4"}],
            },
            None,
            Read,
            Read,
            (Read,),
            [("dtype",), ("compressor",), ("filters", 0)],
        ),
        (
            {"dtype": "<e2", "compressor": {"id": "x"}},
            None,
            Unclaimed,
            Unclaimed,
            None,
            [("dtype",), ("compressor",)],
        ),
        ({"dtype": "|u1"}, PRIVATE, Unclaimed, None, None, [("dtype",)]),
        (
            {"dtype": [["a", "<f4"]], "filters": [], "fill_value": None},
            None,
            Read,
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
    """`read_array_metadata_v2` reads the document once in the scope given (`CORE_V2` by default): `dtype`, `compressor` and `filters` are each `Read`, `Unclaimed` or None as written, `fields()` walks them in document order with a struct's record types after it, and a document with no problem has none."""
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
    """A reading of a document with a problem holds every problem `validate_array_metadata_v2` finds, a `Refused` field where one was refused, and no model."""
    reading = read_array_metadata_v2(value)
    assert [(p.loc, p.kind) for p in reading.problems] == problems
    assert reading.metadata is None
    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(value)] == problems
    if isinstance(value, dict) and "dtype" in problems[0][0]:
        assert isinstance(reading.dtype, Refused)


def test_the_v2_readers_read_in_the_scope_given() -> None:
    """`validate_array_metadata_v2`, `is_array_metadata_v2` and `parse_array_metadata_v2` take a scope: in one that does not claim `|u1`, the default document still validates (an unclaimed dtype is left unjudged), and in one where a private `zlib` takes no level, a written level is a problem."""
    assert validate_array_metadata_v2(BASE, context=PRIVATE) == ()
    assert is_array_metadata_v2(BASE, context=PRIVATE)
    assert parse_array_metadata_v2(BASE, context=PRIVATE)["dtype"] == "|u1"
    bare = ZarrV2CodecDefinition(name="zlib", configuration=EmptyConfiguration)
    doc = {**BASE, "compressor": {"id": "zlib", "level": 1}}
    assert validate_array_metadata_v2(doc) == ()
    found = validate_array_metadata_v2(doc, context=CORE_V2.extended_with(bare))
    assert [p.loc for p in found] == [("compressor", "level")]
