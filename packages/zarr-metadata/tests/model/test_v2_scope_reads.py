"""The v2 array validator reads its dtype, codecs and fill value in the v2 scope."""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ArrayMetadata,
    validate_array_metadata_v2,
)

Loc = tuple[str | int, ...]
BASE: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default().to_json())


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "<f4", "fill_value": "NaN"},
        {"dtype": [["x", "<f4"], ["y", "<i4", [2]]], "fill_value": "AAAAAAAAAAA="},
        {"dtype": "<M8[ns]", "fill_value": "NaT"},
        {"dtype": "<e2", "fill_value": "anything"},
        {"compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}},
        {"compressor": {"id": "categorize", "labels": ["a"]}},
        {
            "filters": [
                {"id": "delta", "dtype": "<f8"},
                {"id": "quantize", "digits": 2, "dtype": "<f8"},
            ]
        },
        {"filters": []},
        {"fill_value": None},
    ],
    ids=[
        "float-nan",
        "struct",
        "datetime",
        "unknown-code",
        "blosc",
        "unknown-id",
        "filters",
        "no-filters",
        "null",
    ],
)
def test_a_v2_document_the_scope_reads_has_no_problem(changes: dict[str, Any]) -> None:
    """A dtype the scope reads with a fill value its family takes, a compressor and filters the scope reads or leaves unclaimed, and a null fill value, each validate with no problem."""
    assert validate_array_metadata_v2({**BASE, **changes}) == ()


@pytest.mark.parametrize(
    ("changes", "at", "kind"),
    [
        ({"dtype": "float32"}, ("dtype",), "invalid_value"),
        ({"dtype": "<i3"}, ("dtype",), "invalid_value"),
        ({"dtype": [["x", "<f4"], ["x", "<f4"]]}, ("dtype", "fields", 1, 0), "invalid_value"),
        ({"dtype": 3}, ("dtype",), "invalid_type"),
        ({"dtype": "<i4", "fill_value": "NaN"}, ("fill_value",), "invalid_type"),
        ({"dtype": "<u1", "fill_value": 256}, ("fill_value",), "invalid_value"),
        ({"dtype": "|S2", "fill_value": "no!"}, ("fill_value",), "invalid_value"),
        ({"compressor": {"id": "zlib", "level": 10}}, ("compressor", "level"), "invalid_value"),
        ({"compressor": {"level": 1}}, ("compressor", "id"), "missing_key"),
        ({"compressor": "zlib"}, ("compressor",), "invalid_type"),
        ({"filters": [{"id": "delta"}]}, ("filters", 0, "dtype"), "missing_key"),
        ({"filters": {"id": "delta", "dtype": "<f8"}}, ("filters",), "invalid_type"),
    ],
    ids=[
        "dtype-name",
        "dtype-size",
        "struct-duplicate",
        "dtype-type",
        "fill-nan-int",
        "fill-range",
        "fill-base64",
        "codec-range",
        "codec-no-id",
        "codec-string",
        "filter-missing",
        "filters-not-array",
    ],
)
def test_error_what_the_scope_refuses_is_a_problem_of_the_document(
    changes: dict[str, Any], at: Loc, kind: str
) -> None:
    """A dtype, fill value, compressor or filter the scope refuses is one problem of the document, at the field or the parameter that is wrong; before, the string content of a dtype and the parameters of a codec went unjudged."""
    problems = validate_array_metadata_v2({**BASE, **changes})
    assert [(p.loc, p.kind) for p in problems] == [(at, kind)]


def test_create_default_with_a_dtype_takes_no_fill_value() -> None:
    """`create_default` given a dtype and no fill value keeps `0` when the family takes it, and takes `null` otherwise; a fill value given is kept."""
    assert ZarrV2ArrayMetadata.create_default(dtype="|b1").fill_value is None
    assert ZarrV2ArrayMetadata.create_default(dtype="<f8").fill_value == 0
    assert ZarrV2ArrayMetadata.create_default(dtype="<i4").fill_value == 0
    assert ZarrV2ArrayMetadata.create_default(dtype="|S3").fill_value is None
    assert ZarrV2ArrayMetadata.create_default(dtype="<f4", fill_value=1.5).fill_value == 1.5
    assert ZarrV2ArrayMetadata.create_default().fill_value == 0


def test_a_v2_model_refuses_a_dtype_the_scope_refuses() -> None:
    """`ZarrV2ArrayMetadata` checks itself when built, so a dtype the scope refuses raises as any other problem does."""
    with pytest.raises(MetadataValidationError, match="typestr"):
        ZarrV2ArrayMetadata.create_default(dtype="float32")
