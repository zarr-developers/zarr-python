"""Tests for the whole-document validators in `zarr_metadata.rules`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.model import MetadataValidationError
from zarr_metadata.model import (
    is_array_metadata_v3 as model_is_array_metadata_v3,
)
from zarr_metadata.rules import (
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from zarr_metadata import ZarrV2ArrayMetadataJSON, ZarrV3ArrayMetadataJSON
    from zarr_metadata.model import ValidationProblem

    # The validators are uniform in their inputs (any object) and differ
    # only in the document type they hand back, which these tests never
    # depend on.
    Validator = Callable[[object], tuple[ValidationProblem, ...]]
    Parser = Callable[[object], Mapping[str, object]]

V3_ARRAY: ZarrV3ArrayMetadataJSON = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}

V2_ARRAY: ZarrV2ArrayMetadataJSON = {
    "zarr_format": 2,
    "shape": (4,),
    "chunks": (2,),
    "dtype": "<i4",
    "compressor": None,
    "fill_value": 0,
    "order": "C",
    "filters": None,
}

# (validate, parse, document) — every entry must validate cleanly through
# both entry points; list-spelled arrays check that parse normalizes.
# Error paths get their own tests below.
CASES: dict[str, tuple[Validator, Parser, Mapping[str, object]]] = {
    "v3-array": (
        validate_array_metadata_v3,
        parse_array_metadata_v3,
        V3_ARRAY,
    ),
    "v3-array-list-spelled": (
        validate_array_metadata_v3,
        parse_array_metadata_v3,
        {**V3_ARRAY, "shape": [4, 4], "codecs": ["bytes"]},
    ),
    "v2-array": (
        validate_array_metadata_v2,
        parse_array_metadata_v2,
        V2_ARRAY,
    ),
}


@pytest.mark.parametrize(("validate", "parse", "doc"), CASES.values(), ids=list(CASES))
def test_valid_documents(validate: Validator, parse: Parser, doc: Mapping[str, object]) -> None:
    parsed = parse(doc)
    assert validate(parsed) == ()
    shape = doc["shape"]
    assert isinstance(shape, (list, tuple))
    assert parsed["shape"] == tuple(shape)


def test_error_v3_combined_report() -> None:
    # One raise carrying problems from both passes: a structural problem
    # (bad node_type) and a composition problem (fill_value vs data_type).
    with pytest.raises(MetadataValidationError) as info:
        parse_array_metadata_v3({**V3_ARRAY, "node_type": "grid", "fill_value": 300})
    kinds = {(p.loc, p.kind) for p in info.value.problems}
    assert (("node_type",), "invalid_value") in kinds
    assert (("fill_value",), "invalid_value") in kinds


def test_error_v3_dimension_names_reported_once() -> None:
    # Regression: this fault used to be reported twice — once by the
    # structural validator, once by the composition rule. The check now
    # has one owner.
    problems = validate_array_metadata_v3({**V3_ARRAY, "dimension_names": ("x",)})
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_names",), "invalid_value")]


def test_error_v2_chunks_rank() -> None:
    problems = validate_array_metadata_v2({**V2_ARRAY, "chunks": (2, 2)})
    assert [(p.loc, p.kind) for p in problems] == [(("chunks",), "invalid_value")]


def test_error_v2_parse_raises() -> None:
    with pytest.raises(MetadataValidationError, match="same number of dimensions"):
        parse_array_metadata_v2({**V2_ARRAY, "chunks": (2, 2)})


def test_composition_invalid_document_still_satisfies_the_typeddict() -> None:
    # The model layer's TypeIs narrows a composition-invalid document,
    # which is why the rules layer offers no guard of its own: a
    # fill_value out of range does not stop the value being an instance
    # of ZarrV3ArrayMetadataJSON.
    doc = {**V3_ARRAY, "fill_value": 300}
    assert model_is_array_metadata_v3(doc) is True
    assert validate_array_metadata_v3(doc) != ()
