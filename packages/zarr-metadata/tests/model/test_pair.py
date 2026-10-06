"""A v3 model is a document and the scope it was read in: the pair, and what follows from it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from zarr_metadata._json import refine_json
from zarr_metadata.model import (
    UNSET,
    MetadataValidationError,
    ZarrV3ArrayMetadata,
    read_array_metadata_v3,
)
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, Context, Read, Unclaimed

ARRAY: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
    "chunk_key_encoding": {"name": "default"},
    "codecs": ["bytes"],
}
SPELLED_OUT: dict[str, Any] = {
    **ARRAY,
    "data_type": {"name": "uint8"},
    "codecs": [{"name": "bytes", "configuration": {}}],
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
}


@pytest.mark.parametrize(
    ("document", "context", "expected_context"),
    [
        (ARRAY, None, CORE_AND_EXTENSIONS),
        (ARRAY, CORE, CORE),
        (SPELLED_OUT, Context.of(), Context.of()),
    ],
    ids=["default-scope", "core", "empty-scope"],
)
def test_a_model_is_its_document_read_in_its_scope(
    document: dict[str, Any], context: Context | None, expected_context: Context
) -> None:
    """A model built from a document holds that document, as written and refined, and the scope it was read in, `CORE_AND_EXTENSIONS` when none is given."""
    model = ZarrV3ArrayMetadata(document, context=context)
    assert model.context == expected_context
    assert model.to_json() == refine_json(document)[0]


@pytest.mark.parametrize(
    ("document", "key", "written"),
    [
        (ARRAY, "data_type", "uint8"),
        (SPELLED_OUT, "data_type", {"name": "uint8"}),
        (ARRAY, "codecs", ("bytes",)),
        (SPELLED_OUT, "codecs", ({"name": "bytes", "configuration": {}},)),
    ],
    ids=["bare-data-type", "object-data-type", "bare-codec", "object-codec"],
)
def test_to_json_keeps_the_spelling_the_document_was_written_in(
    document: dict[str, Any], key: str, written: object
) -> None:
    """`to_json` writes each field as the document wrote it, not as a reader would respell it: the model is the document."""
    assert ZarrV3ArrayMetadata(document).to_json()[key] == written


def test_properties_are_what_the_reading_holds() -> None:
    """The typed members -- fields as the scope read them, shape, fill value, attributes -- are views of the reading, read-only."""
    model = ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": [1]}, "acme": 1})
    assert isinstance(model.data_type, Read)
    assert isinstance(model.codecs[0], Read)
    assert model.shape == (4,)
    assert model.fill_value == 0
    assert model.dimension_names is UNSET
    assert isinstance(model.attributes, Mapping)
    assert model.attributes == {"a": (1,)}
    assert model.extra_fields == {"acme": 1}
    assert (model.zarr_format, model.node_type) == (3, "array")
    with pytest.raises(TypeError):
        model.attributes["b"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        model.shape = (5,)  # type: ignore[misc]
    assert isinstance(ZarrV3ArrayMetadata(ARRAY, context=Context.of()).data_type, Unclaimed)


def test_a_reading_without_problems_builds_the_model_without_reading_again() -> None:
    """`read_array_metadata_v3` hands its reading to the model it builds, so the model's `reading` is that reading, not a second one."""
    reading = read_array_metadata_v3(ARRAY)
    assert reading.metadata is not None
    assert reading.metadata.reading.pipeline is reading.pipeline


def test_error_a_document_with_a_problem_is_refused_at_construction() -> None:
    """The constructor raises `MetadataValidationError` with every problem, as `from_json` does."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata({**ARRAY, "fill_value": 300, "shape": [-1]})
    assert sorted(problem.loc for problem in raised.value.problems) == [
        ("fill_value",),
        ("shape",),
    ]
