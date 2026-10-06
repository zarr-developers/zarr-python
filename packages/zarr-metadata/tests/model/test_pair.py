"""A v3 model is a document and the scope it was read in: the pair, and what follows from it."""

from __future__ import annotations

import pickle
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
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    Context,
    Read,
    Unclaimed,
)

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
MY_BYTES = CodecDefinition(name="bytes", configuration=Empty, kind="array_bytes", size="static")
"""A private `bytes`: another meaning under the core name, which takes no configuration."""
PRIVATE = CORE.extended_with(MY_BYTES)
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


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(SPELLED_OUT), True),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), ZarrV3ArrayMetadata(ARRAY), True),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=PRIVATE), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=Context.of()), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata({**ARRAY, "attributes": {}}), True),
    ],
    ids=["spellings", "unused-definitions", "private-bytes", "unclaimed", "empty-attributes"],
)
def test_models_are_equal_by_what_their_documents_mean(
    left: ZarrV3ArrayMetadata, right: ZarrV3ArrayMetadata, equal: bool
) -> None:
    """Two models are equal when every field reads the same by the same definition and the rest is the same JSON: spelling and unused definitions do not matter, a private definition under a core name does, and so does a name read against one left unclaimed."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


@pytest.mark.parametrize(
    "context", [None, CORE, PRIVATE, Context.of()], ids=["default", "core", "private", "empty"]
)
def test_a_model_round_trips_through_its_document_in_its_scope(context: Context | None) -> None:
    """`from_json(m.to_json(), context=m.context) == m`: the document and the scope determine the model, and pickle carries both."""
    model = ZarrV3ArrayMetadata(ARRAY, context=context)
    assert ZarrV3ArrayMetadata(model.to_json(), context=model.context) == model
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded == model
    assert loaded.context == model.context
    assert loaded.to_json() == model.to_json()


def test_error_a_model_whose_scope_does_not_pickle_says_so() -> None:
    """A scope holding a definition with a local function does not pickle, and the model raises the error pickling a `Context` raises."""
    local = CodecDefinition(
        name="acme.c",
        configuration=Empty,
        kind="bytes_bytes",
        size="dynamic",
        rules=lambda configuration, nested: iter(()),
    )
    model = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ["bytes", "acme.c"]}, context=CORE.extended_with(local)
    )
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(model)
