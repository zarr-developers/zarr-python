"""Executable example: integrating the metadata models with pydantic (v2).

Do NOT hand the dataclass to pydantic for field-by-field validation: the
models keep their annotation-only imports behind `TYPE_CHECKING` (so the
string annotations are unresolvable at runtime, and pydantic raises
`class-not-fully-defined`), and even if they resolved, pydantic's coercion
rules would diverge from the library's structural validation. Delegate
wholesale instead — treat the model as an opaque value:

- `InstanceOf` makes pydantic's core schema an is-instance check (no field
  introspection),
- validation goes through `from_json` (the single source of truth for what
  a well-formed document is, including normalization: bare-string metadata
  fields, arrays-to-tuples),
- serialization goes through `to_json` (the canonical document form).

`MetadataValidationError` subclasses `ValueError`, so pydantic converts a
failed parse into its own `ValidationError` with the loc-annotated problem
messages intact.
"""

from typing import Annotated

import pytest
from pydantic import (
    BaseModel,
    BeforeValidator,
    InstanceOf,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
)

from zarr_metadata.model import ArrayMetadataModelV3

# --- the integration (this is the example) -----------------------------------


def _as_array_metadata_v3(value: object) -> ArrayMetadataModelV3:
    """Accept an existing model instance or a raw metadata document."""
    if isinstance(value, ArrayMetadataModelV3):
        return value
    return ArrayMetadataModelV3.from_json(value)


# return_type is explicit because to_json's own annotation (`ArrayMetadataV3`)
# is a TYPE_CHECKING-only name pydantic cannot resolve at runtime.
ArrayMetadataV3Field = Annotated[
    InstanceOf[ArrayMetadataModelV3],
    BeforeValidator(_as_array_metadata_v3),
    PlainSerializer(ArrayMetadataModelV3.to_json, return_type=dict),
]
"""A pydantic-ready field type for v3 array metadata.

Validates raw documents via `from_json`, passes model instances through,
and serializes to the canonical document form via `to_json`.
"""


class ArrayManifest(BaseModel):
    """Example consumer model: a named array with its metadata document."""

    path: str
    metadata: ArrayMetadataV3Field


# --- tests pinning the example ------------------------------------------------

VALID_DOC = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [10],
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
    "chunk_key_encoding": {"name": "default"},
    "codecs": [{"name": "bytes"}],
}


def test_raw_document_is_validated_into_a_model() -> None:
    """A raw metadata document on a pydantic field is parsed by from_json,
    with the library's normalization applied (tuples, canonical field form)."""
    manifest = ArrayManifest(path="a/b", metadata=VALID_DOC)  # type: ignore[arg-type]
    assert isinstance(manifest.metadata, ArrayMetadataModelV3)
    assert manifest.metadata.shape == (10,)
    assert manifest.metadata.data_type.name == "uint8"


def test_model_instance_passes_through() -> None:
    """An already-constructed model instance is accepted unchanged."""
    model = ArrayMetadataModelV3.from_json(VALID_DOC)
    manifest = ArrayManifest(path="a/b", metadata=model)
    assert manifest.metadata is model


def test_invalid_document_surfaces_problems_in_validation_error() -> None:
    """A structurally-invalid document fails pydantic validation, carrying the
    loc-annotated problem messages from MetadataValidationError."""
    doc = dict(VALID_DOC)
    del doc["chunk_key_encoding"]
    with pytest.raises(ValidationError) as exc_info:
        ArrayManifest(path="a/b", metadata=doc)  # type: ignore[arg-type]
    assert "chunk_key_encoding: missing required key" in str(exc_info.value)


def test_dump_emits_canonical_document() -> None:
    """model_dump serializes the field via to_json — the canonical document,
    not pydantic's field-by-field view of the dataclass."""
    manifest = ArrayManifest(path="a/b", metadata=VALID_DOC)  # type: ignore[arg-type]
    dumped = manifest.model_dump()
    assert dumped["metadata"] == manifest.metadata.to_json()
    # the bare-string data_type was normalized to the canonical object form
    assert dumped["metadata"]["data_type"] == {"name": "uint8", "configuration": {}}


def test_json_roundtrip_through_pydantic() -> None:
    """model_dump_json output re-validates to an equal manifest (JSON emits
    tuples as arrays; from_json converts them back)."""
    manifest = ArrayManifest(path="a/b", metadata=VALID_DOC)  # type: ignore[arg-type]
    revived = ArrayManifest.model_validate_json(manifest.model_dump_json())
    assert revived == manifest


def test_type_adapter_standalone() -> None:
    """The annotated alias also works without a BaseModel, via TypeAdapter."""
    adapter = TypeAdapter(ArrayMetadataV3Field)
    model = adapter.validate_python(VALID_DOC)
    assert isinstance(model, ArrayMetadataModelV3)
    assert adapter.dump_python(model) == model.to_json()
