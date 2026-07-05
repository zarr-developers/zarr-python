"""Executable example: integrating the metadata models with pydantic (v2).

Pydantic's native dataclass introspection CAN be made to work (see
`test_native_dataclass_introspection_is_possible_but_diverges`): the models
keep their annotation-only imports behind `TYPE_CHECKING`, so a bare
`TypeAdapter(ArrayMetadataModelV3)` raises `class-not-fully-defined`, but
`rebuild(_types_namespace=...)` with the names supplied resolves the schema.
It is still the wrong tool: it validates the MODEL SHAPE, not the DOCUMENT —
no `from_json` normalization (a bare-string `data_type` is rejected), and
pydantic's lax coercion silently re-opens holes the library's validators
close (`shape=[True, -5]` coerces to `(1, -5)`; a wrong `dimension_names`
count passes). The recommended integration delegates wholesale — treat the
model as an opaque value:

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


# --- the road not taken: native dataclass introspection ----------------------


def test_native_dataclass_introspection_is_possible_but_diverges() -> None:
    """Pydantic CAN introspect the model dataclass after a namespace rebuild,
    but that path validates the model shape, not the document: it rejects the
    document form, coerces booleans into dimensions, and skips the library's
    cross-field checks. This test documents why the delegation pattern above
    is the recommended integration."""
    from zarr_metadata._common import JSONValue
    from zarr_metadata.model import NamedConfigModelV3
    from zarr_metadata.v3._common import MetadataV3
    from zarr_metadata.v3.array import ArrayMetadataV3, ExtensionFieldV3

    adapter = TypeAdapter(ArrayMetadataModelV3)
    adapter.rebuild(
        force=True,
        _types_namespace={
            "JSONValue": JSONValue,
            "ExtensionFieldV3": ExtensionFieldV3,
            "MetadataV3": MetadataV3,
            "ArrayMetadataV3": ArrayMetadataV3,
            "NamedConfigModelV3": NamedConfigModelV3,
            "MetadataFieldModelV3": NamedConfigModelV3,
        },
    )
    model_shaped = {
        "shape": [10],
        "fill_value": 0,
        "data_type": {"name": "uint8", "configuration": {}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "codecs": [{"name": "bytes", "configuration": {}}],
        "chunk_key_encoding": {"name": "default", "configuration": {}},
        "dimension_names": None,
        "attributes": {},
        "storage_transformers": [],
        "extra_fields": {},
    }
    # model-shaped data validates, nested named configs and all
    model = adapter.validate_python(model_shaped)
    assert isinstance(model.data_type, NamedConfigModelV3)

    # divergence 1: the DOCUMENT form is rejected — no from_json normalization
    with pytest.raises(ValidationError):
        adapter.validate_python(model_shaped | {"data_type": "uint8"})

    # divergence 2: lax coercion re-opens holes the library validators close
    coerced = adapter.validate_python(model_shaped | {"shape": [True, -5]})
    assert coerced.shape == (1, -5)  # from_json would reject both entries

    # __post_init__ invariants DO still run under pydantic construction
    with pytest.raises(ValidationError, match="Extra fields"):
        adapter.validate_python(
            model_shaped | {"extra_fields": {"shape": {"must_understand": False}}}
        )
