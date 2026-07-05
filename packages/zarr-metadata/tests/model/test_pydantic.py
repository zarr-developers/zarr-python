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

from collections.abc import Mapping
from typing import Annotated, Generic, TypeVar

import pytest
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    InstanceOf,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from zarr_metadata import JSONValue
from zarr_metadata.model import ArrayMetadataModelV3, NamedConfigModelV3

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
    manifest = ArrayManifest.model_validate({"path": "a/b", "metadata": VALID_DOC})
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
        ArrayManifest.model_validate({"path": "a/b", "metadata": doc})
    assert "chunk_key_encoding: missing required key" in str(exc_info.value)


def test_dump_emits_canonical_document() -> None:
    """model_dump serializes the field via to_json — the canonical document,
    not pydantic's field-by-field view of the dataclass."""
    manifest = ArrayManifest.model_validate({"path": "a/b", "metadata": VALID_DOC})
    dumped = manifest.model_dump()
    assert dumped["metadata"] == manifest.metadata.to_json()
    # the bare-string data_type was normalized to the canonical object form
    assert dumped["metadata"]["data_type"] == {"name": "uint8", "configuration": {}}


def test_json_roundtrip_through_pydantic() -> None:
    """model_dump_json output re-validates to an equal manifest (JSON emits
    tuples as arrays; from_json converts them back)."""
    manifest = ArrayManifest.model_validate({"path": "a/b", "metadata": VALID_DOC})
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
    from zarr_metadata.model import UnsetType
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
            "UnsetType": UnsetType,
        },
    )
    model_shaped = {
        "shape": [10],
        "fill_value": 0,
        "data_type": {"name": "uint8", "configuration": {}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "codecs": [{"name": "bytes", "configuration": {}}],
        "chunk_key_encoding": {"name": "default", "configuration": {}},
        "dimension_names": UnsetType.UNSET,
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


# --- a first-class pydantic model, engine-backed (the pydantic-zarr pattern) --
#
# When a consumer wants a real BaseModel — JSON schema generation, and
# generics for typed attributes, as in pydantic-zarr's ArraySpec — the model
# fields are pydantic-native, but validation and serialization still route
# through the library: a mode="before" validator canonicalizes every input
# document with from_json(...).to_json(), so the structural validators and
# normalization run BEFORE pydantic parses fields (no coercion divergence),
# and the document form is the bridge in both directions.

AttrsT = TypeVar("AttrsT")


class NamedConfig(BaseModel):
    """Pydantic mirror of a normalized metadata field (name + configuration)."""

    name: str
    configuration: dict[str, JSONValue] = {}


class ArrayMetadataV3Spec(BaseModel, Generic[AttrsT]):
    """A pydantic-native, attribute-typed view of a v3 array metadata document.

    The library is the engine: every input is canonicalized and structurally
    validated by `ArrayMetadataModelV3.from_json` before pydantic sees the
    fields, and `to_document` / `to_metadata_model` emit through the library.
    """

    model_config = ConfigDict(frozen=True)

    zarr_format: int = 3
    node_type: str = "array"
    shape: tuple[int, ...]
    data_type: NamedConfig
    chunk_grid: NamedConfig
    chunk_key_encoding: NamedConfig
    fill_value: JSONValue
    codecs: tuple[NamedConfig, ...]
    attributes: AttrsT
    dimension_names: tuple[str | None, ...] | None = None
    storage_transformers: tuple[NamedConfig, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data: object) -> object:
        """Route every input document through the library's validation and
        normalization; pydantic then parses only canonical documents."""
        if isinstance(data, Mapping):
            doc = dict(ArrayMetadataModelV3.from_json(data).to_json())
            doc.setdefault("attributes", {})
            return doc
        return data

    def to_metadata_model(self) -> ArrayMetadataModelV3:
        """Bridge back to the canonical model, via the document form.

        In the document, "no dimension names" is key-absence, not null; the
        pydantic-side None translates to dropping the key.
        """
        doc = self.model_dump()
        if doc["dimension_names"] is None:
            del doc["dimension_names"]
        return ArrayMetadataModelV3.from_json(doc)

    def to_document(self) -> dict[str, object]:
        """The canonical document (omit-empty conventions applied)."""
        return dict(self.to_metadata_model().to_json())


class MicroscopyAttrs(BaseModel):
    """Example of consumer-typed attributes, pydantic-zarr style."""

    resolution_um: float


def test_spec_typed_attributes() -> None:
    """The generic parameter types the attributes, so consumers get validated,
    attribute-level access — the pydantic-zarr ArraySpec pattern."""
    doc = dict(VALID_DOC) | {"attributes": {"resolution_um": 0.5}}
    spec = ArrayMetadataV3Spec[MicroscopyAttrs].model_validate(doc)
    assert spec.attributes.resolution_um == 0.5
    assert spec.data_type == NamedConfig(name="uint8")


def test_spec_engine_validates_before_pydantic() -> None:
    """The library's structural validation runs before pydantic's parsing, so
    coercion cannot re-open validator holes (contrast with the native
    introspection test above, where [True, -5] coerced to (1, -5))."""
    with pytest.raises(ValidationError, match="shape"):
        ArrayMetadataV3Spec[MicroscopyAttrs].model_validate(
            dict(VALID_DOC) | {"shape": [True, -5], "attributes": {"resolution_um": 0.5}}
        )


def test_spec_bridges_to_canonical_model_and_document() -> None:
    """to_metadata_model / to_document round-trip through the document form,
    and the emitted document matches what the library itself would emit."""
    doc = dict(VALID_DOC) | {"attributes": {"resolution_um": 0.5}}
    spec = ArrayMetadataV3Spec[MicroscopyAttrs].model_validate(doc)
    model = spec.to_metadata_model()
    assert isinstance(model, ArrayMetadataModelV3)
    assert spec.to_document() == dict(model.to_json())
    # and back: the document revalidates to an equal spec
    assert ArrayMetadataV3Spec[MicroscopyAttrs].model_validate(spec.to_document()) == spec


def test_spec_json_schema_generation() -> None:
    """A real BaseModel means model_json_schema works — the capability the
    opaque InstanceOf pattern cannot provide."""
    schema = ArrayMetadataV3Spec[MicroscopyAttrs].model_json_schema()
    assert schema["properties"]["shape"]["type"] == "array"
    assert "MicroscopyAttrs" in schema["$defs"]
