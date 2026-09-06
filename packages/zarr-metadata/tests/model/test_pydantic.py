"""Executable example: integrating the metadata models with pydantic (v2).

Pydantic's native dataclass introspection CAN be made to work (see
`test_native_dataclass_introspection_is_possible_but_diverges`): the models
keep their annotation-only imports behind `TYPE_CHECKING`, so a bare
`TypeAdapter(ZarrV3ArrayMetadata)` raises `class-not-fully-defined`, but
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
    PydanticSchemaGenerationError,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from zarr_metadata import JSONValue
from zarr_metadata.model import ZarrV3ArrayMetadata, ZarrV3NamedConfig

# --- the integration (this is the example) -----------------------------------


def _as_array_metadata_v3(value: object) -> ZarrV3ArrayMetadata:
    """Accept an existing model instance or a raw metadata document."""
    if isinstance(value, ZarrV3ArrayMetadata):
        return value
    return ZarrV3ArrayMetadata.from_json(value)


# return_type is explicit because to_json's own annotation (`ZarrV3ArrayMetadataJSON`)
# is a TYPE_CHECKING-only name pydantic cannot resolve at runtime.
ArrayMetadataV3Field = Annotated[
    InstanceOf[ZarrV3ArrayMetadata],
    BeforeValidator(_as_array_metadata_v3),
    PlainSerializer(ZarrV3ArrayMetadata.to_json, return_type=dict),
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
    assert isinstance(manifest.metadata, ZarrV3ArrayMetadata)
    assert manifest.metadata.shape == (10,)
    assert manifest.metadata.data_type.name == "uint8"


def test_model_instance_passes_through() -> None:
    """An already-constructed model instance is accepted unchanged."""
    model = ZarrV3ArrayMetadata.from_json(VALID_DOC)
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
    # Empty configurations use the extension-definition shorthand form.
    assert dumped["metadata"]["data_type"] == "uint8"


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
    assert isinstance(model, ZarrV3ArrayMetadata)
    assert adapter.dump_python(model) == model.to_json()


# --- the road not taken: native dataclass introspection ----------------------


def test_native_dataclass_introspection_is_not_supported() -> None:
    """Pydantic cannot field-introspect the model dataclasses: the UNSET
    sentinel (PEP 661, typing_extensions.Sentinel) in the optional-field
    annotations has no pydantic schema (as of pydantic 2.13), so even the
    rebuild-with-namespace recipe fails. Introspection was already the wrong
    tool before the sentinel existed — it validated the model shape rather
    than the document, and its lax coercion re-opened validator holes (e.g.
    shape=[True, -5] coerced to (1, -5)) — so the delegation patterns above
    are the only supported integrations. If this test ever fails because
    pydantic learned to handle sentinels, revisit whether the introspection
    path needs its divergences documented again."""
    from zarr_metadata._common import JSONValue
    from zarr_metadata.model import UNSET
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON, ZarrV3ExtensionField

    def build_and_use() -> None:
        adapter = TypeAdapter(ZarrV3ArrayMetadata)
        adapter.rebuild(
            force=True,
            _types_namespace={
                "JSONValue": JSONValue,
                "ZarrV3ExtensionField": ZarrV3ExtensionField,
                "ZarrV3MetadataFieldJSON": ZarrV3MetadataFieldJSON,
                "ZarrV3ArrayMetadataJSON": ZarrV3ArrayMetadataJSON,
                "ZarrV3NamedConfig": ZarrV3NamedConfig,
                "ZarrV3MetadataField": ZarrV3NamedConfig,
                "UNSET": UNSET,
            },
        )
        adapter.validate_python({})

    with pytest.raises((AttributeError, PydanticSchemaGenerationError, PydanticUserError)):
        build_and_use()


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
    """Pydantic mirror of a normalized metadata extension envelope."""

    name: str
    configuration: dict[str, JSONValue] = {}
    must_understand: bool = True


class ArrayMetadataV3Spec(BaseModel, Generic[AttrsT]):
    """A pydantic-native, attribute-typed view of a v3 array metadata document.

    The library is the engine: every input is canonicalized and structurally
    validated by `ZarrV3ArrayMetadata.from_json` before pydantic sees the
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
            doc = dict(ZarrV3ArrayMetadata.from_json(data).to_json())
            for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
                if isinstance(doc[key], str):
                    doc[key] = {"name": doc[key]}
            for key in ("codecs", "storage_transformers"):
                doc[key] = tuple(
                    {"name": item} if isinstance(item, str) else item for item in doc.get(key, ())
                )
            doc.setdefault("attributes", {})
            return doc
        return data

    def to_metadata_model(self) -> ZarrV3ArrayMetadata:
        """Bridge back to the canonical model, via the document form.

        In the document, "no dimension names" is key-absence, not null; the
        pydantic-side None translates to dropping the key.
        """
        doc = self.model_dump()
        if doc["dimension_names"] is None:
            del doc["dimension_names"]
        return ZarrV3ArrayMetadata.from_json(doc)

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
    assert isinstance(model, ZarrV3ArrayMetadata)
    assert spec.to_document() == dict(model.to_json())
    # and back: the document revalidates to an equal spec
    assert ArrayMetadataV3Spec[MicroscopyAttrs].model_validate(spec.to_document()) == spec


def test_spec_json_schema_generation() -> None:
    """A real BaseModel means model_json_schema works — the capability the
    opaque InstanceOf pattern cannot provide."""
    schema = ArrayMetadataV3Spec[MicroscopyAttrs].model_json_schema()
    assert schema["properties"]["shape"]["type"] == "array"
    assert "MicroscopyAttrs" in schema["$defs"]
