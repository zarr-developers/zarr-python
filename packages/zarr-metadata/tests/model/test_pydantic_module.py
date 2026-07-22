"""Tests for `zarr_metadata.pydantic`, the optional pydantic field-type module.

The hand-rolled recipes in `test_pydantic.py` document how the integration
works; this module ships it. Instances are the CORE model classes (no
parallel hierarchy), so values interoperate freely with non-pydantic code.
"""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter, ValidationError

import zarr_metadata.pydantic as zmp
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3NamedConfig,
)

V3_ARRAY_DOC = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
V2_ARRAY_DOC = dict(ZarrV2ArrayMetadata.create_default(shape=(4,), chunks=(2,)).to_json())
V3_GROUP_DOC = {"zarr_format": 3, "node_type": "group", "attributes": {"a": 1}}
V2_GROUP_DOC = {"zarr_format": 2, "attributes": {"a": 1}}
V3_CONSOLIDATED_DOC = {
    "kind": "inline",
    "must_understand": False,
    "metadata": {"a": dict(V3_ARRAY_DOC)},
}
V2_CONSOLIDATED_DOC = {
    "zarr_consolidated_format": 1,
    "metadata": {".zgroup": {"zarr_format": 2}},
}

FIELD_CASES = [
    pytest.param(zmp.ZarrV3ArrayMetadata, ZarrV3ArrayMetadata, V3_ARRAY_DOC, id="array-v3"),
    pytest.param(zmp.ZarrV2ArrayMetadata, ZarrV2ArrayMetadata, V2_ARRAY_DOC, id="array-v2"),
    pytest.param(zmp.ZarrV3GroupMetadata, ZarrV3GroupMetadata, V3_GROUP_DOC, id="group-v3"),
    pytest.param(zmp.ZarrV2GroupMetadata, ZarrV2GroupMetadata, V2_GROUP_DOC, id="group-v2"),
    pytest.param(
        zmp.ZarrV3ConsolidatedMetadata,
        ZarrV3ConsolidatedMetadata,
        V3_CONSOLIDATED_DOC,
        id="consolidated-v3",
    ),
    pytest.param(
        zmp.ZarrV2ConsolidatedMetadata,
        ZarrV2ConsolidatedMetadata,
        V2_CONSOLIDATED_DOC,
        id="consolidated-v2",
    ),
    pytest.param(zmp.ZarrV3MetadataField, ZarrV3NamedConfig, {"name": "bytes"}, id="field-v3"),
]


@pytest.mark.parametrize(("field_type", "model_cls", "doc"), FIELD_CASES)
def test_field_type_validates_and_dumps_canonically(
    field_type: object, model_cls: type, doc: dict[str, object]
) -> None:
    """Each field type parses its raw document into the CORE model class,
    passes existing instances through unchanged, and dumps the canonical
    document via to_json."""
    adapter = TypeAdapter(field_type)
    model = adapter.validate_python(doc)
    assert type(model) is model_cls
    assert adapter.validate_python(model) is model
    assert adapter.dump_python(model) == model.to_json()


def test_core_instances_interoperate() -> None:
    """A core model instance (e.g. handed out by zarr-python) drops straight
    into a pydantic field — the reason the module ships Annotated aliases over
    the core classes rather than pydantic-aware subclasses."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadata

    core = ZarrV3ArrayMetadata.from_json(V3_ARRAY_DOC)
    manifest = Manifest(metadata=core)
    assert manifest.metadata is core


def test_validation_error_carries_problems() -> None:
    """A defective document fails with the library's loc-annotated messages."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadata

    doc = dict(V3_ARRAY_DOC)
    del doc["chunk_key_encoding"]
    with pytest.raises(ValidationError, match="chunk_key_encoding: missing required key"):
        Manifest.model_validate({"metadata": doc})


def test_json_schema_generation() -> None:
    """model_json_schema works, describing the document form each field accepts."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadata
        codec: zmp.ZarrV3MetadataField

    schema = Manifest.model_json_schema()
    metadata_schema = schema["$defs"]["ZarrV3ArrayMetadataJSON"]
    assert schema["properties"]["metadata"]["$ref"] == "#/$defs/ZarrV3ArrayMetadataJSON"
    assert metadata_schema["required"] == [
        "zarr_format",
        "node_type",
        "data_type",
        "shape",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
    ]
    assert metadata_schema["properties"]["zarr_format"] == {
        "const": 3,
        "title": "Zarr Format",
        "type": "integer",
    }
    assert schema["properties"]["codec"]["anyOf"] == [
        {"type": "string"},
        {"$ref": "#/$defs/ZarrV3NamedConfigJSON"},
    ]


def test_v2_recursive_structured_dtype_is_in_pydantic_schema() -> None:
    """The schema accepts nested structured dtypes supported by the v2 specification."""
    doc = json.loads(json.dumps(V2_ARRAY_DOC))
    doc["dtype"] = [["outer", [["inner", "<i4"]]]]
    adapter = TypeAdapter(zmp.ZarrV2ArrayMetadata)

    assert adapter.validate_python(doc).dtype == (("outer", (("inner", "<i4"),)),)
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


def _assert_runtime_and_schema_reject(field_type: object, document: dict[str, object]) -> None:
    adapter = TypeAdapter(field_type)
    with pytest.raises(ValidationError):
        adapter.validate_python(document)
    assert list(Draft202012Validator(adapter.json_schema()).iter_errors(document))


def test_v3_array_schema_rejects_empty_codecs() -> None:
    """The generated schema mirrors the runtime non-empty codec pipeline rule."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc["codecs"] = []

    _assert_runtime_and_schema_reject(zmp.ZarrV3ArrayMetadata, doc)


def test_array_schemas_reject_negative_dimensions() -> None:
    """Both array schemas mirror the runtime non-negative dimension rule."""
    for field_type, source in (
        (zmp.ZarrV3ArrayMetadata, V3_ARRAY_DOC),
        (zmp.ZarrV2ArrayMetadata, V2_ARRAY_DOC),
    ):
        doc = json.loads(json.dumps(source))
        doc["shape"] = [-1]
        _assert_runtime_and_schema_reject(field_type, doc)


@pytest.mark.parametrize("field", ["data_type", "chunk_grid", "chunk_key_encoding"])
def test_v3_array_schema_rejects_false_at_mandatory_extension_points(field: str) -> None:
    """Mandatory v3 extension points cannot opt out of understanding."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc[field] = {"name": "example", "must_understand": False}

    _assert_runtime_and_schema_reject(zmp.ZarrV3ArrayMetadata, doc)


def test_metadata_field_schema_rejects_unknown_members() -> None:
    """Named-configuration envelopes are closed in both runtime and schema validation."""
    _assert_runtime_and_schema_reject(
        zmp.ZarrV3MetadataField,
        {"name": "example", "unexpected": 1},
    )


@pytest.mark.parametrize(
    ("field_type", "source"),
    [
        (zmp.ZarrV2ArrayMetadata, V2_ARRAY_DOC),
        (zmp.ZarrV2GroupMetadata, V2_GROUP_DOC),
        (zmp.ZarrV2ConsolidatedMetadata, V2_CONSOLIDATED_DOC),
    ],
)
def test_v2_schema_rejects_unknown_document_members(
    field_type: object, source: dict[str, object]
) -> None:
    """Closed v2 merged documents expose their runtime boundary in JSON Schema."""
    doc = json.loads(json.dumps(source))
    doc["unexpected"] = 1

    _assert_runtime_and_schema_reject(field_type, doc)


def test_v3_array_schema_allows_unknown_extension_fields() -> None:
    """Schema constraints do not close the v3 top-level extension namespace."""
    doc = json.loads(json.dumps(V3_ARRAY_DOC))
    doc["vendor_extension"] = {"anything": [1, 2]}
    adapter = TypeAdapter(zmp.ZarrV3ArrayMetadata)

    assert adapter.validate_python(doc).extra_fields == {"vendor_extension": {"anything": (1, 2)}}
    assert Draft202012Validator(adapter.json_schema()).is_valid(doc)


def test_json_roundtrip() -> None:
    """model_dump_json output re-validates to an equal pydantic model."""

    class Manifest(BaseModel):
        metadata: zmp.ZarrV3ArrayMetadata

    manifest = Manifest.model_validate({"metadata": V3_ARRAY_DOC})
    assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_metadata_field_serializes_shorthand_and_false_object() -> None:
    """The optional integration exposes the core model's canonical extension form."""
    adapter = TypeAdapter(zmp.ZarrV3MetadataField)
    assert adapter.dump_python(adapter.validate_python({"name": "bytes"})) == "bytes"
    assert adapter.dump_python(
        adapter.validate_python({"name": "optional", "must_understand": False})
    ) == {"name": "optional", "must_understand": False}


def test_core_package_does_not_import_pydantic() -> None:
    """Importing zarr_metadata (in a fresh interpreter) must not import
    pydantic: the integration is opt-in via zarr_metadata.pydantic."""
    import subprocess
    import sys

    code = "import sys, zarr_metadata; assert 'pydantic' not in sys.modules, 'leaked'"
    subprocess.run([sys.executable, "-c", code], check=True)
