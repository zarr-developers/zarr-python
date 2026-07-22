"""Tests for `zarr_metadata.pydantic`, the optional pydantic field-type module.

The hand-rolled recipes in `test_pydantic.py` document how the integration
works; this module ships it. Instances are the CORE model classes (no
parallel hierarchy), so values interoperate freely with non-pydantic code.
"""

import pytest
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
    assert schema["properties"]["metadata"] == {"type": "object", "title": "ZarrV3ArrayMetadata"}
    assert schema["properties"]["codec"]["anyOf"] == [{"type": "string"}, {"type": "object"}]


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
