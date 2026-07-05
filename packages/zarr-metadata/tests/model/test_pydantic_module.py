"""Tests for `zarr_metadata.pydantic`, the optional pydantic field-type module.

The hand-rolled recipes in `test_pydantic.py` document how the integration
works; this module ships it. Instances are the CORE model classes (no
parallel hierarchy), so values interoperate freely with non-pydantic code.
"""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

import zarr_metadata.pydantic as zmp
from zarr_metadata.model import (
    ArrayMetadataModelV2,
    ArrayMetadataModelV3,
    ConsolidatedMetadataModelV2,
    ConsolidatedMetadataModelV3,
    GroupMetadataModelV2,
    GroupMetadataModelV3,
    NamedConfigModelV3,
)

V3_ARRAY_DOC = dict(ArrayMetadataModelV3.create_default(shape=(4,)).to_json())
V2_ARRAY_DOC = dict(ArrayMetadataModelV2.create_default(shape=(4,), chunks=(2,)).to_json())
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
    pytest.param(zmp.ArrayMetadataV3, ArrayMetadataModelV3, V3_ARRAY_DOC, id="array-v3"),
    pytest.param(zmp.ArrayMetadataV2, ArrayMetadataModelV2, V2_ARRAY_DOC, id="array-v2"),
    pytest.param(zmp.GroupMetadataV3, GroupMetadataModelV3, V3_GROUP_DOC, id="group-v3"),
    pytest.param(zmp.GroupMetadataV2, GroupMetadataModelV2, V2_GROUP_DOC, id="group-v2"),
    pytest.param(
        zmp.ConsolidatedMetadataV3,
        ConsolidatedMetadataModelV3,
        V3_CONSOLIDATED_DOC,
        id="consolidated-v3",
    ),
    pytest.param(
        zmp.ConsolidatedMetadataV2,
        ConsolidatedMetadataModelV2,
        V2_CONSOLIDATED_DOC,
        id="consolidated-v2",
    ),
    pytest.param(zmp.MetadataFieldV3, NamedConfigModelV3, {"name": "bytes"}, id="field-v3"),
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
    assert adapter.dump_python(model) == dict(model.to_json())


def test_core_instances_interoperate() -> None:
    """A core model instance (e.g. handed out by zarr-python) drops straight
    into a pydantic field — the reason the module ships Annotated aliases over
    the core classes rather than pydantic-aware subclasses."""

    class Manifest(BaseModel):
        metadata: zmp.ArrayMetadataV3

    core = ArrayMetadataModelV3.from_json(V3_ARRAY_DOC)
    manifest = Manifest(metadata=core)
    assert manifest.metadata is core


def test_validation_error_carries_problems() -> None:
    """A defective document fails with the library's loc-annotated messages."""

    class Manifest(BaseModel):
        metadata: zmp.ArrayMetadataV3

    doc = dict(V3_ARRAY_DOC)
    del doc["chunk_key_encoding"]
    with pytest.raises(ValidationError, match="chunk_key_encoding: missing required key"):
        Manifest(metadata=doc)  # type: ignore[arg-type]


def test_json_schema_generation() -> None:
    """model_json_schema works, describing the document form each field accepts."""

    class Manifest(BaseModel):
        metadata: zmp.ArrayMetadataV3
        codec: zmp.MetadataFieldV3

    schema = Manifest.model_json_schema()
    assert schema["properties"]["metadata"] == {"type": "object", "title": "ArrayMetadataV3"}
    assert schema["properties"]["codec"]["anyOf"] == [{"type": "string"}, {"type": "object"}]


def test_json_roundtrip() -> None:
    """model_dump_json output re-validates to an equal pydantic model."""

    class Manifest(BaseModel):
        metadata: zmp.ArrayMetadataV3

    manifest = Manifest(metadata=V3_ARRAY_DOC)  # type: ignore[arg-type]
    assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_core_package_does_not_import_pydantic() -> None:
    """Importing zarr_metadata (in a fresh interpreter) must not import
    pydantic: the integration is opt-in via zarr_metadata.pydantic."""
    import subprocess
    import sys

    code = "import sys, zarr_metadata; assert 'pydantic' not in sys.modules, 'leaked'"
    subprocess.run([sys.executable, "-c", code], check=True)
