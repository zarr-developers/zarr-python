"""Tests for the group and consolidated metadata models in `zarr_metadata.model`."""

import json

import pytest

from zarr_metadata.model._group import (
    GroupMetadataModelV2,
    GroupMetadataModelV3,
)
from zarr_metadata.model._validation import (
    MetadataValidationError,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
)

# --- GroupMetadataModelV3 ---------------------------------------------------


def test_group_v3_roundtrip() -> None:
    """A v3 group document round-trips through the model unchanged."""
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": (1, 2)}}
    model = GroupMetadataModelV3.from_json(doc)
    assert model.to_json() == doc


def test_group_v3_omits_empty_attributes() -> None:
    """to_json omits the attributes key when attributes is empty."""
    model = GroupMetadataModelV3.create_default()
    assert "attributes" not in model.to_json()


def test_group_v3_lists_become_tuples() -> None:
    """from_json converts JSON arrays in attributes to tuples."""
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": [1, 2]}}
    model = GroupMetadataModelV3.from_json(doc)
    assert model.attributes == {"a": (1, 2)}


def test_group_v3_extra_fields_roundtrip() -> None:
    """Unknown top-level keys land in extra_fields and reappear in to_json."""
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "my_extension": {"name": "thing", "must_understand": False},
    }
    model = GroupMetadataModelV3.from_json(doc)
    assert model.extra_fields == {"my_extension": {"name": "thing", "must_understand": False}}
    assert model.to_json() == doc


def test_group_v3_extra_fields_overlap_rejected() -> None:
    """Constructing a v3 group model with extra_fields shadowing a standard key raises."""
    with pytest.raises(ValueError, match="Extra fields"):
        GroupMetadataModelV3(
            attributes={},
            consolidated_metadata=None,
            extra_fields={"node_type": {"name": "x", "must_understand": False}},
        )


def test_group_v3_consolidated_extra_field_rejected() -> None:
    """extra_fields may not shadow the consolidated_metadata convention key."""
    with pytest.raises(ValueError, match="Extra fields"):
        GroupMetadataModelV3(
            attributes={},
            consolidated_metadata=None,
            extra_fields={"consolidated_metadata": {"name": "x", "must_understand": False}},
        )


def test_group_v3_missing_required_key() -> None:
    """parse_group_metadata_v3 reports each missing required key."""
    with pytest.raises(MetadataValidationError, match="node_type"):
        parse_group_metadata_v3({"zarr_format": 3})


def test_group_v3_bad_attributes() -> None:
    """parse_group_metadata_v3 rejects a non-mapping attributes value."""
    with pytest.raises(MetadataValidationError, match="attributes"):
        parse_group_metadata_v3({"zarr_format": 3, "node_type": "group", "attributes": 5})


def test_group_v3_key_value_roundtrip() -> None:
    """from_key_value(to_key_value()) is the identity for v3 groups."""
    model = GroupMetadataModelV3.create_default(attributes={"a": 1})
    assert GroupMetadataModelV3.from_key_value(model.to_key_value()) == model


def test_group_v3_update() -> None:
    """update replaces the given fields and returns a new instance."""
    base = GroupMetadataModelV3.create_default()
    updated = base.update(attributes={"a": 1})
    assert updated.attributes == {"a": 1}
    assert base.attributes == {}


# --- GroupMetadataModelV2 ---------------------------------------------------


def test_group_v2_key_value_split() -> None:
    """v2 to_key_value writes .zgroup and .zattrs; from_key_value merges them."""
    model = GroupMetadataModelV2.create_default(attributes={"a": 1})
    kv = model.to_key_value()
    assert set(kv) == {".zgroup", ".zattrs"}
    assert json.loads(kv[".zgroup"]) == {"zarr_format": 2}
    assert GroupMetadataModelV2.from_key_value(kv) == model


def test_group_v2_from_key_value_without_zattrs() -> None:
    """A v2 group with no .zattrs file parses with empty attributes."""
    model = GroupMetadataModelV2.from_key_value({".zgroup": b'{"zarr_format": 2}'})
    assert model.attributes == {}


def test_group_v2_json_roundtrip() -> None:
    """A merged-form v2 group document round-trips through the model unchanged."""
    doc = {"zarr_format": 2, "attributes": {"a": 1}}
    model = GroupMetadataModelV2.from_json(doc)
    assert model.to_json() == doc


def test_group_v2_omits_empty_attributes() -> None:
    """to_json omits the attributes key when attributes is empty."""
    assert "attributes" not in GroupMetadataModelV2.create_default().to_json()


def test_group_v2_not_a_mapping() -> None:
    """parse_group_metadata_v2 rejects a non-mapping document."""
    with pytest.raises(MetadataValidationError, match="expected a mapping"):
        parse_group_metadata_v2([1, 2, 3])


def test_group_v2_missing_required_key() -> None:
    """parse_group_metadata_v2 reports a missing zarr_format key."""
    with pytest.raises(MetadataValidationError, match="zarr_format"):
        parse_group_metadata_v2({})
