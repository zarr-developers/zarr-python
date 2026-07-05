"""Tests for the group and consolidated metadata models in `zarr_metadata.model`."""

import dataclasses
import json

import pytest

from zarr_metadata.model._array import ArrayMetadataModelV3
from zarr_metadata.model._group import (
    ConsolidatedMetadataModelV2,
    ConsolidatedMetadataModelV3,
    GroupMetadataModelV2,
    GroupMetadataModelV2Partial,
    GroupMetadataModelV3,
    GroupMetadataModelV3Partial,
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


# --- Partial TypedDict drift guards -----------------------------------------


def test_group_partial_keys_match_settable_model_fields() -> None:
    """Each group partial TypedDict must list exactly the settable model fields.

    Guards against drift: adding/removing a settable field on a group model
    without updating its `*Partial` TypedDict fails here.
    """
    for model_cls, partial_cls in (
        (GroupMetadataModelV3, GroupMetadataModelV3Partial),
        (GroupMetadataModelV2, GroupMetadataModelV2Partial),
    ):
        settable = {f.name for f in dataclasses.fields(model_cls) if f.init}
        assert set(partial_cls.__annotations__) == settable


# --- ConsolidatedMetadataModelV3 --------------------------------------------


def test_consolidated_v3_roundtrip() -> None:
    """A v3 group with inline consolidated metadata round-trips, with child
    entries parsed into array/group models."""
    child = ArrayMetadataModelV3.create_default(shape=(2,)).to_json()
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": child, "g": {"zarr_format": 3, "node_type": "group"}},
        },
    }
    model = GroupMetadataModelV3.from_json(doc)
    assert isinstance(model.consolidated_metadata, ConsolidatedMetadataModelV3)
    assert isinstance(model.consolidated_metadata.metadata["a"], ArrayMetadataModelV3)
    assert isinstance(model.consolidated_metadata.metadata["g"], GroupMetadataModelV3)
    assert model.to_json() == doc


def test_consolidated_v3_must_understand_true_rejected() -> None:
    """ConsolidatedMetadataModelV3 enforces must_understand=False at runtime."""
    with pytest.raises(ValueError, match="must_understand"):
        ConsolidatedMetadataModelV3(must_understand=True, metadata={})


def test_consolidated_v3_from_json_must_understand_true_rejected() -> None:
    """from_json rejects a consolidated document carrying must_understand=true."""
    with pytest.raises(MetadataValidationError, match="must_understand"):
        ConsolidatedMetadataModelV3.from_json(
            {"kind": "inline", "must_understand": True, "metadata": {}}
        )


def test_consolidated_v3_entry_without_node_type_rejected() -> None:
    """from_json rejects a consolidated entry lacking a recognizable node_type."""
    with pytest.raises(MetadataValidationError, match="node_type"):
        ConsolidatedMetadataModelV3.from_json(
            {"kind": "inline", "must_understand": False, "metadata": {"a": {"zarr_format": 3}}}
        )


def test_consolidated_v3_not_a_mapping() -> None:
    """from_json rejects a non-mapping consolidated document."""
    with pytest.raises(MetadataValidationError, match="expected a mapping"):
        ConsolidatedMetadataModelV3.from_json(5)


# --- ConsolidatedMetadataModelV2 --------------------------------------------


def test_consolidated_v2_verbatim_roundtrip() -> None:
    """The v2 .zmetadata model holds the flat file-keyed map verbatim,
    including nodes that have no .zattrs entry."""
    doc = {
        "zarr_consolidated_format": 1,
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "a/.zarray": {
                "zarr_format": 2,
                "shape": (2,),
                "chunks": (2,),
                "dtype": "|u1",
                "fill_value": 0,
                "order": "C",
                "compressor": None,
                "filters": None,
            },
        },
    }
    model = ConsolidatedMetadataModelV2.from_json(doc)
    assert model.to_json() == doc


def test_consolidated_v2_key_value_roundtrip() -> None:
    """from_key_value(to_key_value()) is the identity for .zmetadata documents."""
    model = ConsolidatedMetadataModelV2.from_json(
        {"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}}
    )
    assert ConsolidatedMetadataModelV2.from_key_value(model.to_key_value()) == model


def test_consolidated_v2_lists_become_tuples() -> None:
    """from_json converts JSON arrays inside entries to tuples."""
    doc = {
        "zarr_consolidated_format": 1,
        "metadata": {"a/.zarray": {"shape": [2, 3]}},
    }
    model = ConsolidatedMetadataModelV2.from_json(doc)
    assert model.metadata == {"a/.zarray": {"shape": (2, 3)}}


def test_consolidated_v2_envelope_validation() -> None:
    """from_json rejects a .zmetadata document missing the metadata key."""
    with pytest.raises(MetadataValidationError, match="metadata"):
        ConsolidatedMetadataModelV2.from_json({"zarr_consolidated_format": 1})


def test_consolidated_v2_not_a_mapping() -> None:
    """from_json rejects a non-mapping .zmetadata document."""
    with pytest.raises(MetadataValidationError, match="expected a mapping"):
        ConsolidatedMetadataModelV2.from_json([1])
