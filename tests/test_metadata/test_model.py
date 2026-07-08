"""Tests for the conversions between zarr-python's runtime metadata classes
and the ``zarr_metadata`` model layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from zarr_metadata.model import (
    UNSET,
    ZarrV2ArrayMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    ZarrV3NamedConfig,
)

from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata.model import (
    array_metadata_from_model,
    array_metadata_to_model,
    group_metadata_from_model,
    group_metadata_to_model,
)
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.errors import UnknownCodecError
from zarr.testing.strategies import array_metadata

if TYPE_CHECKING:
    from zarr.core.metadata.v2 import ArrayV2Metadata


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(metadata=array_metadata())  # type: ignore[misc]
@settings(max_examples=100)
def test_array_metadata_round_trip(metadata: ArrayV2Metadata | ArrayV3Metadata) -> None:
    """Runtime -> model -> runtime is the identity (runtime equality compares
    the JSON document form, so NaN fill values compare equal)."""
    model = array_metadata_to_model(metadata)
    if metadata.zarr_format == 2:
        assert isinstance(model, ZarrV2ArrayMetadata)
    else:
        assert isinstance(model, ZarrV3ArrayMetadata)
    assert array_metadata_from_model(model) == metadata


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@given(metadata=array_metadata())  # type: ignore[misc]
@settings(max_examples=100)
def test_array_metadata_model_stable(metadata: ArrayV2Metadata | ArrayV3Metadata) -> None:
    """The model's own document form re-parses to an equal model: conversion
    lands on the model layer's canonical representation."""
    model = array_metadata_to_model(metadata)
    assert type(model).from_json(model.to_json()) == model


def test_v3_nan_fill_value_round_trip() -> None:
    metadata = ArrayV3Metadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": (4,),
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,)}},
            "chunk_key_encoding": {"name": "default"},
            "fill_value": "NaN",
            "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        }
    )
    model = array_metadata_to_model(metadata)
    assert model.fill_value == "NaN"
    restored = array_metadata_from_model(model)
    assert isinstance(restored, ArrayV3Metadata)
    assert math.isnan(restored.fill_value)
    assert restored == metadata


def test_v3_dimension_names_absent_maps_to_unset() -> None:
    model = array_metadata_to_model(
        ArrayV3Metadata.from_dict(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": (2, 2),
                "data_type": "uint8",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1, 1)}},
                "chunk_key_encoding": {"name": "default"},
                "fill_value": 0,
                "codecs": ({"name": "bytes"},),
            }
        )
    )
    assert model.dimension_names is UNSET


def test_v3_extra_fields_round_trip() -> None:
    metadata = ArrayV3Metadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": (2,),
            "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
            "chunk_key_encoding": {"name": "default"},
            "fill_value": 0,
            "codecs": ({"name": "bytes"},),
            "my-extension": {"must_understand": False, "value": 10},
        }
    )
    model = array_metadata_to_model(metadata)
    assert isinstance(model, ZarrV3ArrayMetadata)
    assert model.extra_fields == {"my-extension": {"must_understand": False, "value": 10}}
    assert model.must_understand_fields == {}
    assert array_metadata_from_model(model) == metadata


def test_model_with_unknown_codec_rejected_by_runtime() -> None:
    """model -> runtime raises exactly where opening the document would: on
    extension points this installation cannot interpret."""
    model = ZarrV3ArrayMetadata.create_default(
        shape=(4,),
        codecs=(ZarrV3NamedConfig(name="does-not-exist", configuration={}),),
    )
    with pytest.raises(UnknownCodecError):
        array_metadata_from_model(model)


@pytest.mark.parametrize("attributes", [{}, {"a": 1, "nested": {"b": [1, 2]}}])
def test_group_metadata_v3_round_trip(attributes: dict[str, object]) -> None:
    metadata = GroupMetadata(attributes=attributes, zarr_format=3)
    model = group_metadata_to_model(metadata)
    assert isinstance(model, ZarrV3GroupMetadata)
    assert model.consolidated_metadata is UNSET
    assert group_metadata_from_model(model) == metadata


@pytest.mark.parametrize("attributes", [{}, {"a": 1}])
def test_group_metadata_v2_round_trip(attributes: dict[str, object]) -> None:
    metadata = GroupMetadata(attributes=attributes, zarr_format=2)
    model = group_metadata_to_model(metadata)
    assert isinstance(model, ZarrV2GroupMetadata)
    assert model.attributes == attributes
    assert group_metadata_from_model(model) == metadata


def test_group_metadata_v3_consolidated_round_trip() -> None:
    array_meta = ArrayV3Metadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": (4,),
            "data_type": "int32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,)}},
            "chunk_key_encoding": {"name": "default"},
            "fill_value": 0,
            "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        }
    )
    metadata = GroupMetadata(
        attributes={"root": True},
        zarr_format=3,
        consolidated_metadata=ConsolidatedMetadata(
            metadata={
                "child-group": GroupMetadata(
                    attributes={"child": True},
                    zarr_format=3,
                    consolidated_metadata=ConsolidatedMetadata(metadata={"nested": array_meta}),
                ),
                "child-array": array_meta,
            }
        ),
    )
    model = group_metadata_to_model(metadata)
    assert isinstance(model, ZarrV3GroupMetadata)
    assert model.consolidated_metadata is not UNSET
    # The model holds consolidated entries flat, keyed by full path.
    assert set(model.consolidated_metadata.metadata) == {
        "child-group",
        "child-group/nested",
        "child-array",
    }
    restored = group_metadata_from_model(model)
    assert restored == metadata
    # The nesting structure survives, not just equality of the JSON form.
    assert restored.consolidated_metadata is not None
    child = restored.consolidated_metadata.metadata["child-group"]
    assert isinstance(child, GroupMetadata)
    assert child.consolidated_metadata is not None
    assert set(child.consolidated_metadata.metadata) == {"nested"}


def test_group_metadata_v2_consolidated_dropped() -> None:
    """v2 consolidated metadata lives in `.zmetadata`, not `.zgroup`; the v2
    group document model has no field for it."""
    metadata = GroupMetadata(
        attributes={},
        zarr_format=2,
        consolidated_metadata=ConsolidatedMetadata(metadata={}),
    )
    model = group_metadata_to_model(metadata)
    assert isinstance(model, ZarrV2GroupMetadata)
    restored = group_metadata_from_model(model)
    assert restored.consolidated_metadata is None


def test_group_model_with_extra_fields_rejected() -> None:
    model = ZarrV3GroupMetadata.create_default(
        extra_fields={"my-extension": {"must_understand": False}}
    )
    with pytest.raises(ValueError, match="extra fields"):
        group_metadata_from_model(model)


@given(attributes=st.dictionaries(st.text(), st.integers() | st.text(), max_size=3))
def test_group_metadata_v3_attribute_fidelity(attributes: dict[str, object]) -> None:
    metadata = GroupMetadata(attributes=attributes, zarr_format=3)
    assert group_metadata_from_model(group_metadata_to_model(metadata)) == metadata
