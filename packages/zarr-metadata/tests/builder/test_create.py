"""Tests for the `create_*` document factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typing_extensions

import zarr_metadata
import zarr_metadata.builder
from zarr_metadata.builder._create import (
    create_zarr_v2_array_metadata_json,
    create_zarr_v2_consolidated_metadata_json,
    create_zarr_v2_group_metadata_json,
    create_zarr_v2_zarray_json,
    create_zarr_v2_zgroup_json,
    create_zarr_v3_array_metadata_json,
    create_zarr_v3_consolidated_metadata_json,
    create_zarr_v3_group_metadata_json,
)
from zarr_metadata.model import MetadataValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    # Factories differ in the document type they return; these tests only
    # ever compare the result as a mapping.
    Factory = Callable[..., Mapping[str, object]]

V3_ARRAY: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}

V2_ZARRAY: dict[str, object] = {
    "zarr_format": 2,
    "shape": (4,),
    "chunks": (2,),
    "dtype": "<i4",
    "compressor": None,
    "fill_value": 0,
    "order": "C",
    "filters": None,
}

# (factory, kwargs, expected result). Every entry must succeed; error paths
# get their own tests below. List-spelled arrays in the input check tuple
# normalization against the tuple-spelled expectation.
CASES: dict[str, tuple[Factory, dict[str, object], dict[str, object]]] = {
    "v3-array": (create_zarr_v3_array_metadata_json, V3_ARRAY, V3_ARRAY),
    "v3-array-normalizes-lists": (
        create_zarr_v3_array_metadata_json,
        {**V3_ARRAY, "shape": [4, 4], "codecs": ["bytes"]},
        V3_ARRAY,
    ),
    "v3-array-with-extensions": (
        create_zarr_v3_array_metadata_json,
        {**V3_ARRAY, "extensions": {"my_ext": {"must_understand": False}}},
        {**V3_ARRAY, "my_ext": {"must_understand": False}},
    ),
    "v3-group": (
        create_zarr_v3_group_metadata_json,
        {"zarr_format": 3, "node_type": "group", "attributes": {"a": None}},
        {"zarr_format": 3, "node_type": "group", "attributes": {"a": None}},
    ),
    "v3-consolidated": (
        create_zarr_v3_consolidated_metadata_json,
        {"kind": "inline", "must_understand": False, "metadata": {"a": V3_ARRAY}},
        {"kind": "inline", "must_understand": False, "metadata": {"a": V3_ARRAY}},
    ),
    "v2-array": (
        create_zarr_v2_array_metadata_json,
        {**V2_ZARRAY, "attributes": {"unit": "m"}},
        {**V2_ZARRAY, "attributes": {"unit": "m"}},
    ),
    "v2-group": (
        create_zarr_v2_group_metadata_json,
        {"zarr_format": 2, "attributes": {"unit": "m"}},
        {"zarr_format": 2, "attributes": {"unit": "m"}},
    ),
    "v2-zarray": (create_zarr_v2_zarray_json, dict(V2_ZARRAY), V2_ZARRAY),
    "v2-zgroup": (create_zarr_v2_zgroup_json, {"zarr_format": 2}, {"zarr_format": 2}),
    "v2-consolidated": (
        create_zarr_v2_consolidated_metadata_json,
        {"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}},
        {"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}},
    ),
}


@pytest.mark.parametrize(("factory", "kwargs", "expected"), CASES.values(), ids=list(CASES))
def test_create(factory: Factory, kwargs: dict[str, object], expected: dict[str, object]) -> None:
    assert factory(**kwargs) == expected


def test_output_shares_no_state_with_arguments() -> None:
    grid: dict[str, object] = {"name": "regular", "configuration": {"chunk_shape": (2, 2)}}
    document = create_zarr_v3_array_metadata_json(**{**V3_ARRAY, "chunk_grid": grid})
    grid["configuration"]["chunk_shape"] = (9, 9)  # caller mutates after the fact
    assert document["chunk_grid"]["configuration"]["chunk_shape"] == (2, 2)


# -- the package rule, enforced ----------------------------------------------

# Public TypedDicts ending in JSON that are field/helper shapes rather than
# documents. Closed by hand, like the naming-grammar vocabulary.
_HELPER_SHAPES = frozenset({"ZarrV3NamedConfigJSON"})

# Every public document TypedDict, mapped to its factory. Kept here rather
# than in the package: the mapping exists only so this test can hold the
# two sets equal.
_FACTORIES: dict[str, Factory] = {
    "ZarrV2ArrayMetadataJSON": create_zarr_v2_array_metadata_json,
    "ZarrV2ConsolidatedMetadataJSON": create_zarr_v2_consolidated_metadata_json,
    "ZarrV2GroupMetadataJSON": create_zarr_v2_group_metadata_json,
    "ZarrV2ZArrayJSON": create_zarr_v2_zarray_json,
    "ZarrV2ZGroupJSON": create_zarr_v2_zgroup_json,
    "ZarrV3ArrayMetadataJSON": create_zarr_v3_array_metadata_json,
    "ZarrV3ConsolidatedMetadataJSON": create_zarr_v3_consolidated_metadata_json,
    "ZarrV3GroupMetadataJSON": create_zarr_v3_group_metadata_json,
}


def test_every_document_typeddict_has_a_factory() -> None:
    documents = {
        name
        for name in zarr_metadata.__all__
        if typing_extensions.is_typeddict(getattr(zarr_metadata, name))
        and name.endswith("JSON")
        and name not in _HELPER_SHAPES
    }
    assert documents == set(_FACTORIES)
    for factory in _FACTORIES.values():
        assert factory.__name__ in zarr_metadata.builder.__all__


# -- error cases, one test per failure mode ----------------------------------


def test_error_v3_array_semantic_rules_run() -> None:
    with pytest.raises(MetadataValidationError, match=r"\[0, 255\]"):
        create_zarr_v3_array_metadata_json(**{**V3_ARRAY, "fill_value": 300})


def test_error_v3_array_structural_garbage_from_untyped_caller() -> None:
    with pytest.raises(MetadataValidationError) as info:
        create_zarr_v3_array_metadata_json(zarr_format="3")  # type: ignore[arg-type]
    assert {p.loc[0] for p in info.value.problems if p.kind == "missing_key"} >= {
        "node_type",
        "shape",
    }


def test_error_v3_array_extension_shadows_standard_key() -> None:
    with pytest.raises(MetadataValidationError, match="standard metadata key"):
        create_zarr_v3_array_metadata_json(**V3_ARRAY, extensions={"shape": (9,)})


def test_error_v3_group_extension_shadows_standard_key() -> None:
    with pytest.raises(MetadataValidationError, match="standard metadata key"):
        create_zarr_v3_group_metadata_json(
            zarr_format=3, node_type="group", extensions={"attributes": {}}
        )


def test_error_v3_consolidated_invalid() -> None:
    with pytest.raises(MetadataValidationError):
        create_zarr_v3_consolidated_metadata_json(
            kind="inline",
            must_understand=True,
            metadata={},  # type: ignore[typeddict-item]
        )


def test_error_v3_consolidated_child_violates_composition_rules() -> None:
    child = {**V3_ARRAY, "fill_value": 300}
    with pytest.raises(MetadataValidationError) as exc_info:
        create_zarr_v3_consolidated_metadata_json(
            kind="inline", must_understand=False, metadata={"a": child}
        )
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        (("metadata", "a", "fill_value"), "invalid_value")
    ]


def test_error_v2_array_structural() -> None:
    with pytest.raises(MetadataValidationError):
        create_zarr_v2_array_metadata_json(**{**V2_ZARRAY, "order": "K"})  # type: ignore[typeddict-item]


def test_error_v3_array_malformed_raw_dtype() -> None:
    # r<N> names outside the family grammar are misspellings of a known
    # family, not unknown extensions, and must not escape judgment.
    with pytest.raises(MetadataValidationError, match="positive multiple of 8"):
        create_zarr_v3_array_metadata_json(**{**V3_ARRAY, "data_type": "r12", "fill_value": (1,)})


def test_error_v2_zarray_attributes_via_splat() -> None:
    # The signature excludes `attributes` statically, but a splatted call
    # bypasses that; the runtime backstop must hold the strict shape.
    with pytest.raises(MetadataValidationError, match=".zattrs"):
        create_zarr_v2_zarray_json(**{**V2_ZARRAY, "attributes": {"unit": "m"}})


def test_error_v2_zgroup_attributes_via_splat() -> None:
    splatted: dict[str, object] = {"zarr_format": 2, "attributes": {"unit": "m"}}
    with pytest.raises(MetadataValidationError, match=".zattrs"):
        create_zarr_v2_zgroup_json(**splatted)


def test_error_v2_consolidated_envelope() -> None:
    with pytest.raises(MetadataValidationError, match="expected a mapping"):
        create_zarr_v2_consolidated_metadata_json(
            zarr_consolidated_format=1,
            metadata="not a mapping",  # type: ignore[typeddict-item]
        )


def test_error_v2_consolidated_format_is_not_one() -> None:
    with pytest.raises(MetadataValidationError) as exc_info:
        create_zarr_v2_consolidated_metadata_json(
            zarr_consolidated_format=2,
            metadata={},
        )
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        (("zarr_consolidated_format",), "invalid_value")
    ]


def test_error_v2_consolidated_array_entry_is_invalid() -> None:
    with pytest.raises(MetadataValidationError) as exc_info:
        create_zarr_v2_consolidated_metadata_json(
            zarr_consolidated_format=1,
            metadata={"foo/.zarray": {}},  # type: ignore[typeddict-item]
        )
    assert any(
        problem.loc[:2] == ("metadata", "foo/.zarray") and problem.kind == "missing_key"
        for problem in exc_info.value.problems
    )


def test_error_v2_consolidated_entry_has_unknown_suffix() -> None:
    with pytest.raises(MetadataValidationError, match="metadata file suffix"):
        create_zarr_v2_consolidated_metadata_json(
            zarr_consolidated_format=1,
            metadata={"foo/data": {}},  # type: ignore[typeddict-item]
        )
