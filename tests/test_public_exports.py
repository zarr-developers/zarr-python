"""Public import paths for types that the public API returns or passes to extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import zarr.abc.codec
import zarr.metadata
from zarr.core.array_spec import ArraySpec
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayMetadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from zarr.types import ZarrFormat


@pytest.mark.parametrize(
    ("public", "private"),
    [
        (zarr.metadata.ArrayMetadata, ArrayMetadata),
        (zarr.metadata.ArrayV2Metadata, ArrayV2Metadata),
        (zarr.metadata.ArrayV3Metadata, ArrayV3Metadata),
        (zarr.metadata.ConsolidatedMetadata, ConsolidatedMetadata),
        (zarr.metadata.GroupMetadata, GroupMetadata),
        (zarr.abc.codec.ArraySpec, ArraySpec),
    ],
)
def test_public_path_is_the_implementation(public: Any, private: Any) -> None:
    """Each public name is the object zarr itself uses, so isinstance checks agree."""
    assert public is private


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_and_group_metadata_types(zarr_format: ZarrFormat) -> None:
    """The metadata of arrays and groups is an instance of a public metadata type."""
    store: dict[str, Any] = {}
    group = zarr.group(store=store, zarr_format=zarr_format)
    array = group.create_array("a", shape=(2,), dtype="i4")
    expected = zarr.metadata.ArrayV2Metadata if zarr_format == 2 else zarr.metadata.ArrayV3Metadata
    assert isinstance(array.metadata, expected)
    assert isinstance(group.metadata, zarr.metadata.GroupMetadata)
