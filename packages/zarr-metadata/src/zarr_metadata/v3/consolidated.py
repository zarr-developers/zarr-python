"""Zarr v3 consolidated metadata types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3.array import ArrayMetadataV3
    from zarr_metadata.v3.group import GroupMetadataV3


class ConsolidatedMetadataV3(TypedDict):
    """
    Inline consolidated metadata embedded in a v3 group.

    The ``metadata`` map contains only v3 array and group entries - v2
    entries are excluded by design. Mixing v2 entries into a v3
    consolidated metadata document is invalid per spec.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ArrayMetadataV3 | GroupMetadataV3]


__all__ = [
    "ConsolidatedMetadataV3",
]
