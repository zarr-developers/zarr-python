"""Zarr v3 consolidated metadata types.

There is no Zarr v3 specification for consolidated metadata. This module
models the inline-on-group convention used by the reference Python
implementation, where consolidated metadata is embedded as an extra field
on a group's `zarr.json`.
"""

from collections.abc import Mapping
from typing import Literal

from typing_extensions import TypedDict

from zarr_metadata.v3.array import ArrayMetadataV3
from zarr_metadata.v3.group import GroupMetadataV3


class ConsolidatedMetadataV3(TypedDict):
    """
    Inline consolidated metadata embedded in a v3 group.

    The `metadata` map contains only v3 array and group entries - v2
    entries are excluded by design. Mixing v2 entries into a v3
    consolidated metadata document is invalid per spec.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ArrayMetadataV3 | GroupMetadataV3]


__all__ = [
    "ConsolidatedMetadataV3",
]
