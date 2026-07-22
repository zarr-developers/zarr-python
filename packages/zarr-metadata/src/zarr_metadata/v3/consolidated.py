"""Zarr v3 consolidated metadata types.

There is no Zarr v3 specification for consolidated metadata. This module
models the inline-on-group convention used by the reference Python
implementation (and zarrs), where consolidated metadata is embedded as
an extension field on a group's `zarr.json`.

This is a known non-core interoperability extension. Its
`{kind, must_understand, metadata}` payload is an unknown top-level JSON value
to the core document model; implementations that recognize the convention may
interpret it through this dedicated type.
"""

from collections.abc import Mapping
from typing import Literal

from typing_extensions import TypedDict

from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


class ZarrV3ConsolidatedMetadataJSON(TypedDict):
    """
    Inline consolidated metadata embedded in a v3 group.

    The `metadata` map contains only v3 array and group entries - v2
    entries are excluded by design. Mixing v2 entries into a v3
    consolidated metadata document is invalid per spec.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]


__all__ = [
    "ZarrV3ConsolidatedMetadataJSON",
]
