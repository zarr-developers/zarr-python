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
from typing import Final, Literal

from typing_extensions import TypedDict

from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


class ZarrV3ConsolidatedMetadataJSON(TypedDict):
    """
    Inline consolidated metadata embedded in a v3 group.

    The `metadata` map contains only v3 array and group entries. V2 entries
    are excluded from this interoperability convention by design; the v3 core
    specification does not define consolidated metadata.
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]


ZARR_V3_CONSOLIDATED_METADATA_KEY: Final = "consolidated_metadata"
"""The key under which consolidated metadata is embedded in a v3 group document.

Unlike the v2 `.zmetadata` file, this is not a store key: consolidated metadata
is carried as an extension field inside the group's own `zarr.json`. Like its v2
counterpart it is a reference-implementation convention, not a spec artifact.
"""


__all__ = [
    "ZARR_V3_CONSOLIDATED_METADATA_KEY",
    "ZarrV3ConsolidatedMetadataJSON",
]
