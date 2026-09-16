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
    are excluded from this interoperability convention by design. The v3 core
    specification acknowledges `consolidated_metadata` as a historical
    additional field and fixes this envelope, but leaves the entries to the
    reference implementation:
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]


ZARR_V3_CONSOLIDATED_METADATA_KEY: Final = "consolidated_metadata"
"""The key under which consolidated metadata is embedded in a v3 group document.

Unlike the v2 `.zmetadata` file, this is not a store key: consolidated metadata
is carried as an additional field inside the group's own `zarr.json`. The core
spec names the field and its envelope ("For historical reasons, group metadata
documents may contain an additional field named ``consolidated_metadata``");
the entry format, like the v2 counterpart, is a reference-implementation
convention.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816
"""


__all__ = [
    "ZARR_V3_CONSOLIDATED_METADATA_KEY",
    "ZarrV3ConsolidatedMetadataJSON",
]
