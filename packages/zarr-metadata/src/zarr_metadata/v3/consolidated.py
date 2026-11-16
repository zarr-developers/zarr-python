"""Zarr v3 consolidated metadata types.

There is no Zarr v3 specification for consolidated metadata. This module
models the inline-on-group convention used by the reference Python
implementation (and zarrs), where consolidated metadata is embedded as
an extension field on a group's `zarr.json`.

This is a known non-core interoperability extension. The core spec fixes
only its `{kind, must_understand, metadata}` envelope, which
`ZarrV3ConsolidatedMetadataEnvelopeJSON` types with any JSON as the
entries. `CONSOLIDATED_METADATA_EXTRA_FIELD` reads a group's member as
that envelope, and `CORE_CONTEXT` holds it. `ZarrV3ConsolidatedMetadataJSON` types
the entries as the reference implementation writes them.
"""

from typing import Any, Final

from zarr_metadata.v3._definition import GroupExtraFieldDefinition
from zarr_metadata.v3._types.consolidated import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ZarrV3ConsolidatedMetadataEnvelopeJSON,
    ZarrV3ConsolidatedMetadataJSON,
)

CONSOLIDATED_METADATA_EXTRA_FIELD: Final[GroupExtraFieldDefinition[Any]] = (
    GroupExtraFieldDefinition(
        name=ZARR_V3_CONSOLIDATED_METADATA_KEY,
        configuration=ZarrV3ConsolidatedMetadataEnvelopeJSON,
        finite=False,
    )
)
"""A group's `consolidated_metadata` member, read as the envelope the core spec fixes. The documents it lists are not read: they may hold the non-finite numbers that node attributes may, and `ZarrV3ConsolidatedMetadata` reads them."""


__all__ = [
    "CONSOLIDATED_METADATA_EXTRA_FIELD",
    "ZARR_V3_CONSOLIDATED_METADATA_KEY",
    "ZarrV3ConsolidatedMetadataEnvelopeJSON",
    "ZarrV3ConsolidatedMetadataJSON",
]
