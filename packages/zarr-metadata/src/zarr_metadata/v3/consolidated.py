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

from collections.abc import Mapping
from typing import Any, Final, Literal

from typing_extensions import TypedDict

from zarr_metadata._common import UserJSONValue
from zarr_metadata.v3._definition import GroupExtraFieldDefinition
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


class ZarrV3ConsolidatedMetadataEnvelopeJSON(TypedDict, closed=True):
    """
    The `consolidated_metadata` member of a v3 group document, as the core spec fixes it.

    The spec allows this one additional field on a group, for historical
    reasons, and fixes its envelope: `kind` is `"inline"`,
    `must_understand` is false, and `metadata` is an object. The spec
    leaves the entries of `metadata` to the reference implementation, so
    here they are any JSON, as user data: the attributes of a listed
    document may hold `NaN`, `Infinity` and `-Infinity`.
    `ZarrV3ConsolidatedMetadataJSON` types the
    entries as the reference implementation writes them: an array or group
    document at each path.

    See https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, UserJSONValue]


class ZarrV3ConsolidatedMetadataJSON(TypedDict, closed=True):
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
