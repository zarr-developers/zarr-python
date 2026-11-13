"""The types of v3 consolidated metadata: the envelope that the core spec fixes, and the entries as the reference implementation writes them.

This module holds types only and imports no definition.
`zarr_metadata.v3.consolidated` re-exports its names, and holds the
definition that reads a group's `consolidated_metadata` member.
"""

from collections.abc import Mapping
from typing import Final, Literal

from typing_extensions import TypedDict

from zarr_metadata._common import UserJSONValue
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
    """How the entries are carried, `"inline"`: in the group's own document."""
    must_understand: Literal[False]
    """Always `false`: a reader that does not recognize consolidated metadata may ignore it."""
    metadata: Mapping[str, UserJSONValue]
    """The entries, by each node's path from the group, as any JSON."""


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
    """How the entries are carried, `"inline"`: in the group's own document."""
    must_understand: Literal[False]
    """Always `false`: a reader that does not recognize consolidated metadata may ignore it."""
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]
    """The metadata document of each node below the group, by its path from the group.

    A path joins its segments with `"/"`, e.g. `"child/grandchild"`.
    """


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
    "ZarrV3ConsolidatedMetadataEnvelopeJSON",
    "ZarrV3ConsolidatedMetadataJSON",
]
