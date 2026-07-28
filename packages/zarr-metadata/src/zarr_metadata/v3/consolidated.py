"""Zarr v3 consolidated metadata types.

There is no Zarr v3 specification for consolidated metadata. This module
models the inline-on-group convention used by the reference Python
implementation (and zarrs), where consolidated metadata is embedded as
an extension field on a group's `zarr.json`.

The shape modeled here (`{kind, must_understand, metadata}` with no `name`
field) reflects the original Zarr v3.0 reading of the extension-field
rules. Under the strict Zarr v3.1 reading, every extension field must
also include a `name: str` key, which would make this shape — and every
real-world consolidated metadata document in the wild — out of spec.
See `ExtensionFieldV3` and
https://github.com/zarr-developers/zarr-specs/issues/371 for the
ongoing discussion.
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
