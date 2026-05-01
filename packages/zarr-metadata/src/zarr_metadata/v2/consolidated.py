"""Zarr v2 consolidated metadata (`.zmetadata` file).

This module models the de-facto `.zmetadata` file used by the reference
Python implementation of Zarr v2. **This is NOT a spec artifact.** There
is no Zarr v2 specification that defines `.zmetadata`; it is a
canonical-implementation convention.
"""

from collections.abc import Mapping
from typing import TypedDict

from zarr_metadata.v2.array import ArrayMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2


class ConsolidatedMetadataV2(TypedDict):
    """
    `.zmetadata` file contents.

    The `metadata` map uses flat path keys (`"foo/bar/.zarray"`,
    `"foo/.zattrs"`, etc.) pointing to the JSON contents of the file at
    that path. The keys include the filename suffix, not just the node path.
    """

    zarr_consolidated_format: int
    metadata: Mapping[str, GroupMetadataV2 | ArrayMetadataV2]


__all__ = [
    "ConsolidatedMetadataV2",
]
