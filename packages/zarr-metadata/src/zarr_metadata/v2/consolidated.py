"""Zarr v2 consolidated metadata (`.zmetadata` file).

This module models the de-facto `.zmetadata` file used by the reference
Python implementation of Zarr v2. **This is NOT a spec artifact.** There
is no Zarr v2 specification that defines `.zmetadata`; it is a
canonical-implementation convention.
"""

from collections.abc import Mapping

from typing_extensions import TypedDict

from zarr_metadata.v2.array import ZArrayMetadata
from zarr_metadata.v2.attributes import ZAttrsMetadata
from zarr_metadata.v2.group import ZGroupMetadata


class ZarrV2ConsolidatedMetadataJSON(TypedDict):
    """
    `.zmetadata` file contents.

    The `metadata` map uses flat path keys (`"foo/bar/.zarray"`,
    `"foo/.zattrs"`, etc.) pointing to the JSON contents of the file at
    that path. The keys include the filename suffix, not just the node
    path; the value's shape is determined by which file the key points at:

    - `<path>/.zarray` -> `ZArrayMetadata`
    - `<path>/.zgroup` -> `ZGroupMetadata`
    - `<path>/.zattrs` -> `ZAttrsMetadata`

    The TypedDict cannot discriminate the value shape on the key suffix
    at the type level; consumers should narrow at runtime by inspecting
    `key.endswith(".zarray")` etc.
    """

    zarr_consolidated_format: int
    metadata: Mapping[str, ZArrayMetadata | ZGroupMetadata | ZAttrsMetadata]


__all__ = [
    "ZarrV2ConsolidatedMetadataJSON",
]
