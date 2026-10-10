"""Models of the metadata documents that describe Zarr arrays and groups.

`zarr.Array.metadata` is an `ArrayV2Metadata` or an `ArrayV3Metadata`, and
`zarr.Group.metadata` is a `GroupMetadata`. Construct these objects with `from_dict`.
"""

from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayMetadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

__all__ = [
    "ArrayMetadata",
    "ArrayV2Metadata",
    "ArrayV3Metadata",
    "ConsolidatedMetadata",
    "GroupMetadata",
]
