from zarr_metadata.common import JSON, NamedConfig
from zarr_metadata.v2.array import ArrayMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2
from zarr_metadata.v3.array import ArrayMetadataV3
from zarr_metadata.v3.group import GroupMetadataV3

ArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3
"""Any Zarr array metadata document (v2 or v3)."""

GroupMetadata = GroupMetadataV2 | GroupMetadataV3
"""Any Zarr group metadata document (v2 or v3)."""


__all__ = [
    "JSON",
    "ArrayMetadata",
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "GroupMetadata",
    "GroupMetadataV2",
    "GroupMetadataV3",
    "NamedConfig",
]
