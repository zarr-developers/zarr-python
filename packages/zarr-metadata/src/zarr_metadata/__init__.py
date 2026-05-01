from zarr_metadata._common import NamedConfig
from zarr_metadata.v2.array import ArrayMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2
from zarr_metadata.v3.array import ArrayMetadataV3
from zarr_metadata.v3.group import GroupMetadataV3

__version__ = "0.1.0"
"""Hardcoded package version. Must match the `version` field in
`pyproject.toml`; the sync is enforced by `tests/test_version.py`."""

ArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3
"""Any Zarr array metadata document (v2 or v3)."""

GroupMetadata = GroupMetadataV2 | GroupMetadataV3
"""Any Zarr group metadata document (v2 or v3)."""


__all__ = [
    "ArrayMetadata",
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "GroupMetadata",
    "GroupMetadataV2",
    "GroupMetadataV3",
    "NamedConfig",
    "__version__",
]
