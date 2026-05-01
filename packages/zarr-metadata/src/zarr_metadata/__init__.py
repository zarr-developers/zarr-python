from zarr_metadata._common import NamedConfig
from zarr_metadata.v2.array import (
    ArrayDimensionSeparatorV2,
    ArrayMetadataV2,
    ArrayOrderV2,
    DataTypeMetadataV2,
)
from zarr_metadata.v2.codec import CodecMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2
from zarr_metadata.v3._common import MetadataFieldV3
from zarr_metadata.v3.array import ArrayMetadataV3, ExtraFieldV3
from zarr_metadata.v3.group import GroupMetadataV3

__version__ = "0.1.0"
"""Hardcoded package version. Must match the `version` field in
`pyproject.toml`; the sync is enforced by `tests/test_version.py`."""


__all__ = [
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "ArrayOrderV2",
    "CodecMetadataV2",
    "DataTypeMetadataV2",
    "ExtraFieldV3",
    "GroupMetadataV2",
    "GroupMetadataV3",
    "MetadataFieldV3",
    "NamedConfig",
    "__version__",
]
