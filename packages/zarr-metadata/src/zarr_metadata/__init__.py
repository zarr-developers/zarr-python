from importlib.metadata import version

from zarr_metadata._common import JSONValue, NamedConfig
from zarr_metadata.v2.array import (
    ArrayDimensionSeparatorV2,
    ArrayMetadataV2,
    ArrayMetadataV2Partial,
    ArrayOrderV2,
    DataTypeMetadataV2,
    ZArrayMetadata,
)
from zarr_metadata.v2.attributes import ZAttrsMetadata
from zarr_metadata.v2.codec import CodecMetadataV2
from zarr_metadata.v2.consolidated import ConsolidatedMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial, ZGroupMetadata
from zarr_metadata.v3._common import MetadataFieldV3
from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial, ExtensionFieldV3
from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial

__version__ = version("zarr-metadata")


__all__ = [
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayMetadataV2Partial",
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ArrayOrderV2",
    "CodecMetadataV2",
    "ConsolidatedMetadataV2",
    "ConsolidatedMetadataV3",
    "DataTypeMetadataV2",
    "ExtensionFieldV3",
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
    "JSONValue",
    "MetadataFieldV3",
    "NamedConfig",
    "ZArrayMetadata",
    "ZAttrsMetadata",
    "ZGroupMetadata",
    "__version__",
]
