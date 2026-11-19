from .v2 import ArrayMetadataV2, ArrayV2Metadata, ArrayV2MetadataDict
from .v3 import ArrayMetadataJSON_V3, ArrayMetadataV3, ArrayV3Metadata

ArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3
type ArrayMetadataDict = ArrayV2MetadataDict | ArrayMetadataJSON_V3

__all__ = [
    "ArrayMetadata",
    "ArrayMetadataDict",
    "ArrayMetadataJSON_V3",
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "ArrayV2Metadata",
    "ArrayV2MetadataDict",
    "ArrayV3Metadata",
]
