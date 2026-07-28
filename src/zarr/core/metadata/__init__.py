from .v2 import ArrayV2Metadata, ArrayV2MetadataDict
from .v3 import ArrayMetadataJSON_V3, ArrayV3Metadata

ArrayMetadata = ArrayV2Metadata | ArrayV3Metadata
type ArrayMetadataDict = ArrayV2MetadataDict | ArrayMetadataJSON_V3

__all__ = [
    "ArrayMetadata",
    "ArrayMetadataDict",
    "ArrayMetadataJSON_V3",
    "ArrayV2Metadata",
    "ArrayV2MetadataDict",
    "ArrayV3Metadata",
]
