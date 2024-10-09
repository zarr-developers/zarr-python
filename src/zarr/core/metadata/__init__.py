from typing import TypeAlias

from .v2 import ArrayV2Metadata, ArrayV2MetadataDict
from .v3 import ArrayV3Metadata, ArrayV3MetadataDict

ArrayMetadataT: TypeAlias = ArrayV2Metadata | ArrayV3Metadata
ArrayMetadataDictT: TypeAlias = ArrayV2MetadataDict | ArrayV3MetadataDict

__all__ = [
    "ArrayV2Metadata",
    "ArrayV3Metadata",
    "ArrayMetadataT",
    "ArrayMetadataDictT",
    "ArrayV3MetadataDict",
    "ArrayV2MetadataDict",
]
