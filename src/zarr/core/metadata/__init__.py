from typing import TypeAlias, TypeVar

from ._v2 import ArrayV2Metadata, ArrayV2MetadataDict
from ._v3 import ArrayV3Metadata, ArrayV3MetadataDict, DataType

ArrayMetadata: TypeAlias = ArrayV2Metadata | ArrayV3Metadata
ArrayMetadataDict: TypeAlias = ArrayV2MetadataDict | ArrayV3MetadataDict
T_ArrayMetadata = TypeVar("T_ArrayMetadata", ArrayV2Metadata, ArrayV3Metadata)

__all__ = [
    "ArrayMetadata",
    "ArrayMetadataDict",
    "ArrayV2Metadata",
    "ArrayV2MetadataDict",
    "ArrayV3Metadata",
    "ArrayV3MetadataDict",
    "DataType",
]
