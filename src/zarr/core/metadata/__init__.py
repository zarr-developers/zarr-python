from typing import TypeAlias, TypeVar

from ..types import ArrayMetadataJSON_V2, ArrayMetadataJSON_V3
from .v2 import ArrayV2Metadata
from .v3 import ArrayV3Metadata

ArrayMetadata: TypeAlias = ArrayV2Metadata | ArrayV3Metadata
ArrayMetadataDict: TypeAlias = ArrayMetadataJSON_V2 | ArrayMetadataJSON_V3
T_ArrayMetadata = TypeVar("T_ArrayMetadata", ArrayV2Metadata, ArrayV3Metadata)

__all__ = [
    "ArrayMetadata",
    "ArrayMetadataDict",
    "ArrayMetadataJSON_V2",
    "ArrayMetadataJSON_V3",
    "ArrayV2Metadata",
    "ArrayV3Metadata",
]
