from typing import TypeAlias, TypeVar

from .v2 import ArrayV2Metadata
from .v3 import ArrayV3Metadata

ArrayMetadata: TypeAlias = ArrayV2Metadata | ArrayV3Metadata
T_ArrayMetadata = TypeVar("T_ArrayMetadata", ArrayV2Metadata, ArrayV3Metadata)

__all__ = [
    "ArrayMetadata",
    "ArrayV2Metadata",
    "ArrayV3Metadata",
]
